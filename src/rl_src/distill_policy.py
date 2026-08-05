"""RL 정책을 해석가능 규칙으로 증류하고 17지역 성능을 재검증한다 (피드백 #4).

두 가지 증류:
  (1) 충실 트리(faithful surrogate): obs221 → RL action(int) DecisionTree depth 스윕.
      깊이별 action-match fidelity 곡선 → "RL 을 트리로 얼마나 재현 가능한가".
  (2) 합성 해석규칙(discovered policy) — analyze_policy 발견 3종 결합:
      · 우선순위: 축 A 트리 (R·Y 동시 대기 시 Red/Yellow)
      · 이송수단: 'AMB 우선, 현장 구급차 0 → UAV' (축 B, fidelity 99.5%)
      · 목적지:  유효 병원 중 최소혼잡(n_occupied 최소) — 부하분산 (축 C)
  → 합성정책을 policy_fn 으로 만들어 eval_policy 로 휴리스틱·풀 PPO 와 동일 시드 비교.

출력: results/plan1nat_f3_distill_eval.csv, results/rl/plan1nat_f3/distill_priority_tree.pkl

사용:
  MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/distill_policy.py \
    --manifest scenarios/manifests/plan1nat_manifest.json \
    --model results/rl/plan1nat_f3/national/ppo/final_model.zip \
    --heur_csv results/plan1nat_f3_eval.csv --n_episodes 100
"""
import argparse
import contextlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from evaluate import eval_policy, make_eval_env, ppo_policy

SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import Universal_Rule  # noqa: E402
from ShinHeuristics import ShinHeuristicRule  # noqa: E402


@contextlib.contextmanager
def _suppress_stdout():
    """sim_src print 폭발을 /dev/null 로 우회 (stderr 진행로그는 유지)."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def parse_rule(rule_name):
    p = [x.strip() for x in rule_name.split(",")]
    if len(p) == 2 and p[0].startswith("Shin ") and p[1].startswith("Mode "):
        return ShinHeuristicRule(
            p[0].replace("Shin ", "", 1).strip(),
            p[1].replace("Mode ", "", 1).strip(),
        )
    if len(p) != 4:
        raise ValueError(f"알 수 없는 휴리스틱 규칙명: {rule_name}")
    return Universal_Rule(p[0], p[1], p[2].replace("Red", "", 1).strip(),
                          p[3].replace("Yellow", "", 1).strip())


def make_codec(H):
    """plan1nat: AMB·UAV 둘 다 있어 mode 자유 → 표준 3D 평탄화."""
    n_dest, n_mode = H + 1, 2

    def decode(a):
        a = int(a)
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        return c, rem // n_mode, rem % n_mode

    def encode(c, d, m):
        return int(c) * (n_dest * n_mode) + int(d) * n_mode + int(m)

    return decode, encode


# ---------------------------------------------------------------- policies
def make_heuristic_policy(rule_name, policy_seed=0):
    """규칙정책을 만든다.

    ``policy_seed``는 Threshold/2Step의 0.5 병원 선택처럼 정책 내부에만 쓰며,
    시뮬레이션 동역학 RNG와 별도 generator를 유지한다. 기본값 0은 기존 호출과
    결과를 그대로 보존하고, 전국 paired 평가는 episode seed에서 파생한 값을 넘긴다.
    """
    rule = parse_rule(rule_name)
    state = {"init": False, "codec": None}

    def fn(obs, mask, env):
        if not state["init"]:
            # (v6) 패딩 env 는 mask 레이아웃(H_pad)과 sim 실H 가 다르다 — encode 는 반드시
            # 레이아웃 코덱(마스크 길이 유도). 실H 코덱이면 flat 인덱스 오정렬로 mask[a]
            # 상시 False → 첫 유효행동 폴백만 반복(자연-H 판정 heur PDR 0.89 붕괴 원인).
            # RuleManager 의 dest(1..실H)는 레이아웃의 접두 구간이라 그대로 유효.
            mode_free = (int(getattr(env, "amb_num", 0)) > 0
                         and int(getattr(env, "uav_num", 0)) > 0)
            H_layout = len(mask) // 4 - 1 if mode_free else len(mask) // 2 - 1
            from loadbalance_heuristic import _codec_from_mask
            state["codec"] = (None, _codec_from_mask(len(mask), H_layout))
            rule.set_seed(np.random.default_rng(policy_seed))
            rule.init_with_scenario({"EntityManager": env.en_manager})
            state["init"] = True
        _, encode = state["codec"]
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        c, d, m = rule.select(dobs)
        a = encode(0, 0, 0) if c < 0 else encode(c, d, m)
        if a < len(mask) and mask[a]:
            return a
        v = np.flatnonzero(mask)
        return int(v[0]) if v.size else 0

    return fn


def make_composed_policy(priority_tree, labels):
    """축 A 트리 + AMB우선(축B) + 최소혼잡 라우팅(축C)."""
    lab_idx = {c: i for i, c in enumerate(labels)}
    n_amb_idx = lab_idx["n_amb_at_site"]
    occ_idx = {int(c[1:-4]): i for c, i in lab_idx.items()
               if c.startswith("h") and c.endswith("_occ")}

    def fn(obs, mask, env):
        obs = np.asarray(obs, dtype=np.float32)
        H = int(env.en_manager.en_properties["hospital"]["hos_num"])
        decode, encode = make_codec(H)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        dec = np.array([decode(int(a)) for a in valid])
        # ★ 'class selectable' ≠ 'transportable': dest=0(STAY) 때문에 환자 없는 클래스도
        #   valid 로 잡힌다. 이송(dest>0) 액션만으로 판정해야 STAY 붕괴를 피한다.
        trans = dec[dec[:, 1] > 0]
        if trans.size == 0:
            a = encode(0, 0, 0)  # 이송 불가 → 대기 (차량 복귀/추가 구조 진행)
            return a if (a < len(mask) and mask[a]) else int(valid[0])
        tclasses = set(trans[:, 0].tolist())
        # 1. 우선순위: 이송 가능한 R/Y 중에서
        ry = [c for c in (0, 1) if c in tclasses]
        if len(ry) == 2:
            pred_red = int(priority_tree.predict(obs.reshape(1, -1))[0])  # 1=Red
            chosen_c = 0 if pred_red == 1 else 1
        elif len(ry) == 1:
            chosen_c = ry[0]
        else:
            chosen_c = min(tclasses)  # R/Y 없으면 최저 등급(주로 Green) 이송
        # 2. 이송수단: AMB 우선, 현장 구급차 0 → UAV
        cmodes = set(trans[trans[:, 0] == chosen_c, 2].tolist())
        pref = 0 if obs[n_amb_idx] > 0.5 else 1
        chosen_m = pref if pref in cmodes else min(cmodes)
        # 3. 목적지: 유효 이송병원 중 최소혼잡 (부하분산)
        cand = trans[(trans[:, 0] == chosen_c) & (trans[:, 2] == chosen_m)]
        dests = [int(d) for d in cand[:, 1]]
        chosen_d = min(dests, key=lambda d: obs[occ_idx.get(d - 1, n_amb_idx)])
        a = encode(chosen_c, chosen_d, chosen_m)
        if a < len(mask) and mask[a]:
            return a
        return int(valid[0])

    return fn


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--heur_csv", required=True)
    ap.add_argument("--tag", default="plan1nat_f3")
    ap.add_argument("--analysis_dir", default="results/analysis")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--out_csv", default="results/plan1nat_f3_distill_eval.csv")
    args = ap.parse_args()

    # ---- 데이터 로드 ----
    obs = np.load(os.path.join(args.analysis_dir, f"decisions_{args.tag}.npz"))["obs"]
    meta = pd.read_csv(os.path.join(args.analysis_dir, f"decisions_{args.tag}_meta.csv"))
    with open(os.path.join(args.analysis_dir, f"decisions_{args.tag}_labels.json"), encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    # ---- (1) 충실 트리 fidelity 스윕: obs → RL action ----
    rl_action = (meta["rl_class"] * 0).astype(int)  # placeholder; reconstruct action int per region H
    # action int 은 지역마다 H 동일(fixed_hos_num)이라 decode 역연산으로 복원
    H_global = sum(1 for c in labels if c.startswith("h") and c.endswith("_occ"))
    _, encode_g = make_codec(H_global)
    y_action = np.array([encode_g(c, d, m) for c, d, m in
                         zip(meta["rl_class"], meta["rl_dest"], meta["rl_mode"])])
    Xtr, Xte, ytr, yte = train_test_split(obs, y_action, test_size=0.3, random_state=0)
    print("=== (1) 충실 트리 fidelity (obs221 → RL action) ===")
    for depth in [4, 6, 8, 12, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
        print(f"  depth={str(depth):>4}: train={t.score(Xtr,ytr):.3f}  test={t.score(Xte,yte):.3f}  "
              f"leaves={t.get_n_leaves()}")

    # ---- (2) 합성 정책: 우선순위 트리 학습 ----
    both = (pd.DataFrame(obs, columns=labels)["atsite_Red"] > 0) & \
           (pd.DataFrame(obs, columns=labels)["atsite_Yellow"] > 0)
    selA = both.values & meta["rl_class"].isin([0, 1]).values
    yA = (meta.loc[selA, "rl_class"] == 0).astype(int).values  # 1=Red
    priority_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=0)
    priority_tree.fit(obs[selA], yA)
    tree_dir = f"results/rl/{args.tag}"  # tag-aware: 멀티시드 동시 실행 충돌 방지
    os.makedirs(tree_dir, exist_ok=True)
    with open(f"{tree_dir}/distill_priority_tree.pkl", "wb") as f:
        pickle.dump({"tree": priority_tree, "labels": labels}, f)
    print(f"\n=== (2) 합성 정책: 우선순위 트리 depth6 fidelity={priority_tree.score(obs[selA],yA):.3f} "
          f"(n={selA.sum()}) ===")

    # ---- 17지역 평가: 휴리스틱-best / 풀 PPO / 합성정책 (동일 시드) ----
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    heur = pd.read_csv(args.heur_csv)
    best_rule = dict(zip(heur["region"], heur["heuristic_rule"]))

    rows = []
    for ri, (region, cfg) in enumerate(manifest.items()):
        if region not in best_rule:
            continue
        sys.stderr.write(f"[{ri+1}/{len(manifest)}] {region} eval ...\n"); sys.stderr.flush()
        ef = make_eval_env(cfg)
        with _suppress_stdout():
            m_heur = eval_policy(ef, make_heuristic_policy(best_rule[region]), args.n_episodes, args.seed_base)
            m_ppo = eval_policy(ef, ppo_policy(model), args.n_episodes, args.seed_base)
            m_dist = eval_policy(ef, make_composed_policy(priority_tree, labels), args.n_episodes, args.seed_base)
        rows.append({
            "region": region,
            "heur_R": m_heur["mean_R"], "PPO_R": m_ppo["mean_R"], "distill_R": m_dist["mean_R"],
            "heur_R_woG": m_heur["mean_R_woG"], "PPO_R_woG": m_ppo["mean_R_woG"],
            "distill_R_woG": m_dist["mean_R_woG"],
            "PPO_vs_heur": m_ppo["mean_R"] - m_heur["mean_R"],
            "distill_vs_heur": m_dist["mean_R"] - m_heur["mean_R"],
            "distill_vs_PPO": m_dist["mean_R"] - m_ppo["mean_R"],
        })
        r = rows[-1]
        sys.stderr.write(f"    heur={r['heur_R']:.2f} PPO={r['PPO_R']:.2f} distill={r['distill_R']:.2f} "
                         f"(distill vs heur {r['distill_vs_heur']:+.2f}, vs PPO {r['distill_vs_PPO']:+.2f})\n")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n=== 증류 정책 17지역 성능 (n_episodes={}) ===".format(args.n_episodes))
    print(df[["region", "heur_R", "PPO_R", "distill_R", "PPO_vs_heur", "distill_vs_heur", "distill_vs_PPO"]]
          .to_string(index=False))
    print(f"\n평균: 휴리스틱 {df['heur_R'].mean():.2f} | 풀PPO {df['PPO_R'].mean():.2f} "
          f"({df['PPO_vs_heur'].mean():+.2f}) | 증류 {df['distill_R'].mean():.2f} "
          f"({df['distill_vs_heur'].mean():+.2f})")
    print(f"증류가 휴리스틱 추월: {(df['distill_vs_heur']>0).sum()}/{len(df)} 지역 | "
          f"풀 PPO 마진 유지율: {df['distill_vs_heur'].mean()/df['PPO_vs_heur'].mean()*100:.0f}%")
    print(f"[저장: {args.out_csv}]", file=sys.stderr)


if __name__ == "__main__":
    main()
