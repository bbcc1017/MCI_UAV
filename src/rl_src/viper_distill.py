"""Phase 3d — VIPER 트리 증류 (Bastani et al., NeurIPS 2018).

학습된 MaskablePPO(오라클)를 **해석가능 결정트리**로 증류한다. 기존 1-shot distill_policy.py
보다 충실도 높은 작은 트리를 얻기 위해 논문 Algorithm 1(반복 Q-DAGGER)을 구현한다.

알고리즘 (Algorithm 1):
  D←∅, π̂_0←오라클
  for i in 1..N:
    현재 트리 π̂_{i-1} 로 M개 에피소드 롤아웃 → 방문 상태 수집 (i=1 은 오라클로 롤아웃)
    각 상태 s: 오라클 라벨 a*=π*(s), 가중치 w(s)=criticality
    D ← D ∪ {(s, a*, w)}
    π̂_i ← DecisionTree.fit(D, sample_weight=w)   # CART, 가중 = 리샘플과 동치
  return CV 보상 기준 best π̂

가중치(criticality) ℓ̃(s) = V*(s) − min_a Q*(s,a). MaskablePPO 는 명시적 Q 가 없으므로
논문(§2)대로 max-entropy 대용 Q(s,a)=log π*(s,a) 사용 →
  --crit loggap   : max_a logπ − min_a logπ  (마스킹된 valid action; 기본·논문 충실)
  --crit probmargin: top1 − top2 확률
  --crit uniform  : 1 (plain DAGGER baseline)

obs/env: HospitalFeatureWrapper(특징 obs). 정보수준 ablation 은 MCI_OBS_VARIANT=local/comms
로 토글(오라클 PPO 도 동일 variant 로 학습돼 있어야 함).

재사용: evaluate.eval_policy/ppo_policy, make_codec(distill_policy), sklearn DecisionTree.

예:
  MCI_REDUCED_OBS=1 python src/rl_src/viper_distill.py \\
    --manifest scenarios/manifests/plan1nat_manifest.json \\
    --model results/rl/ppo_feature/final_model.zip \\
    --n_iter 5 --rollout_eps 10 --max_depth 8 --crit loggap \\
    --heur_csv results/plan1nat_f3_eval.csv --out_dir results/viper
"""
import argparse
import contextlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch as th
from sklearn.tree import DecisionTreeClassifier, export_text

from env_factory import make_base_env
from hospital_feature_wrapper import HospitalFeatureWrapper
from evaluate import eval_policy, ppo_policy
from distill_policy import make_codec


@contextlib.contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


# ---------- feature env factory (HospitalFeatureWrapper) ----------
def make_feature_env(config_path: str):
    """eval_policy 용 env_factory(seed)->env (특징 obs, eval_mode)."""
    if config_path.endswith(".json"):
        from train_ppo_feature import FeatureMultiRegionEnv

        def _f(seed: int = 0):
            return FeatureMultiRegionEnv(config_path, seed=seed, eval_mode=True)
        return _f

    def _f(seed: int = 0):
        base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=True)
        return HospitalFeatureWrapper(base)
    return _f


# ---------- 오라클 criticality 가중치 ----------
def _masked_probs(model, obs, mask):
    obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32), device=model.device).unsqueeze(0)
    mask_t = th.as_tensor(np.asarray(mask, dtype=bool), device=model.device).unsqueeze(0)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask_t)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()
    return probs  # (n_actions,), invalid ≈ 0


def make_weight_fn(model, crit: str):
    def fn(obs, mask):
        if crit == "uniform":
            return 1.0
        p = _masked_probs(model, obs, mask)
        pv = p[np.asarray(mask, dtype=bool)]
        pv = pv[pv > 1e-12]
        if pv.size < 2:
            return 1.0
        if crit == "loggap":
            return float(np.log(pv.max()) - np.log(pv.min()))  # = max logπ − min logπ
        if crit == "probmargin":
            s = np.sort(pv)[::-1]
            return float(s[0] - s[1])
        return 1.0
    return fn


# ---------- 트리 정책 (마스크 준수 masked-argmax) ----------
def make_tree_policy(tree):
    classes = tree.classes_

    def fn(obs, mask, env=None):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        proba = tree.predict_proba(obs)[0]
        order = np.argsort(-proba)
        m = np.asarray(mask, dtype=bool)
        for j in order:
            a = int(classes[j])
            if a < len(m) and m[a]:
                return a
        v = np.flatnonzero(m)
        return int(v[0]) if v.size else 0
    return fn


# ---------- 롤아웃: 방문 상태(obs, mask) 수집 ----------
def rollout_states(env_factory, policy_fn, n_episodes, seed_base):
    obs_list, mask_list = [], []
    for ep in range(n_episodes):
        env = env_factory(seed=seed_base + ep)
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            mask = env.action_masks()
            obs_list.append(np.asarray(obs, dtype=np.float32))
            mask_list.append(np.asarray(mask, dtype=bool))
            a = policy_fn(obs, mask, env.unwrapped)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    return obs_list, mask_list


# ---------- VIPER 메인 루프 ----------
def viper(env_factory, model, n_iter, rollout_eps, eval_eps, max_depth,
          min_samples_leaf, crit, seed_base):
    oracle = ppo_policy(model)
    weight_fn = make_weight_fn(model, crit)
    D_obs, D_act, D_w = [], [], []
    best = {"tree": None, "reward": -np.inf, "iter": -1}
    history = []

    for i in range(n_iter):
        roll_policy = oracle if i == 0 else make_tree_policy(best_iter_tree)
        with _suppress_stdout():
            obs_l, mask_l = rollout_states(env_factory, roll_policy, rollout_eps, seed_base + 100 * i)
        # 오라클 라벨 + 가중치
        for s, mk in zip(obs_l, mask_l):
            a_star = oracle(s, mk, None)
            D_obs.append(s); D_act.append(a_star); D_w.append(weight_fn(s, mk))
        X = np.asarray(D_obs); y = np.asarray(D_act); w = np.asarray(D_w, dtype=np.float64)
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                      random_state=0)
        tree.fit(X, y, sample_weight=w)
        best_iter_tree = tree
        with _suppress_stdout():
            m = eval_policy(env_factory, make_tree_policy(tree), eval_eps, seed_base + 9000)
        rew = m["mean_R"]
        # 충실도: 누적 D 에서 오라클 라벨 재현율
        fidelity = float((tree.predict(X) == y).mean())
        history.append({"iter": i, "reward": rew, "fidelity": fidelity,
                        "leaves": int(tree.get_n_leaves()), "depth": int(tree.get_depth()),
                        "n_data": len(D_obs)})
        sys.stderr.write(f"[VIPER it{i}] reward={rew:.2f} fidelity={fidelity:.3f} "
                         f"leaves={tree.get_n_leaves()} depth={tree.get_depth()} n={len(D_obs)}\n")
        sys.stderr.flush()
        if rew > best["reward"]:
            best = {"tree": tree, "reward": rew, "iter": i}
    return best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help=".json 매니페스트(멀티지역) — 단일 config 면 --config_path")
    ap.add_argument("--config_path", default=None)
    ap.add_argument("--model", required=True, help="오라클 MaskablePPO(.zip), 특징 obs 로 학습된 것")
    ap.add_argument("--n_iter", type=int, default=5)
    ap.add_argument("--rollout_eps", type=int, default=10)
    ap.add_argument("--eval_eps", type=int, default=20)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_samples_leaf", type=int, default=20)
    ap.add_argument("--crit", choices=["loggap", "probmargin", "uniform"], default="loggap")
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--heur_csv", default=None, help="지역별 best 휴리스틱 룰 CSV(있으면 비교)")
    ap.add_argument("--out_dir", default="results/viper")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    src = args.manifest or args.config_path
    if not src:
        raise SystemExit("--manifest 또는 --config_path 필요")

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)

    # 트리 학습용 롤아웃 env (manifest 면 멀티지역, 아니면 단일)
    train_factory = make_feature_env(src)
    print(f"[VIPER] crit={args.crit} n_iter={args.n_iter} rollout_eps={args.rollout_eps} "
          f"max_depth={args.max_depth} variant={os.environ.get('MCI_OBS_VARIANT','(full)')}")
    best, history = viper(train_factory, model, args.n_iter, args.rollout_eps, args.eval_eps,
                          args.max_depth, args.min_samples_leaf, args.crit, args.seed_base)
    tree = best["tree"]
    print(f"\n[VIPER] best iter={best['iter']} reward={best['reward']:.2f} "
          f"leaves={tree.get_n_leaves()} depth={tree.get_depth()}")

    # 저장 (pkl + 규칙 텍스트)
    tag = f"viper_{args.crit}_d{args.max_depth}"
    with open(os.path.join(args.out_dir, f"{tag}.pkl"), "wb") as f:
        pickle.dump({"tree": tree, "history": history, "crit": args.crit}, f)
    with open(os.path.join(args.out_dir, f"{tag}_rules.txt"), "w", encoding="utf-8") as f:
        f.write(export_text(tree, max_depth=min(args.max_depth, 10)))
    print(f"[저장] {args.out_dir}/{tag}.pkl, {tag}_rules.txt")

    # 17지역 비교 (manifest + heur_csv 있으면)
    if args.manifest and args.heur_csv and os.path.exists(args.heur_csv):
        import pandas as pd
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        heur = pd.read_csv(args.heur_csv)
        from distill_policy import make_heuristic_policy
        best_rule = dict(zip(heur["region"], heur.get("heuristic_rule", heur.iloc[:, 1])))
        tree_policy = make_tree_policy(tree)
        rows = []
        for ri, (region, cfg) in enumerate(manifest.items()):
            ef = make_feature_env(cfg)
            with _suppress_stdout():
                m_ppo = eval_policy(ef, ppo_policy(model), args.eval_eps, args.seed_base)
                m_tree = eval_policy(ef, tree_policy, args.eval_eps, args.seed_base)
                m_heur = None
                if region in best_rule:
                    # 휴리스틱은 dict-obs 룰이라 특징 obs 와 무관(env.unwrapped 사용) → 그대로 호환
                    m_heur = eval_policy(ef, make_heuristic_policy(best_rule[region]), args.eval_eps, args.seed_base)
            row = {"region": region, "PPO_R": m_ppo["mean_R"], "tree_R": m_tree["mean_R"],
                   "tree_vs_PPO": m_tree["mean_R"] - m_ppo["mean_R"]}
            if m_heur:
                row["heur_R"] = m_heur["mean_R"]
                row["tree_vs_heur"] = m_tree["mean_R"] - m_heur["mean_R"]
            rows.append(row)
            sys.stderr.write(f"[{ri+1}/{len(manifest)}] {region}: PPO={row['PPO_R']:.2f} "
                             f"tree={row['tree_R']:.2f}\n"); sys.stderr.flush()
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_dir, f"{tag}_region_eval.csv"), index=False)
        print("\n=== VIPER 트리 vs PPO(vs 휴리스틱) 17지역 ===")
        print(df.to_string(index=False))
        if "tree_vs_heur" in df and "PPO_R" in df and "heur_R" in df:
            ppo_margin = (df["PPO_R"] - df["heur_R"]).mean()
            tree_margin = df["tree_vs_heur"].mean()
            print(f"\nPPO margin(vs heur) {ppo_margin:+.2f} | tree margin {tree_margin:+.2f} | "
                  f"margin 유지율 {tree_margin/ppo_margin*100:.0f}%" if abs(ppo_margin) > 1e-6 else "")


if __name__ == "__main__":
    main()
