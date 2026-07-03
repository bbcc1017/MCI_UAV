"""VIPER 통신-계층 버전 증류 — 보수(현장정보만) → 진보(병원+차량 상호통신).

같은 오라클(occ, 풀정보 학습)을 고정하고, **트리가 볼 수 있는 피처를 마스킹**해 4버전 증류:
  V1 site-only : cap_remain(병원통신)·fleet(차량통신) 둘 다 가림 — 현장중심정보만(보수)
  V2 +vehicle  : fleet 만 허용(구급차/센터 차량 상태 통신)
  V3 +hospital : cap_remain 만 허용(병원 실시간 용량 통신)
  V4 full      : 전부 허용(진보 — 상호통신 모두)
각 버전을 시뮬레이션(eval_policy)으로 평가 → 통신 계층의 가치를 해석가능 정책에서 정량화.

DAGGER 데이터셋은 1회 수집(풀트리 롤아웃) 후 4버전 마스킹 트리를 동일 데이터에 적합(효율).
평가는 각 버전을 실제 sim 에 돌려 정직하게 측정. 원본 시나리오 미수정(read-only).

예:
  MCI_OBS_VARIANT=essential MCI_GREEN_MASK=1 MCI_CAP_GATE=occ python src/rl_src/viper_comms_versions.py \\
    --manifest scenarios/manifests/sigungu_osrm_manifest.json \\
    --model results/rl/sigungu_nat/ds_ess_woG_occ_s0/final_model.zip \\
    --out_dir results/viper/comms_versions/sigungu_occ
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from viper_distill import (make_feature_env, load_vecnorm, make_weight_fn,
                           rollout_states, _suppress_stdout)
from evaluate import eval_policy, ppo_policy
import viper_interpret as VI

GLOBAL_DIM = 21  # patient R/Y 10 + fleet 10 + time 1


def masks_for(H):
    """4버전의 '제외할 피처 인덱스' 집합."""
    cap_remain = [h * 4 + 1 for h in range(H)]            # 병원 실시간 용량(병원통신)
    fleet = list(range(H * 4 + 10, H * 4 + 20))           # AMB/UAV fleet 10개(차량통신)
    return {
        "V1_site_only":  set(cap_remain + fleet),
        "V2_plus_vehicle": set(cap_remain),
        "V3_plus_hospital": set(fleet),
        "V4_full":       set(),
    }


def apply_mask(X, excluded):
    if not excluded:
        return X
    X = np.array(X, dtype=np.float32, copy=True)
    idx = sorted(excluded)
    X[:, idx] = 0.0
    return X


def make_masked_tree_policy(tree, excluded):
    classes = tree.classes_
    idx = sorted(excluded)

    def fn(obs, mask, env=None):
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1).copy()
        if idx:
            o[:, idx] = 0.0
        proba = tree.predict_proba(o)[0]
        order = np.argsort(-proba)
        m = np.asarray(mask, dtype=bool)
        for j in order:
            a = int(classes[j])
            if a < len(m) and m[a]:
                return a
        v = np.flatnonzero(m)
        return int(v[0]) if v.size else 0
    return fn


def main():
    ap = argparse.ArgumentParser(description="VIPER 통신-계층 4버전 증류 + 시뮬레이션 평가")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--config_path", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--n_iter", type=int, default=6)
    ap.add_argument("--rollout_eps", type=int, default=15)
    ap.add_argument("--eval_eps", type=int, default=30)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--min_samples_leaf", type=int, default=20)
    ap.add_argument("--crit", default="loggap")
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--label", default="")
    ap.add_argument("--out_dir", default="results/viper/comms_versions/run")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    src = args.manifest or args.config_path
    if not src:
        raise SystemExit("--manifest 또는 --config_path 필요")

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    vn = args.vecnorm
    if vn is None:
        for c in [os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")]:
            if os.path.exists(c):
                vn = c; break
    norm = load_vecnorm(vn) if vn and os.path.exists(vn) else None
    print(f"[comms] VecNorm={'로드' if norm else '없음'} | model={args.model}")

    factory = make_feature_env(src, norm)
    oracle = ppo_policy(model)
    weight_fn = make_weight_fn(model, args.crit)

    # ---------- DAGGER 데이터셋 1회 수집 (풀트리 롤아웃) ----------
    D_obs, D_act, D_w = [], [], []
    full_tree = None
    for i in range(args.n_iter):
        roll = oracle if i == 0 else make_masked_tree_policy(full_tree, set())
        with _suppress_stdout():
            obs_l, mask_l = rollout_states(factory, roll, args.rollout_eps, args.seed_base + 100 * i)
        for s, mk in zip(obs_l, mask_l):
            D_obs.append(s); D_act.append(oracle(s, mk, None)); D_w.append(weight_fn(s, mk))
        Xf = np.asarray(D_obs); yf = np.asarray(D_act); wf = np.asarray(D_w, dtype=np.float64)
        full_tree = DecisionTreeClassifier(max_depth=args.max_depth,
                                           min_samples_leaf=args.min_samples_leaf, random_state=0)
        full_tree.fit(Xf, yf, sample_weight=wf)
        sys.stderr.write(f"[DAGGER it{i}] n={len(D_obs)}\n"); sys.stderr.flush()

    X = np.asarray(D_obs); y = np.asarray(D_act); w = np.asarray(D_w, dtype=np.float64)
    H = (X.shape[1] - GLOBAL_DIM) // 4
    masks = masks_for(H)

    # 오라클 시뮬 기준선 (woG = 학습목표; raw 는 Green 지배라 비교 둔감)
    with _suppress_stdout():
        m_ppo = eval_policy(factory, oracle, args.eval_eps, args.seed_base + 9000)
    print(f"[comms] 오라클(PPO) 시뮬 보상 woG={m_ppo['mean_R_woG']:.2f} (raw={m_ppo['mean_R']:.2f})")

    # ---------- 4버전 마스킹 트리 적합 + 시뮬 평가 ----------
    rows = []
    for vname, excl in masks.items():
        Xm = apply_mask(X, excl)
        tree = DecisionTreeClassifier(max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf, random_state=0)
        tree.fit(Xm, y, sample_weight=w)
        fidelity = float((tree.predict(Xm) == y).mean())
        pol = make_masked_tree_policy(tree, excl)
        with _suppress_stdout():
            m = eval_policy(factory, pol, args.eval_eps, args.seed_base + 9000)
        n_feat_used = int((tree.feature_importances_ > 0).sum())
        with open(os.path.join(args.out_dir, f"{vname}.pkl"), "wb") as f:
            pickle.dump({"tree": tree, "excluded": sorted(excl), "version": vname,
                         "crit": args.crit, "H": H}, f)
        row = {"version": vname,
               "tree_R_woG": round(m["mean_R_woG"], 2),
               "ppo_R_woG": round(m_ppo["mean_R_woG"], 2),
               "tree_vs_ppo_woG": round(m["mean_R_woG"] - m_ppo["mean_R_woG"], 2),
               "tree_R_raw": round(m["mean_R"], 2),
               "fidelity": round(fidelity, 3), "leaves": int(tree.get_n_leaves()),
               "depth": int(tree.get_depth()), "n_feat_used": n_feat_used}
        rows.append(row)
        sys.stderr.write(f"[{vname}] tree_woG={row['tree_R_woG']} (raw={row['tree_R_raw']}) "
                         f"fid={row['fidelity']} feat={n_feat_used}\n"); sys.stderr.flush()
        # 해석본 (피처명·역정규화·액션디코드)
        names = VI.build_feature_names(H)
        mean, std = (VI.load_vecnorm(vn) if vn and os.path.exists(vn) else (None, None))
        L = [f"== {args.label} {vname} (제외 {len(excl)}피처) =="]
        L += [f"tree_woG={row['tree_R_woG']} vs PPO_woG={row['ppo_R_woG']} (raw {row['tree_R_raw']}) "
              f"| fidelity={row['fidelity']} | leaves={row['leaves']} | 사용피처={n_feat_used}"]
        L += ["── 피처 중요도 Top ──"]
        L += [f"   {v:.3f}  {nm}" for nm, v in VI.feature_importance(tree, names, 12)]
        L += ["── 결정 규칙(상위) ──"]
        L += VI.walk_tree(tree, names, mean, std, H, max_print_depth=4)
        with open(os.path.join(args.out_dir, f"{vname}_interpreted.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")

    import csv
    with open(os.path.join(args.out_dir, "comms_versions_summary.csv"), "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    with open(os.path.join(args.out_dir, "comms_versions_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "ppo_R": m_ppo["mean_R"], "H": H, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n=== {args.label} 통신-계층 4버전 (시뮬 보상, woG=학습목표) ===")
    print(f"{'버전':<18}{'tree_woG':>9}{'vs_PPO':>8}{'raw':>8}{'fidelity':>9}{'사용피처':>8}")
    for r in rows:
        print(f"{r['version']:<18}{r['tree_R_woG']:>9}{r['tree_vs_ppo_woG']:>8}{r['tree_R_raw']:>8}{r['fidelity']:>9}{r['n_feat_used']:>8}")
    print(f"[저장] {args.out_dir}/comms_versions_summary.csv")


if __name__ == "__main__":
    main()
