"""UAV 한계가치 곡선 평가 (플랜 v2 Phase 2).

레벨 k∈{0,5,10,15,26} 각각의 시나리오(MCI_UAV_NUM=k)에서 4개 정책을 같은 시드로 paired 평가:
  - 레벨별 RL 모델(uav{k}: k=0 deepsets/action96, k=5·10·15 pointer, k=26=L3_pointer)
  - LB-T4 · 적응T-LB · 휴리best (규칙 baseline, 대수에 따라 UAV 가용성 자동 반영)
지표 PDR_woG(낮을수록 좋음). 산출: 정책별 PDR_woG vs UAV 대수 곡선 → "몇 대면 충분/첫 5대의 가치".

핵심: MCI_UAV_NUM 은 ScenarioManager 빌드 시점에 읽힘 → 워커가 env 빌드 전 설정.
obs=essential+load(355) 전 레벨 공통. occ 게이트. 모델별 vecnorm 동결.

예: PYTHONIOENCODING=utf-8 python src/rl_src/uav_curve_eval.py --n_eps 1000 --workers 34
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
# 레벨 → (모델 디렉터리, extractor)
LEVELS = {0: ("uav0_deepsets", "deepsets"), 5: ("uav5_pointer", "pointer"),
          10: ("uav10_pointer", "pointer"), 15: ("uav15_pointer", "pointer"),
          26: ("L3_pointer", "pointer")}


def _rollout(factory, policy_fn, seed):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    w = 0.0
    while not done:
        mask = env.action_masks()
        a = policy_fn(obs, mask, env.unwrapped)
        obs, r, term, trunc, info = env.step(a)
        w += info.get("r_woG", 0.0)
        done = term or trunc
    prev = env.unwrapped.preventable_woG
    return (1.0 - w / prev) if prev > 0 else 0.0


def worker(job):
    level, region, cfg, best_rule, model_root, n_eps = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    os.environ["MCI_OBS_VARIANT"] = "essential+load"
    os.environ["MCI_UAV_NUM"] = str(level)  # ★ env 빌드 전 레벨 고정
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from evaluate import ppo_policy
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy, make_adaptive_cap_policy
    try:
        mdir = os.path.join(model_root, f"{LEVELS[level][0]}_s0")
        with _suppress_stdout():
            model = MaskablePPO.load(os.path.join(mdir, "final_model.zip"), device="cpu")
            vn = os.path.join(mdir, "vecnormalize.pkl")
            norm = load_vecnorm(vn) if os.path.exists(vn) else None
            rl_fac = make_feature_env(cfg, norm); rl_fac(seed=SEED)   # 강제 빌드(현 UAV_NUM)
            rule_fac = make_feature_env(cfg, None); rule_fac(seed=SEED)
            pols = [("rl", rl_fac, ppo_policy(model)),
                    ("lb_T4", rule_fac, make_cap_policy(best_rule, 4)),
                    ("lb_adaptT", rule_fac, make_adaptive_cap_policy(best_rule)),
                    ("heur", rule_fac, make_heuristic_policy(best_rule))]
            P = {n: np.zeros(n_eps) for n, _, _ in pols}
            for ep in range(n_eps):
                s = SEED + ep
                for n, fac, pol in pols:
                    P[n][ep] = _rollout(fac, pol, s)
        out = dict(level=level, region=region, ok=True, _P={n: P[n].tolist() for n in P})
        for n in P:
            out[f"PDR_{n}"] = float(P[n].mean())
        return out
    except Exception as e:
        import traceback
        return dict(level=level, region=region, ok=False, err=(str(e) + traceback.format_exc())[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--heur_csv", default=os.path.join(REPO, "results/sido_osrm_heuristic_best.csv"))
    ap.add_argument("--model_root", default=os.path.join(REPO, "results/rl/redesign"))
    ap.add_argument("--levels", default="0,5,10,15,26")
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=34)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/uav_curve.csv"))
    A = ap.parse_args()
    import numpy as np

    manifest = json.load(open(A.manifest, encoding="utf-8"))
    best = {}
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            best[r["region"]] = r["best_rule"]
    levels = [int(x) for x in A.levels.split(",")]
    regions = [r for r in SIDO17 if r in manifest and r in best]
    jobs = [(lv, rg, manifest[rg], best[rg], A.model_root, A.n_eps)
            for lv in levels for rg in regions]
    print(f"[uav_curve] levels={levels} regions={len(regions)} jobs={len(jobs)} "
          f"n_eps={A.n_eps} workers={A.workers}", flush=True)

    res, t0 = [], time.time()
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
                print(f"  [{k}/{len(jobs)}] uav{r['level']} {r['region']}: "
                      f"RL={r['PDR_rl']:.4f} T4={r['PDR_lb_T4']:.4f} heur={r['PDR_heur']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL uav{r['level']} {r['region']}: {r['err'][:180]}", flush=True)

    ok = [r for r in res if r["ok"]]
    if not ok:
        print("전부 실패", flush=True); return
    pols = ["rl", "lb_T4", "lb_adaptT", "heur"]
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["level", "region"] + [f"PDR_{p}" for p in pols])
        w.writeheader()
        for r in ok:
            w.writerow({k: r.get(k) for k in ["level", "region"] + [f"PDR_{p}" for p in pols]})
    print(f"\n저장 {A.out}  wall={time.time()-t0:.0f}s", flush=True)

    # 곡선: 정책별 평균 PDR_woG vs UAV 대수 (17지역 평균)
    print("\n=== UAV 한계가치 곡선 (평균 PDR_woG, 낮을수록 좋음) ===", flush=True)
    print(f"{'UAV':>4s} | " + " ".join(f"{p:>10s}" for p in pols), flush=True)
    prev_rl = None
    for lv in levels:
        row = [r for r in ok if r["level"] == lv]
        means = {p: np.mean([r[f"PDR_{p}"] for r in row]) for p in pols}
        marg = ""
        if prev_rl is not None:
            marg = f"  (RL Δ{prev_rl - means['rl']:+.4f} vs 직전대수)"
        prev_rl = means["rl"]
        print(f"{lv:>4d} | " + " ".join(f"{means[p]:>10.4f}" for p in pols) + marg, flush=True)
    # RL vs LB-T4 각 레벨 승/무/패
    print("\n=== 각 레벨 RL vs LB-T4 (PDR_woG paired, 승/무/패) ===", flush=True)
    for lv in levels:
        row = [r for r in ok if r["level"] == lv]
        wtl = [0, 0, 0]
        for r in row:
            d = np.array(r["_P"]["lb_T4"]) - np.array(r["_P"]["rl"])  # >0 = RL 우수
            md = d.mean(); ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            wtl[0 if md > ci else 2 if md < -ci else 1] += 1
        print(f"  uav{lv}: 승{wtl[0]}/무{wtl[1]}/패{wtl[2]}", flush=True)


if __name__ == "__main__":
    main()
