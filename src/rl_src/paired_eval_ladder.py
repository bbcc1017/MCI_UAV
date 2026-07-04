"""L사다리 paired 평가 — RL 4런(L0~L3) + 규칙 3종(적응T-LB·LB-T4·휴리best)을 같은 시드에서 비교.

플랜 v2 Phase 1 판정 하네스. 같은 (region, seed) 실현에서 7개 정책을 각자 롤아웃 →
per-episode woG·PDR_woG 배열 → 사다리 기여(L1−L0, L2−L1, L3−L2)와 baseline 대비
승/무/패·평균차·95%CI. 지표는 PDR_woG(규모 불변) 주, woG 보조.

핵심 주의:
  - 모델별 obs variant/vecnorm 상이: L0/L1=essential(209), L2/L3=essential+load(355).
    env 는 정책마다 자기 variant+자기 vecnorm 으로 빌드(빌드 전 MCI_OBS_VARIANT 설정).
    dynamics 는 obs/norm 과 무관 → reset(seed=s) 실현 동일 → paired 성립.
  - 규칙 정책은 obs 비의존(en_manager·get_static_eta) → norm 없는 essential env 로 실행.
  - occ 게이트 고정(플랜 탐구단계). MCI_CAP_GATE 는 step 시점에 읽히므로 워커 전역 설정.
  - --use_ckpt: 최신 checkpoint + norm=None 로 배관 스모크(정규화 없어 성능 무의미, 형상만 검증).

예(정식): PYTHONIOENCODING=utf-8 python src/rl_src/paired_eval_ladder.py --n_eps 1000 --workers 17
예(스모크): ... --use_ckpt --n_eps 3 --regions 서울,강원
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
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # eval 은 info['r_woG'] 를 직접 읽음(모드 무관)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()

# 모델 → obs variant (학습 시와 동일해야 로드/forward 정합)
MODEL_VARIANT = {
    "L0_base": "essential", "L1_hygiene": "essential",
    "L2_loadobs": "essential+load", "L3_pointer": "essential+load",
}


def _rollout_woG(factory, policy_fn, seed):
    """1 에피소드 롤아웃 → (woG 합, PDR_woG). factory(seed) 는 캐시된 env 재사용."""
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
    pdr = 1.0 - w / prev if prev > 0 else 0.0
    return w, pdr


def worker(job):
    region, cfg, best_rule, model_root, models, n_eps, use_ckpt = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"  # 탐구단계 고정
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (deepsets 역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from evaluate import ppo_policy
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy, make_adaptive_cap_policy

    def build_factory(variant, norm):
        os.environ["MCI_OBS_VARIANT"] = variant
        fac = make_feature_env(cfg, norm)
        fac(seed=SEED)  # 강제 빌드(현재 variant 로 캐시 고정) — 이후 env var 바뀌어도 무관
        return fac

    try:
        with _suppress_stdout():
            # ---- 정책별 (factory, policy_fn) 구성 ----
            entries = []  # (name, factory, policy_fn)
            for m in models:
                mdir = os.path.join(model_root, f"{m}_s0")
                if use_ckpt:
                    cks = sorted([f for f in os.listdir(os.path.join(mdir, "checkpoints"))
                                  if f.endswith(".zip")],
                                 key=lambda f: int(f.split("_")[-2]))
                    if not cks:
                        continue
                    zip_path = os.path.join(mdir, "checkpoints", cks[-1])
                    norm = None  # 체크포인트엔 vecnorm 없음 → 스모크(형상만)
                else:
                    zip_path = os.path.join(mdir, "final_model.zip")
                    if not os.path.exists(zip_path):
                        continue
                    vn = os.path.join(mdir, "vecnormalize.pkl")
                    norm = load_vecnorm(vn) if os.path.exists(vn) else None
                model = MaskablePPO.load(zip_path, device="cpu")
                fac = build_factory(MODEL_VARIANT[m], norm)
                entries.append((m, fac, ppo_policy(model)))

            # 규칙 3종: norm 없는 essential env(obs 비의존이나 형상 유지)
            rule_fac = build_factory("essential", None)
            entries.append(("heur", rule_fac, make_heuristic_policy(best_rule)))
            entries.append(("lb_T4", rule_fac, make_cap_policy(best_rule, 4)))
            entries.append(("lb_adaptT", rule_fac, make_adaptive_cap_policy(best_rule)))

            names = [e[0] for e in entries]
            W = {n: np.zeros(n_eps) for n in names}
            P = {n: np.zeros(n_eps) for n in names}
            for ep in range(n_eps):
                s = SEED + ep
                for name, fac, pol in entries:
                    w, pdr = _rollout_woG(fac, pol, s)
                    W[name][ep] = w
                    P[name][ep] = pdr

        out = {"region": region, "n_eps": n_eps, "ok": True, "names": names}
        for n in names:
            out[f"woG_{n}"] = float(W[n].mean())
            out[f"PDR_{n}"] = float(P[n].mean())
        # paired 배열 보존(집계용) — PDR_woG 기준 승/무/패는 main 에서
        out["_P"] = {n: P[n].tolist() for n in names}
        return out
    except Exception as e:
        import traceback
        return {"region": region, "ok": False, "err": (str(e) + traceback.format_exc())[:400]}


def _paired(a, b):
    """a,b: per-ep PDR_woG 배열. PDR 은 낮을수록 좋음 → 개선 = b−a(a가 모델). 반환 (mean_impr, ci, sig)."""
    import numpy as np
    d = np.asarray(b) - np.asarray(a)  # >0 = a(모델)가 baseline b 보다 PDR 낮음(우수)
    md = float(d.mean())
    n = len(d)
    ci = 1.96 * float(np.std(d, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    sig = "win" if md > ci else "loss" if md < -ci else "tie"
    return md, ci, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--heur_csv", default=os.path.join(REPO, "results/sido_osrm_heuristic_best.csv"))
    ap.add_argument("--regions", default="", help="쉼표구분(기본 시도17 전체)")
    ap.add_argument("--model_root", default=os.path.join(REPO, "results/rl/redesign"))
    ap.add_argument("--models", default="L0_base,L1_hygiene,L2_loadobs,L3_pointer")
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--use_ckpt", action="store_true", help="스모크: 최신 ckpt+norm없음")
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/paired_ladder.csv"))
    A = ap.parse_args()

    import numpy as np  # noqa
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    # 휴리 best_rule (BOM 대응)
    best = {}
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            best[r["region"]] = r["best_rule"]
    regions = A.regions.split(",") if A.regions else [r for r in SIDO17 if r in manifest]
    models = A.models.split(",")
    jobs = [(rg, manifest[rg], best[rg], A.model_root, models, A.n_eps, A.use_ckpt)
            for rg in regions if rg in manifest and rg in best]
    print(f"[paired] regions={len(jobs)} models={models} n_eps={A.n_eps} "
          f"use_ckpt={A.use_ckpt} workers={A.workers}", flush=True)

    res, t0 = [], time.time()
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
                print(f"  [{k}/{len(jobs)}] {r['region']}: "
                      + " ".join(f"{n}={r['PDR_'+n]:.4f}" for n in r["names"])
                      + f"  ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}", flush=True)

    ok = [r for r in res if r["ok"]]
    if not ok:
        print("전부 실패", flush=True); return
    names = ok[0]["names"]
    rl_models = [m for m in models if f"PDR_{m}" in ok[0]]
    baselines = [b for b in ("heur", "lb_T4", "lb_adaptT") if b in names]

    # 절대 PDR_woG (낮을수록 좋음) 저장
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        cols = ["region", "n_eps"] + [f"PDR_{n}" for n in names] + [f"woG_{n}" for n in names]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ok:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\n저장 {A.out}  wall={time.time()-t0:.0f}s", flush=True)

    # paired 요약: 각 RL vs 각 baseline (PDR_woG 개선 = baseline−RL, 승=RL이 유의 낮음)
    print("\n=== paired PDR_woG (양수=RL 우수, 승/무/패 across 지역) ===", flush=True)
    for m in rl_models:
        line = f"[{m}]"
        for b in baselines:
            diffs = [_paired(r["_P"][m], r["_P"][b]) for r in ok]
            md = np.mean([d[0] for d in diffs])
            win = sum(d[2] == "win" for d in diffs); tie = sum(d[2] == "tie" for d in diffs)
            loss = sum(d[2] == "loss" for d in diffs)
            line += f"  vs {b}: {md:+.4f} ({win}/{tie}/{loss})"
        print(line, flush=True)
    # 사다리 기여 (인접 단계 PDR 개선)
    print("\n=== 사다리 기여 (양수=상위단계가 PDR 낮춤) ===", flush=True)
    order = [m for m in ("L0_base", "L1_hygiene", "L2_loadobs", "L3_pointer") if m in rl_models]
    for i in range(1, len(order)):
        diffs = [_paired(r["_P"][order[i]], r["_P"][order[i-1]]) for r in ok]
        md = np.mean([d[0] for d in diffs])
        win = sum(d[2] == "win" for d in diffs); loss = sum(d[2] == "loss" for d in diffs)
        print(f"  {order[i]} − {order[i-1]}: {md:+.4f}  승{win}/패{loss}", flush=True)


if __name__ == "__main__":
    main()
