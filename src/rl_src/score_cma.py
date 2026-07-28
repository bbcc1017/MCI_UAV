"""스코어 정책 closed-loop 튜닝 (플랜 v2 추출 트랙 B3).

fit_score(B2, 조건부로짓)의 정태 w 를 초기값으로, **튜닝풀 좌표에서 CRN paired 평균
PDR_woG 를 최소화**하도록 자작 CEM(교차엔트로피법)으로 미세조정한다. 정태 acc 가 아니라
closed-loop 성능으로 고르므로 "수작업규칙 −0.92" 함정(정태최적≠폐루프최적)을 회피한다.

자작 CEM(cma 패키지 미설치·설치 금지):
  - popsize 24, elite 25%(=6), 세대 ≤20.
  - 초기 mean = score_fit.json 의 w_vec, 초기 std = |w|·0.5 + 0.1.
  - 세대: N(mean,std) 표본 → 각 후보 정책을 튜닝풀서 평가 → PDR 오름차순 elite 선정 →
    mean=elite 평균, std=elite 표준편차(하한 std_floor). 전역 best 추적.

튜닝풀(⚠️시도17 좌표 사용 금지 — 최종 판정 전용):
  sigungu_osrm_manifest(250) 를 **sigcd 정렬 후 균등간격**(np.linspace 인덱스)으로 40지역 결정
  선정. × --eps_per(기본 30) CRN(seed 11000+ep 고정) → 모든 후보가 동일 실현서 paired 비교.

병렬: 지역당 1잡(Pool workers × maxtasksperchild=1, OMP=1 핀). 각 잡이 env 를 1회 빌드해
전 후보×eps 롤아웃 재사용(규칙 정책이라 NN 불필요·빠름). 잡 반환 = 그 지역의 후보별 평균 PDR.
마스터가 지역 평균 → 후보 스코어. 세대별 진행 로그(flush) + best w 저장(--out, 재개 가능).

CLI 예:  PYTHONIOENCODING=utf-8 python src/rl_src/score_cma.py \
  --init results/rl/redesign/score_fit.json --mode timesave --T_hard 4 \
  --tune_regions 40 --eps_per 30 --pop 24 --gens 20 --workers 12 \
  --out results/rl/redesign/score_cma.json
스모크:  python src/rl_src/score_cma.py --init results/rl/redesign/score_fit.json --smoke \
  --out /tmp/claude/score_cma_smoke.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
# 클래스 축 범용 룰(t_meta_wrapper.GENERIC_RULE 와 동일) — 전국 단일 스코어 정책 규약.
GENERIC_RULE = "START, YellowNearest, Red Both_AMBFirst, Yellow Both_AMBFirst"


def _sigcd(key):
    return key.rsplit("_", 1)[1] if "_" in key else key


def select_tune_regions(manifest_path, k):
    """sigcd 정렬 후 균등간격(np.linspace) 인덱스로 k 지역 결정 선정(재현 가능)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    keys = sorted(manifest.keys(), key=_sigcd)
    if k >= len(keys):
        sel = keys
    else:
        idx = np.unique(np.round(np.linspace(0, len(keys) - 1, k)).astype(int))
        sel = [keys[i] for i in idx]
    return [(rg, manifest[rg]) for rg in sel]


def _make_T_lookup(spec):
    """--T_lookup 프리셋 → f(rho, n_elig)->T (없으면 None). 'rho_step'=평상4/중8/고16."""
    if not spec or spec == "none":
        return None
    if spec == "rho_step":
        def _t(rho, n_elig):
            return 16.0 if rho >= 4.0 else (8.0 if rho >= 2.0 else 4.0)
        return _t
    raise ValueError(f"알 수 없는 T_lookup 프리셋: {spec}")


# ------------------------------------------------------------------ 워커(지역당 1잡)
def worker(job):
    """지역 1곳에서 전 후보(pop)×eps 롤아웃 → 후보별 평균 PDR_woG 배열 반환."""
    (region, cfg, pop, mode, T_hard, T_lookup_spec, guard_n, uav_time_factor,
     uav_red_only, eps_per) = job
    import numpy as _np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    os.environ["MCI_OBS_VARIANT"] = "essential"      # 규칙류 관례(정규화 불요)
    try:
        from viper_distill import make_feature_env, _suppress_stdout
        from score_policy import make_score_policy
        T_lookup = _make_T_lookup(T_lookup_spec)
        pop = _np.asarray(pop, dtype=float)
        P = _np.zeros(pop.shape[0], dtype=float)
        with _suppress_stdout():
            fac = make_feature_env(cfg, None)
            fac(seed=SEED)                            # env 1회 빌드(캐시)
            for j, w in enumerate(pop):
                pol = make_score_policy(w, GENERIC_RULE, mode=mode, T_hard=T_hard,
                                        T_lookup=T_lookup, guard_n=guard_n,
                                        uav_time_factor=uav_time_factor,
                                        uav_red_only=uav_red_only)
                acc = 0.0
                for ep in range(eps_per):
                    s = SEED + ep
                    env = fac(seed=s)
                    obs, _ = env.reset(seed=s)
                    done = False
                    wog = 0.0
                    while not done:
                        m = env.action_masks()
                        a = pol(obs, m, env.unwrapped)
                        obs, r, te, tr, info = env.step(a)
                        wog += info.get("r_woG", 0.0)
                        done = te or tr
                    prev = env.unwrapped.preventable_woG
                    acc += (1.0 - wog / prev) if prev > 0 else 0.0
                P[j] = acc / eps_per
        return dict(region=region, ok=True, P=P.tolist())
    except Exception as e:
        import traceback
        return dict(region=region, ok=False, err=(str(e) + traceback.format_exc())[:400])


# ------------------------------------------------------------------ CEM 마스터
def evaluate_pop(pop, regions, cem_args, workers):
    """전 지역 병렬 평가 → 후보별 지역평균 PDR (pop,). 완료 지역만 평균(실패 지역 로그)."""
    jobs = [(rg, cfg, [w.tolist() for w in pop]) + cem_args for rg, cfg in regions]
    per_region = []
    with Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(worker, jobs):
            if r["ok"]:
                per_region.append(np.asarray(r["P"], dtype=float))
            else:
                print(f"    [FAIL {r['region']}] {r['err'][:200]}", flush=True)
    if not per_region:
        raise RuntimeError("전 지역 평가 실패")
    return np.mean(np.vstack(per_region), axis=0)      # (pop,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(REPO, "results/rl/redesign/score_fit.json"),
                    help="score_fit.json(w_vec) — 초기 mean")
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"))
    ap.add_argument("--mode", choices=["timesave", "joint"], default="timesave")
    ap.add_argument("--T_hard", type=float, default=4.0)
    ap.add_argument("--T_lookup", default="none", help="none|rho_step (설정 시 T_hard 무시)")
    ap.add_argument("--guard_n", type=int, default=None)
    ap.add_argument("--uav_time_factor", type=float, default=0.8)
    ap.add_argument("--uav_red_only", type=int, default=1)
    ap.add_argument("--tune_regions", type=int, default=40)
    ap.add_argument("--eps_per", type=int, default=30)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--elite_frac", type=float, default=0.25)
    ap.add_argument("--std_floor", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/score_cma.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="--out 재개 무시하고 처음부터")
    A = ap.parse_args()

    if A.smoke:
        A.tune_regions, A.eps_per, A.pop, A.gens = 4, 6, 6, 2
        A.workers = min(A.workers, 4)

    # 초기 mean/std
    from score_features import PHI_NAMES, K_PHI
    with open(A.init, encoding="utf-8") as f:
        fit = json.load(f)
    w0 = np.asarray(fit["w_vec"], dtype=float)
    if w0.shape[0] != K_PHI:
        raise ValueError(f"init w_vec 길이 {w0.shape[0]} != K_PHI({K_PHI})")
    mean = w0.copy()
    std = np.abs(w0) * 0.5 + 0.1
    start_gen = 0
    best = {"w": w0.tolist(), "score": float("inf"), "gen": -1}
    history = []

    # 재개(--out 존재 & --fresh 아님 & 스모크 아님)
    if (not A.smoke) and (not A.fresh) and os.path.exists(A.out):
        try:
            with open(A.out, encoding="utf-8") as f:
                prev = json.load(f)
            if "mean" in prev and "std" in prev and prev.get("gen", -1) >= 0:
                mean = np.asarray(prev["mean"], dtype=float)
                std = np.asarray(prev["std"], dtype=float)
                start_gen = int(prev["gen"]) + 1
                best = prev.get("best", best)
                history = prev.get("history", [])
                print(f"[resume] gen {start_gen} 부터 재개 (이전 best={best['score']:.4f})", flush=True)
        except Exception as e:
            print(f"[resume] 재개 실패({e}) — 처음부터", flush=True)

    regions = select_tune_regions(A.manifest, A.tune_regions)
    n_elite = max(2, int(round(A.pop * A.elite_frac)))
    cem_args = (A.mode, A.T_hard, A.T_lookup, A.guard_n,
                A.uav_time_factor, bool(A.uav_red_only), A.eps_per)
    print(f"[score_cma] mode={A.mode} T_hard={A.T_hard} T_lookup={A.T_lookup} "
          f"guard_n={A.guard_n} regions={len(regions)} eps={A.eps_per} pop={A.pop} "
          f"gens={A.gens} elite={n_elite} workers={A.workers}", flush=True)
    print(f"  튜닝풀(sigcd 균등): {[r[0] for r in regions[:6]]}{' ...' if len(regions) > 6 else ''}",
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    t0 = time.time()
    for gen in range(start_gen, A.gens):
        rng = np.random.default_rng(1000 + gen)       # 세대별 재현 가능 표본
        # elite mean 은 그대로 유지(exploit) + 나머지는 표본
        pop = np.empty((A.pop, K_PHI), dtype=float)
        pop[0] = mean                                  # elitist: 현 mean 항상 포함
        pop[1:] = rng.normal(mean, np.maximum(std, A.std_floor), size=(A.pop - 1, K_PHI))
        scores = evaluate_pop(pop, regions, cem_args, A.workers)   # (pop,) PDR (낮을수록 좋음)

        order = np.argsort(scores)
        elite = pop[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), A.std_floor)
        g_best_i = int(order[0])
        if scores[g_best_i] < best["score"]:
            best = {"w": pop[g_best_i].tolist(), "score": float(scores[g_best_i]), "gen": gen}
        hrow = {"gen": gen, "best_pdr": float(scores[g_best_i]),
                "elite_mean_pdr": float(scores[order[:n_elite]].mean()),
                "pop_mean_pdr": float(scores.mean()), "global_best_pdr": best["score"]}
        history.append(hrow)
        print(f"  [gen {gen}] best={scores[g_best_i]:.4f} elite_mean="
              f"{scores[order[:n_elite]].mean():.4f} pop_mean={scores.mean():.4f} "
              f"global_best={best['score']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        # 중간 저장(재개 가능)
        with open(A.out, "w", encoding="utf-8") as f:
            json.dump({"phi_names": PHI_NAMES, "best": best,
                       "w": {nm: float(best["w"][i]) for i, nm in enumerate(PHI_NAMES)},
                       "w_vec": best["w"], "mean": mean.tolist(), "std": std.tolist(),
                       "gen": gen, "history": history,
                       "config": {"mode": A.mode, "T_hard": A.T_hard, "T_lookup": A.T_lookup,
                                  "guard_n": A.guard_n, "tune_regions": len(regions),
                                  "eps_per": A.eps_per, "pop": A.pop, "gens": A.gens,
                                  "init": A.init}}, f, ensure_ascii=False, indent=2)

    print(f"\n저장 {A.out}  best PDR={best['score']:.4f} (gen {best['gen']})  "
          f"wall={time.time()-t0:.0f}s", flush=True)
    print("  best w:", {nm: round(best["w"][i], 3) for i, nm in enumerate(PHI_NAMES)}, flush=True)


if __name__ == "__main__":
    main()
