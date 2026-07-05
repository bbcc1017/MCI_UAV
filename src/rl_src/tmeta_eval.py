"""T-메타 평가·T=f(상태) 추출 (플랜 v2 Phase 3-C).

T-메타 RL + 고정프로그램(T4) + LB-T4 + heur + full-RL(L3)를 같은 시드 paired PDR_woG 비교,
동시에 T-메타의 T 선택을 상태(ρ·시간·대기부하)와 함께 로깅 → T=f(상태) 규칙 추출.
occ·essential+load. 시도17.

예: python src/rl_src/tmeta_eval.py --n_eps 1000 --workers 17
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, gzip, json, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
TMETA_DIR = os.path.join(REPO, "results/rl/redesign/tmeta_s0")
L3_DIR = os.path.join(REPO, "results/rl/redesign/L3_pointer_s0")


def _pdr_woG(factory, pol, seed):
    env = factory(seed=seed); obs, _ = env.reset(seed=seed); done = False; w = 0.0
    while not done:
        m = env.action_masks(); a = pol(obs, m, env.unwrapped)
        obs, r, te, tr, info = env.step(a); w += info.get("r_woG", 0.0); done = te or tr
    prev = env.unwrapped.preventable_woG
    return (1.0 - w / prev) if prev > 0 else 0.0


def worker(job):
    region, cfg, best_rule, n_eps = job
    import numpy as np, torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy
    from program_policy import make_program_policy
    from t_meta_wrapper import TMetaWrapper, T_SET_DEFAULT, GENERIC_RULE
    from evaluate import ppo_policy
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa
    try:
        rec = []  # T 선택 로그
        with _suppress_stdout():
            os.environ["MCI_OBS_VARIANT"] = "essential+load"
            # 규칙류(프로그램 T4·LB·heur): essential+load env(norm없음)
            rule_fac = make_feature_env(cfg, None); rule_fac(seed=SEED)
            entries = [("prog_T4", rule_fac, make_program_policy(GENERIC_RULE, T=4, uav_time_factor=0.8, uav_red_only=False)),
                       ("lb_T4", rule_fac, make_cap_policy(best_rule, 4)),
                       ("heur", rule_fac, make_heuristic_policy(best_rule))]
            # full-RL L3
            l3 = MaskablePPO.load(os.path.join(L3_DIR, "final_model.zip"), device="cpu")
            l3n = load_vecnorm(os.path.join(L3_DIR, "vecnormalize.pkl"))
            l3_fac = make_feature_env(cfg, l3n); l3_fac(seed=SEED)
            entries.append(("rl_L3", l3_fac, ppo_policy(l3)))
            names = [e[0] for e in entries] + ["tmeta"]
            P = {n: np.zeros(n_eps) for n in names}
            # 규칙·RL 롤아웃
            for ep in range(n_eps):
                s = SEED + ep
                for n, fac, pol in entries:
                    P[n][ep] = _pdr_woG(fac, pol, s)
            # T-메타: TMetaWrapper 환경 + 학습 정책, T 선택 로깅
            tm_model = MaskablePPO.load(os.path.join(TMETA_DIR, "final_model.zip"), device="cpu")
            tm_norm = load_vecnorm(os.path.join(TMETA_DIR, "vecnormalize.pkl"))
            base_fac = make_feature_env(cfg, None)  # norm 은 아래서 수동 적용
            hf = base_fac(seed=SEED)  # HospitalFeatureWrapper
            tm_env = TMetaWrapper(hf)
            mean, std, clip = tm_norm
            for ep in range(n_eps):
                s = SEED + ep
                obs, _ = tm_env.reset(seed=s); done = False; w = 0.0
                while not done:
                    o = np.clip((np.asarray(obs, np.float32) - mean) / std, -clip, clip)
                    a, _ = tm_model.predict(o, action_masks=np.asarray(tm_env.action_masks(), bool), deterministic=True)
                    # 상태(rho·time) 로깅 — global obs 에서 (essential+load: idx 21=rho, 25=t_norm)
                    g = np.asarray(obs, np.float32)[47 * 7:]
                    rec.append((region, ep, float(T_SET_DEFAULT[int(a)]), round(float(g[21]), 3), round(float(g[25]), 3)))
                    obs, r, te, tr, info = tm_env.step(int(a)); w += info.get("r_woG", 0.0); done = te or tr
                prev = tm_env.env.unwrapped.preventable_woG
                P["tmeta"][ep] = (1.0 - w / prev) if prev > 0 else 0.0
        out = dict(region=region, ok=True, names=names, _P={n: P[n].tolist() for n in names}, rec=rec)
        for n in names:
            out[f"PDR_{n}"] = float(P[n].mean())
        return out
    except Exception as e:
        import traceback
        return dict(region=region, ok=False, err=(str(e) + traceback.format_exc())[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--heur_csv", default=os.path.join(REPO, "results/sido_osrm_heuristic_best.csv"))
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/tmeta_eval.csv"))
    ap.add_argument("--tlog", default=os.path.join(REPO, "results/rl/redesign/tmeta_Tlog.csv.gz"))
    A = ap.parse_args()
    import numpy as np
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    best = {}
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            best[r["region"]] = r["best_rule"]
    regions = [r for r in SIDO17 if r in manifest and r in best]
    jobs = [(rg, manifest[rg], best[rg], A.n_eps) for rg in regions]
    print(f"[tmeta_eval] regions={len(jobs)} n_eps={A.n_eps}", flush=True)
    res, t0 = [], time.time()
    Trows = []
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
                Trows.extend(r["rec"])
                print(f"  [{k}/{len(jobs)}] {r['region']}: " +
                      " ".join(f"{n}={r['PDR_'+n]:.4f}" for n in r["names"]) + f" ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']}: {r['err'][:160]}", flush=True)
    ok = [r for r in res if r["ok"]]
    if not ok:
        print("전부 실패"); return
    names = ok[0]["names"]
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region"] + [f"PDR_{n}" for n in names]); w.writeheader()
        for r in ok:
            w.writerow({k: r.get(k) for k in ["region"] + [f"PDR_{n}" for n in names]})
    with gzip.open(A.tlog, "wt", newline="", encoding="utf-8") as f:
        wt = csv.writer(f); wt.writerow(["region", "ep", "T", "rho", "t_norm"]); wt.writerows(Trows)
    print(f"\n저장 {A.out}, {A.tlog}  wall={time.time()-t0:.0f}s", flush=True)
    print("\n=== 평균 PDR_woG (낮을수록 좋음) ===", flush=True)
    for n in names:
        print(f"  {n:>10}: {np.mean([r[f'PDR_{n}'] for r in ok]):.4f}", flush=True)

    def paired(a, b):
        ds = []
        for r in ok:
            d = np.array(r["_P"][b]) - np.array(r["_P"][a]); md = d.mean()
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            ds.append((md, "win" if md > ci else "loss" if md < -ci else "tie"))
        return np.mean([x[0] for x in ds]), sum(x[1] == "win" for x in ds), \
            sum(x[1] == "tie" for x in ds), sum(x[1] == "loss" for x in ds)
    print("\n=== T-메타 vs 각 (양수=T메타 우수, 승/무/패) ===", flush=True)
    for b in ["prog_T4", "lb_T4", "rl_L3"]:
        md, wi, ti, lo = paired("tmeta", b)
        print(f"  vs {b}: {md:+.4f} ({wi}/{ti}/{lo})", flush=True)
    # T 분포
    T = np.array([row[2] for row in Trows]); rho = np.array([row[3] for row in Trows])
    print(f"\n=== T 선택 분포 ({len(T)} 결정) ===", flush=True)
    import collections
    for t, c in sorted(collections.Counter(T).items()):
        print(f"  T={t if t < 1e8 else '∞':>4}: {100*c/len(T):.1f}%", flush=True)
    print(f"  ρ 저(<0.05)서 평균T {T[rho<0.05].mean() if (rho<0.05).any() else 0:.2f}, "
          f"ρ 고(≥0.05)서 평균T {T[rho>=0.05].mean() if (rho>=0.05).any() else 0:.2f}", flush=True)


if __name__ == "__main__":
    main()
