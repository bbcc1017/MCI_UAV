"""프로그램 정책 평가·튜닝 (플랜 v2 Phase 3-B).

프로그램 정책(여러 파라미터 조합) + LB-T4 + heur (+옵션 RL L3)를 같은 (region,seed) 실현에서
paired 평가 → per-ep PDR_woG. closed-loop 성능으로 파라미터 선택("수작업규칙 −0.92" 함정 회피),
프로그램 vs LB-T4·RL 승/무/패.

--combos "T:rank:tier3only:redonly,..." (예 "4:3:1:1,4:2:1:1,4:3:0:1"). --with_rl 이면 L3_pointer 추가.
규칙류(프로그램·LB·heur)는 essential env(norm없음), RL은 essential+load+L3 vecnorm. occ 고정.

예 스윕: python src/rl_src/program_eval.py --combos "4:2:1:1,4:3:1:1,4:5:1:1,4:3:0:1" --regions 서울,대구,강원,전남,경기 --n_eps 500
예 최종: python src/rl_src/program_eval.py --combos "4:3:1:1" --with_rl --n_eps 1000
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, json, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()


def _pdr(factory, pol, seed):
    env = factory(seed=seed); obs, _ = env.reset(seed=seed); done = False; w = 0.0
    while not done:
        m = env.action_masks(); a = pol(obs, m, env.unwrapped)
        obs, r, te, tr, info = env.step(a); w += info.get("r_woG", 0.0); done = te or tr
    prev = env.unwrapped.preventable_woG
    return (1.0 - w / prev) if prev > 0 else 0.0


def worker(job):
    region, cfg, best_rule, combos, with_rl, n_eps = job
    import numpy as np, torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy
    from program_policy import make_program_policy
    from evaluate import ppo_policy
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa
    try:
        with _suppress_stdout():
            os.environ["MCI_OBS_VARIANT"] = "essential"
            rule_fac = make_feature_env(cfg, None); rule_fac(seed=SEED)
            entries = [("lb_T4", rule_fac, make_cap_policy(best_rule, 4)),
                       ("heur", rule_fac, make_heuristic_policy(best_rule))]
            for cs in combos:
                T, fac, ro, t3 = cs.split(":")  # T:factor:redonly:tier3pref
                entries.append((f"prog_{cs}", rule_fac,
                                make_program_policy(best_rule, T=float(T), uav_time_factor=float(fac),
                                                    uav_red_only=bool(int(ro)), uav_tier3_pref=bool(int(t3)))))
            if with_rl:
                os.environ["MCI_OBS_VARIANT"] = "essential+load"
                mdir = os.path.join(REPO, "results/rl/redesign/L3_pointer_s0")
                model = MaskablePPO.load(os.path.join(mdir, "final_model.zip"), device="cpu")
                norm = load_vecnorm(os.path.join(mdir, "vecnormalize.pkl"))
                rl_fac = make_feature_env(cfg, norm); rl_fac(seed=SEED)
                entries.append(("rl", rl_fac, ppo_policy(model)))
            names = [e[0] for e in entries]
            P = {n: np.zeros(n_eps) for n in names}
            for ep in range(n_eps):
                s = SEED + ep
                for n, fac, pol in entries:
                    P[n][ep] = _pdr(fac, pol, s)
        out = dict(region=region, ok=True, _P={n: P[n].tolist() for n in names}, names=names)
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
    ap.add_argument("--combos", default="4:3:1:1")
    ap.add_argument("--with_rl", action="store_true")
    ap.add_argument("--regions", default="")
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/program_eval.csv"))
    A = ap.parse_args()
    import numpy as np
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    best = {}
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            best[r["region"]] = r["best_rule"]
    regions = A.regions.split(",") if A.regions else [r for r in SIDO17 if r in manifest]
    combos = A.combos.split(",")
    jobs = [(rg, manifest[rg], best[rg], combos, A.with_rl, A.n_eps)
            for rg in regions if rg in manifest and rg in best]
    print(f"[program] regions={len(jobs)} combos={combos} with_rl={A.with_rl} n_eps={A.n_eps}", flush=True)
    res, t0 = [], time.time()
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
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
    print(f"\n저장 {A.out} wall={time.time()-t0:.0f}s", flush=True)
    print("\n=== 평균 PDR_woG (낮을수록 좋음) ===", flush=True)
    for n in names:
        print(f"  {n:>16}: {np.mean([r[f'PDR_{n}'] for r in ok]):.4f}", flush=True)

    def paired(a, b):  # a=대상, b=baseline; 양수=a우수(PDR낮음)
        diffs = []
        for r in ok:
            d = np.array(r["_P"][b]) - np.array(r["_P"][a]); md = d.mean()
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            diffs.append((md, "win" if md > ci else "loss" if md < -ci else "tie"))
        return np.mean([d[0] for d in diffs]), \
            sum(d[1] == "win" for d in diffs), sum(d[1] == "tie" for d in diffs), sum(d[1] == "loss" for d in diffs)

    print("\n=== 프로그램 vs LB-T4 (양수=프로그램 우수, 승/무/패) ===", flush=True)
    for n in names:
        if n.startswith("prog_"):
            md, wi, ti, lo = paired(n, "lb_T4")
            print(f"  {n}: {md:+.4f} ({wi}/{ti}/{lo})", flush=True)
    if "rl" in names:
        print("\n=== 프로그램 vs RL(L3) ===", flush=True)
        for n in names:
            if n.startswith("prog_"):
                md, wi, ti, lo = paired(n, "rl")
                print(f"  {n}: {md:+.4f} ({wi}/{ti}/{lo})", flush=True)


if __name__ == "__main__":
    main()
