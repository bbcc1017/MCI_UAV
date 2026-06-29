"""해석가능 정책 동물원(zoo) — RL(MaskablePPO) 오라클을 깊이별 얕은 결정트리 + FIGS 로
증류하고, 통용 휴리스틱·RL 대비 woG(1000ep) 로 비교. "해석성-성능 트레이드오프 곡선".

연구 동기: RL 은 휴리스틱을 이기나 비해석. 증류 트리가 (a)휴리 능가 (b)RL 근접 (c)해석가능
한지를, 정보수준 축(occ=진보 실시간통신 / site=보수 통신단절)별로 검증한다.

방법: viper_distill 의 DAgger(criticality 가중) 데이터 수집을 재사용 → 같은 데이터에서
depth ∈ DEPTHS 단일트리와 FIGS 를 각각 fit → woG 평가. 액션 분해 분석은 별도(축별 난이도).

산출: results/viper/zoo/<scope_region_gate>/tree_d{d}.pkl + ZOO_RESULTS.csv.

예: PYTHONIOENCODING=utf-8 python src/rl_src/distill_zoo.py --workers 30 --eval_eps 1000
"""
import os, sys, argparse, csv, json, pickle, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
REGIONS = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
DEPTHS = [3, 4, 6, 8, 12]
DAGGER_EPS = 60      # iter당 롤아웃(오라클 1회 + d12트리 1회 = 2-iter DAgger)
SEED = 2000


def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential", MCI_GREEN_MASK="1", MCI_REWARD_MODE="woG")
    if g == "occ": os.environ.update(MCI_CAP_GATE="occ", MCI_CARED_OBS="1")
    else:          os.environ.update(MCI_CAP_GATE="psent", MCI_CARED_OBS="0")


def worker(job):
    scope, region, gate, model_dir, src, heur_wog, eval_eps, depths = job
    setgate(gate)
    import numpy as np, torch as th; th.set_num_threads(1)
    from sklearn.tree import DecisionTreeClassifier
    from imodels import FIGSClassifier
    from sb3_contrib import MaskablePPO
    from viper_distill import (make_feature_env, load_vecnorm, make_tree_policy,
                               make_weight_fn, rollout_states, _suppress_stdout)
    from evaluate import eval_policy, ppo_policy
    try:
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        factory = make_feature_env(src, norm)
        oracle = ppo_policy(model); wfn = make_weight_fn(model, "loggap")
        # 2-iter DAgger: 오라클 롤아웃 → d12 트리 롤아웃, 각 상태 오라클 라벨+criticality
        Dx, Dy, Dw = [], [], []
        with _suppress_stdout():
            ol, ml = rollout_states(factory, oracle, DAGGER_EPS, SEED)
        for s, mk in zip(ol, ml): Dx.append(s); Dy.append(oracle(s, mk, None)); Dw.append(wfn(s, mk))
        t12 = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=0)
        t12.fit(np.asarray(Dx), np.asarray(Dy), sample_weight=np.asarray(Dw))
        with _suppress_stdout():
            ol2, ml2 = rollout_states(factory, make_tree_policy(t12), DAGGER_EPS, SEED + 1000)
        for s, mk in zip(ol2, ml2): Dx.append(s); Dy.append(oracle(s, mk, None)); Dw.append(wfn(s, mk))
        X = np.asarray(Dx); y = np.asarray(Dy); w = np.asarray(Dw, dtype=np.float64)
        with _suppress_stdout():
            ppo = eval_policy(factory, oracle, eval_eps, SEED + 9000)["mean_R_woG"]
        odir = os.path.join(REPO, "results", "viper", "zoo", f"{scope}_{region}_{gate}")
        os.makedirs(odir, exist_ok=True)
        rows = []
        # 깊이별 단일트리
        for d in depths:
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=0)
            t.fit(X, y, sample_weight=w)
            with _suppress_stdout():
                wg = eval_policy(factory, make_tree_policy(t), eval_eps, SEED + 9000)["mean_R_woG"]
            pickle.dump({"tree": t, "model": f"tree_d{d}"}, open(os.path.join(odir, f"tree_d{d}.pkl"), "wb"))
            rows.append(dict(scope=scope, region=region, gate=gate, model=f"tree_d{d}",
                             complexity=int(t.get_n_leaves()), fit_acc=round(float((t.predict(X) == y).mean()), 3),
                             woG=round(wg, 3), ppo=round(ppo, 3), heur=round(heur_wog, 3),
                             vs_heur=round(wg - heur_wog, 3), vs_ppo=round(wg - ppo, 3)))
        # FIGS (참고)
        fg = FIGSClassifier(max_rules=30); fg.fit(X, y, sample_weight=w)
        with _suppress_stdout():
            wg = eval_policy(factory, make_tree_policy(fg), eval_eps, SEED + 9000)["mean_R_woG"]
        pickle.dump({"tree": fg, "model": "FIGS_r30"}, open(os.path.join(odir, "FIGS_r30.pkl"), "wb"))
        rows.append(dict(scope=scope, region=region, gate=gate, model="FIGS_r30",
                         complexity=int(fg.complexity_), fit_acc=round(float((fg.predict(X) == y).mean()), 3),
                         woG=round(wg, 3), ppo=round(ppo, 3), heur=round(heur_wog, 3),
                         vs_heur=round(wg - heur_wog, 3), vs_ppo=round(wg - ppo, 3)))
        return dict(ok=True, rows=rows)
    except Exception as e:
        import traceback
        return dict(ok=False, scope=scope, region=region, gate=gate,
                    err=(str(e) + "|" + traceback.format_exc())[:400])


def heur_lookup(nat_occ="results/sigungu_heuristic_best.csv",
                nat_psent="results/sigungu_heuristic_psent_best.csv"):
    import pandas as pd
    h = {"occ": {}, "site": {}}
    for g, f in [("occ", "results/sido_osrm_heuristic_best.csv"),
                 ("site", "results/sido_osrm_heuristic_psent_best.csv")]:
        df = pd.read_csv(os.path.join(REPO, f)); h[g] = dict(zip(df.region, df.reward_wog))
    sg = {"occ": float(pd.read_csv(os.path.join(REPO, nat_occ)).reward_wog.mean()),
          "site": float(pd.read_csv(os.path.join(REPO, nat_psent)).reward_wog.mean())}
    return h, sg


def build_jobs(scopes, gates, eval_eps, depths, nat_base="results/rl/sigungu_nat",
               nat_manifest="scenarios/manifests/sigungu_osrm_manifest.json", nat_tag="A전국",
               nat_heur_occ="results/sigungu_heuristic_best.csv",
               nat_heur_psent="results/sigungu_heuristic_psent_best.csv"):
    # 전국(A) 스코프는 nat_* 로 시나리오축 오버라이드(Kakao 등). 기본값=시군구 OSRM(하위호환).
    sido_m = json.load(open(os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json")))
    sg_m = os.path.join(REPO, nat_manifest)
    h, sg = heur_lookup(nat_heur_occ, nat_heur_psent)
    jobs = []
    for gate in gates:
        suf = "occ" if gate == "occ" else "siteonly"
        if "A" in scopes:
            md = os.path.join(REPO, nat_base, f"ds_ess_woG_{suf}_s0")
            jobs.append((nat_tag, "전국", gate, md, sg_m, sg[gate], eval_eps, depths))
        if "B" in scopes:
            for rg in REGIONS:
                md = os.path.join(REPO, "results/rl/sido", f"{rg}_ds_ess_woG_{suf}_s0")
                jobs.append(("B시도", rg, gate, md, sido_m[rg], h[gate][rg], eval_eps, depths))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--scopes", default="A,B")
    ap.add_argument("--gates", default="occ,site")
    ap.add_argument("--eval_eps", type=int, default=1000)
    ap.add_argument("--depths", default="3,4,6,8,12")
    ap.add_argument("--out", default=os.path.join(REPO, "results/viper/ZOO_RESULTS.csv"))
    # 전국(A) 스코프 시나리오축 오버라이드 (Kakao 등; 기본=시군구 OSRM)
    ap.add_argument("--nat_base", default="results/rl/sigungu_nat")
    ap.add_argument("--nat_manifest", default="scenarios/manifests/sigungu_osrm_manifest.json")
    ap.add_argument("--nat_tag", default="A전국")
    ap.add_argument("--nat_heur_occ", default="results/sigungu_heuristic_best.csv")
    ap.add_argument("--nat_heur_psent", default="results/sigungu_heuristic_psent_best.csv")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    jobs = build_jobs(args.scopes.split(","), args.gates.split(","), args.eval_eps, depths,
                      args.nat_base, args.nat_manifest, args.nat_tag, args.nat_heur_occ, args.nat_heur_psent)
    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out, encoding="utf-8")):
            done.add((r["scope"], r["region"], r["gate"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"[zoo] jobs={len(jobs)} (done={len(done)}, workers={args.workers}, eval={args.eval_eps}, depths={depths})", flush=True)
    if not jobs: print("[zoo] 완료"); return
    jobs.sort(key=lambda j: 0 if j[0].startswith("A전국") else 1)  # 무거운 national 먼저
    fields = ["scope", "region", "gate", "model", "complexity", "fit_acc", "woG", "ppo", "heur", "vs_heur", "vs_ppo"]
    new = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new: w.writeheader(); fout.flush()
    t0 = time.time(); nok = nf = 0
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                for row in r["rows"]: w.writerow(row)
                fout.flush(); nok += 1
                r0 = r["rows"][0]
                d6 = next((x for x in r["rows"] if x["model"] == "tree_d6"), r0)
                print(f"  [{k}/{len(jobs)}] {r0['scope']} {r0['region']} {r0['gate']}: "
                      f"d6 woG={d6['woG']}(vs휴리{d6['vs_heur']:+},vsPPO{d6['vs_ppo']:+},잎{d6['complexity']}) "
                      f"PPO={r0['ppo']} ({time.time()-t0:.0f}s)", flush=True)
            else:
                nf += 1; print(f"  [{k}/{len(jobs)}] FAIL {r['scope']} {r['region']} {r['gate']}: {r['err'][:150]}", flush=True)
    fout.close()
    print(f"\n[zoo] 완료 ok={nok} fail={nf} wall={time.time()-t0:.0f}s out={args.out}", flush=True)


if __name__ == "__main__":
    main()
