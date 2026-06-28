"""자원이용률(재난 스트레스) 트레이드오프 스윕 — 의사결정이 생명을 좌우하는 regime 규명.

독립변수 = 재난 스트레스 ρ (= 긴급부하/병원용량). 두 노브로 변주:
  ① MCI_INCIDENT_SIZE  : 사고규모(부하)를 키움 — 발송 게이트는 느슨 유지(깨끗한 축, 권장)
  ② MCI_CAPA_SCALE     : 병원용량을 조임 — 발송 게이트도 같이 닫혀 휴리가 강제분산됨(부차)

각 (region, 스트레스)에서 발송상한 T 여러 값(T=1e9 ≈ 휴리스틱 최근접)을 같은시드 woG paired
평가 → 정책 격차가 스트레스에 따라 어떻게 변하는지, 최적 T 가 어떻게 적응하는지 산출.

근거·배경: docs/MCI_종합보고서_최종.md §3.5. 휴리·부하균형 정책은 loadbalance_heuristic.py.
런타임 노브는 src/sim_src/ScenarioManager.py(setup_patient/setup_hospital)에서 읽음.

예) 시도17 occ, 사고규모 스윕:
  python src/rl_src/tradeoff_sweep.py --scope sido --gate occ \
    --sizes 100,200,350,500 --Ts 2,4,8,16,1e9 --n_ep 1000 --out results/tradeoff_sido_occ.csv
"""
import os, sys, json, argparse, csv, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, "src/rl_src")
from multiprocessing import Pool
import numpy as np

H = 46
SEED = 11000
SIDO = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()


def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential", MCI_GREEN_MASK="1", MCI_REWARD_MODE="woG")
    if g == "occ":
        os.environ.update(MCI_CAP_GATE="occ", MCI_CARED_OBS="1")
    else:
        os.environ.update(MCI_CAP_GATE="psent", MCI_CARED_OBS="0")


def worker(job):
    region, cfg_path, br, gate, isz, capa, Ts, n_ep = job
    if isz:
        os.environ["MCI_INCIDENT_SIZE"] = str(isz)
    else:
        os.environ.pop("MCI_INCIDENT_SIZE", None)
    if capa and capa != 1.0:
        os.environ["MCI_CAPA_SCALE"] = str(capa)
    else:
        os.environ.pop("MCI_CAPA_SCALE", None)
    setgate(gate)
    import torch as th
    th.set_num_threads(1)
    from viper_distill import make_feature_env, _suppress_stdout
    from loadbalance_heuristic import make_cap_policy
    try:
        fac = make_feature_env(cfg_path, None)
        pols = {t: make_cap_policy(br, t, H) for t in Ts}
        W = {t: np.zeros(n_ep) for t in Ts}
        with _suppress_stdout():
            for ep in range(n_ep):
                for t in Ts:
                    env = fac(seed=SEED + ep)
                    ro, _ = env.reset(seed=SEED + ep)
                    done = False
                    w = 0.0
                    pol = pols[t]
                    while not done:
                        mask = np.asarray(env.action_masks(), bool)
                        a = pol(ro, mask, env.unwrapped)
                        ro, r, te, tr, info = env.step(a)
                        w += info.get("r_woG", 0.0)
                        done = te or tr
                    W[t][ep] = w
        Theur = max(Ts)  # 가장 큰 T(=1e9) 를 휴리스틱(최근접) 기준선으로
        out = dict(ok=True, region=region, gate=gate, incident=isz or 0, capa=capa or 1.0)
        for t in Ts:
            out[f"woG_T{t}"] = float(W[t].mean())
        for t in Ts:
            if t == Theur:
                continue
            d = W[t] - W[Theur]
            out[f"gap_T{t}"] = float(d.mean())
            out[f"ci_T{t}"] = 1.96 * float(d.std(ddof=1)) / np.sqrt(n_ep)
        return out
    except Exception as e:
        import traceback
        return dict(ok=False, region=region, gate=gate, incident=isz or 0, capa=capa or 1.0,
                    err=(str(e) + traceback.format_exc())[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sido", choices=["sido", "sigungu"])
    ap.add_argument("--regions", default="", help="콤마구분 부분집합(미지정=scope 전체)")
    ap.add_argument("--gate", default="occ", choices=["occ", "site"])
    ap.add_argument("--sizes", default="100,200,350,500", help="MCI_INCIDENT_SIZE 스윕(0=원본)")
    ap.add_argument("--capa_scales", default="", help="MCI_CAPA_SCALE 스윕(미지정=1.0)")
    ap.add_argument("--Ts", default="2,4,8,16,1e9", help="발송상한 T(1e9≈휴리)")
    ap.add_argument("--n_ep", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/tradeoff_sweep.csv")
    A = ap.parse_args()

    manifest = ("scenarios/manifests/sido_osrm_manifest.json" if A.scope == "sido"
                else "scenarios/manifests/sigungu_osrm_manifest.json")
    heur_csv = ("results/sido_osrm_heuristic_best.csv" if A.gate == "occ"
                else "results/sido_osrm_heuristic_psent_best.csv")
    if A.scope == "sigungu":
        heur_csv = ("results/sigungu_heuristic_best.csv" if A.gate == "occ"
                    else "results/sigungu_heuristic_psent_best.csv")
    import pandas as pd
    cfgs = json.load(open(manifest))
    hb = pd.read_csv(heur_csv, encoding="utf-8-sig")
    # 시군구 휴리 CSV는 region=이름만(동명구 충돌) → sigcd 로 매칭
    if A.scope == "sigungu" and "sigcd" in hb.columns:
        hb["__key"] = hb["sigcd"].astype(str)
        def best_rule(region):
            sgcd = region.rsplit("_", 1)[1]
            return hb.set_index("__key").loc[sgcd, "best_rule"]
    else:
        def best_rule(region):
            return hb.set_index("region").loc[region, "best_rule"]

    regions = (A.regions.split(",") if A.regions else
               (SIDO if A.scope == "sido" else list(cfgs.keys())))
    sizes = [int(x) for x in A.sizes.split(",")] if A.sizes else [0]
    capas = [float(x) for x in A.capa_scales.split(",")] if A.capa_scales else [1.0]
    Ts = [float(x) for x in A.Ts.split(",")]

    jobs = []
    for r in regions:
        try:
            br = best_rule(r)
        except Exception:
            print(f"  [skip] {r}: 휴리 best_rule 없음", flush=True)
            continue
        for sz in sizes:
            for ca in capas:
                jobs.append((r, cfgs[r], br, A.gate, sz, ca, Ts, A.n_ep))
    print(f"[tradeoff] scope={A.scope} gate={A.gate} jobs={len(jobs)} Ts={Ts} n_ep={A.n_ep}", flush=True)

    res = []
    t0 = time.time()
    with Pool(A.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
                bt = max(Ts, key=lambda t: r[f"woG_T{t}"])
                btxt = "heur" if bt > 1e8 else f"T{int(bt)}"
                print(f"  [{k}/{len(jobs)}] {r['region']} N={r['incident']} s={r['capa']}: "
                      f"best={btxt}({r[f'woG_T{bt}']:.1f}) ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']} N={r['incident']}: {r['err'][:140]}", flush=True)

    ok = [r for r in res if r["ok"]]
    cols = ["region", "gate", "incident", "capa"] + \
           [f"woG_T{t}" for t in Ts] + \
           [c for t in Ts if t != max(Ts) for c in (f"gap_T{t}", f"ci_T{t}")]
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ok:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"저장 {A.out} ({len(ok)}/{len(jobs)} ok)  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
