"""자원·부하 다축 트레이드오프 스위퍼 — 한 축씩 변주(나머지는 시나리오 기본값 고정)하며
휴리스틱(최근접) vs 부하균형(발송상한 T) woG 를 같은시드 평가. 각 축이 (a)절대 생존(woG)과
(b)정책 격차(=의사결정의 가치)에 미치는 영향을 관찰.

축(런타임 노브, src/sim_src/ScenarioManager.py):
  incident : MCI_INCIDENT_SIZE  사고규모(부하)        기본 100
  capa     : MCI_CAPA_SCALE      병원 용량 스케일       기본 1.0  (수술실수·병상수·max_send ×s)
  amb      : MCI_AMB_NUM         구급차 대수            기본 30
  uav      : MCI_UAV_NUM         UAV 대수(≤착륙 가능 병원수) 기본 25

T=1e9 ≈ 휴리스틱(최근접). 부하균형은 'p_sent<T 최근접'(병원당 정원제).
배경: docs/MCI_종합보고서_최종.md §3.5. 정책=loadbalance_heuristic.make_cap_policy.

예) 시도 6곳 4축 관찰:
  python src/rl_src/tradeoff_sweep.py --scope sido --gate occ \
    --regions 서울,부산,대구,충북,강원,제주 --n_ep 300 \
    --axes "incident:100,200,350,500,700 capa:1.0,0.5,0.3,0.2 amb:10,20,30,40 uav:5,15,25" \
    --Ts 2,4,8,1e9 --out results/tradeoff_multiaxis_sido_occ.csv
"""
import os, sys, json, argparse, csv, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, "src/rl_src")
from multiprocessing import Pool
import numpy as np

H = 47  # 2026-07-02 성남 정정: 46→47
SEED = 11000
SIDO = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
KNOB = {"incident": "MCI_INCIDENT_SIZE", "capa": "MCI_CAPA_SCALE",
        "amb": "MCI_AMB_NUM", "uav": "MCI_UAV_NUM"}


def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential", MCI_GREEN_MASK="1", MCI_REWARD_MODE="woG")
    if g == "occ":
        os.environ.update(MCI_CAP_GATE="occ", MCI_CARED_OBS="1")
    else:
        os.environ.update(MCI_CAP_GATE="psent", MCI_CARED_OBS="0")


def worker(job):
    region, cfg_path, br, gate, axis, value, Ts, n_ep = job
    # 모든 축 노브 초기화 후 이 축만 설정 → 한 축 변주, 나머지 시나리오 기본값.
    for k in KNOB.values():
        os.environ.pop(k, None)
    if axis != "base":
        os.environ[KNOB[axis]] = str(int(value)) if axis in ("incident", "amb", "uav") else str(value)
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
        Theur = max(Ts)
        out = dict(ok=True, region=region, gate=gate, axis=axis, value=value)
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
        return dict(ok=False, region=region, gate=gate, axis=axis, value=value,
                    err=(str(e) + traceback.format_exc())[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="sido", choices=["sido", "sigungu"])
    ap.add_argument("--regions", default="서울,부산,대구,충북,강원,제주")
    ap.add_argument("--gate", default="occ", choices=["occ", "site"])
    ap.add_argument("--axes", default="incident:100,200,350,500,700 capa:1.0,0.5,0.3,0.2 amb:10,20,30,40 uav:5,15,25")
    ap.add_argument("--Ts", default="2,4,8,1e9")
    ap.add_argument("--n_ep", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/tradeoff_multiaxis.csv")
    A = ap.parse_args()

    manifest = ("scenarios/manifests/sido_osrm_manifest.json" if A.scope == "sido"
                else "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
    heur_csv = ("results/sido_osrm_heuristic_best.csv" if A.gate == "occ"
                else "results/sido_osrm_heuristic_psent_best.csv")
    if A.scope == "sigungu":
        heur_csv = ("results/sigungu_heuristic_best.csv" if A.gate == "occ"
                    else "results/sigungu_heuristic_psent_best.csv")
    import pandas as pd
    cfgs = json.load(open(manifest))
    hb = pd.read_csv(heur_csv, encoding="utf-8-sig")
    if A.scope == "sigungu" and "sigcd" in hb.columns:
        hb["__k"] = hb["sigcd"].astype(str)
        best_rule = lambda r: hb.set_index("__k").loc[r.rsplit("_", 1)[1], "best_rule"]
    else:
        best_rule = lambda r: hb.set_index("region").loc[r, "best_rule"]

    regions = A.regions.split(",")
    Ts = [float(x) for x in A.Ts.split(",")]
    # 축 파싱: "incident:100,200 capa:1.0,0.5 ..."
    axis_vals = {}
    for spec in A.axes.split():
        name, vals = spec.split(":")
        axis_vals[name] = [float(x) for x in vals.split(",")]

    jobs = []
    for r in regions:
        try:
            br = best_rule(r)
        except Exception:
            print(f"  [skip] {r}: 휴리 없음", flush=True)
            continue
        jobs.append((r, cfgs[r], br, A.gate, "base", 0, Ts, A.n_ep))   # 기본값 기준점
        for axis, vals in axis_vals.items():
            for v in vals:
                jobs.append((r, cfgs[r], br, A.gate, axis, v, Ts, A.n_ep))
    print(f"[tradeoff-multi] scope={A.scope} gate={A.gate} regions={len(regions)} axes={list(axis_vals)} "
          f"jobs={len(jobs)} Ts={Ts} n_ep={A.n_ep}", flush=True)

    res = []
    t0 = time.time()
    with Pool(A.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if not r["ok"]:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['axis']}={r['value']}: {r['err'][:120]}", flush=True)
            elif k % 20 == 0:
                print(f"  [{k}/{len(jobs)}] {r['region']} {r['axis']}={r['value']} ({time.time()-t0:.0f}s)", flush=True)
    ok = [r for r in res if r["ok"]]
    cols = ["region", "gate", "axis", "value"] + [f"woG_T{t}" for t in Ts] + \
           [c for t in Ts if t != max(Ts) for c in (f"gap_T{t}", f"ci_T{t}")]
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ok:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"저장 {A.out} ({len(ok)}/{len(jobs)} ok)  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
