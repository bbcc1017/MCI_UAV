"""다축 트레이드오프 시뮬로그 — 4정책(휴리·부하균형·RL·트리) × 4축(incident/capa/amb/uav)을
**고정 인프라 surge** 로 재시뮬(시나리오 재생성 없음, 런타임 노브). SEED=11000 1000ep,
결정·에피소드·병원부하 풀 캡처(sim_logger.py 와 동일 포맷 + axis/value 컬럼).

축(런타임 노브, ScenarioManager): incident=MCI_INCIDENT_SIZE, capa=MCI_CAPA_SCALE,
amb=MCI_AMB_NUM, uav=MCI_UAV_NUM. 한 축만 변주, 나머지 config 기본값(=100명용 provisioned 인프라).
⚠️ RL·트리는 baseline(incident100·amb30·uav25·capa1) 학습본 → 축 변주 시 OOD(자원변화 일반화 테스트).
   휴리·부하균형은 규칙기반이라 OOD 아님. 시도 스코프(지역별 OSRM 모델 + B시도 트리 + 시도 OSRM 휴리).

산출(results/viper/simlog_tradeoff/):
  decisions_<region>_<axis><value>_<gate>_<policy>.csv.gz
  episodes.csv (region,gate,axis,value,policy,ep,woG,raw,...)  hospital_loads.csv

예: python src/rl_src/tradeoff_simlog.py --regions 서울,부산,대구,충북,강원,제주 --gates occ \
      --policies heur,lb4,rl,tree_d6 --n_ep 1000 --workers 14
"""
import os, sys, argparse, csv, gzip, time, pickle, warnings, json
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000; H = 46; ND = H + 1; NM = 2
OUT = os.path.join(REPO, "results/viper/simlog_tradeoff")
MANIFEST = os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json")
MODEL_BASE = "results/rl/sido"; TREE_TAG = "B시도"
HEUR = {"occ": "results/sido_osrm_heuristic_best.csv", "site": "results/sido_osrm_heuristic_psent_best.csv"}
KNOB = {"incident": "MCI_INCIDENT_SIZE", "capa": "MCI_CAPA_SCALE", "amb": "MCI_AMB_NUM", "uav": "MCI_UAV_NUM"}


def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential", MCI_GREEN_MASK="1", MCI_REWARD_MODE="woG")
    if g == "occ":
        os.environ.update(MCI_CAP_GATE="occ", MCI_CARED_OBS="1")
    else:
        os.environ.update(MCI_CAP_GATE="psent", MCI_CARED_OBS="0")


def set_axis(axis, value):
    for k in KNOB.values():
        os.environ.pop(k, None)
    if axis != "base":
        os.environ[KNOB[axis]] = str(int(value)) if axis in ("incident", "amb", "uav") else str(value)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    return 0.0 if s == 0 else float((2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * s) / (n * s))


def worker(job):
    region, gate, axis, value, policy, n_ep = job
    set_axis(axis, value); setgate(gate)
    import torch as th; th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from viper_distill import make_feature_env, load_vecnorm, make_tree_policy, _suppress_stdout
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy
    import pandas as pd
    try:
        suf = "occ" if gate == "occ" else "siteonly"
        md = os.path.join(REPO, f"{MODEL_BASE}/{region}_ds_ess_woG_{suf}_s0")
        cfg = json.load(open(MANIFEST))[region]
        need_norm = policy in ("rl",) or policy.startswith("tree")
        mean = std = clip = None
        if need_norm:
            mean, std, clip = load_vecnorm(os.path.join(md, "vecnormalize.pkl"))
        fac = make_feature_env(cfg, None)
        br = pd.read_csv(os.path.join(REPO, HEUR[gate])).set_index("region").loc[region, "best_rule"]
        if policy == "rl":
            model = MaskablePPO.load(os.path.join(md, "final_model.zip"), device="cpu")
            def act(ro, mask, env): return int(model.predict(np.clip((np.asarray(ro, np.float32) - mean) / std, -clip, clip), action_masks=mask, deterministic=True)[0])
        elif policy == "heur":
            hp = make_heuristic_policy(br)
            def act(ro, mask, env): return hp(ro, mask, env)
        elif policy.startswith("lb"):
            T = float(policy[2:]) if len(policy) > 2 else 4.0
            lp = make_cap_policy(br, T, H)
            def act(ro, mask, env): return lp(ro, mask, env)
        else:  # tree_dN
            dN = policy.split("_d")[1]
            tr = pickle.load(open(os.path.join(REPO, f"results/viper/zoo/{TREE_TAG}_{region}_{gate}/tree_d{dN}.pkl"), "rb"))["tree"]
            tp = make_tree_policy(tr)
            def act(ro, mask, env): return tp(np.clip((np.asarray(ro, np.float32) - mean) / std, -clip, clip), mask, env)
        with _suppress_stdout():
            env0 = fac(seed=SEED); env0.reset(seed=SEED)
        hp_props = env0.unwrapped.en_manager.en_properties['hospital']
        max_capa = np.asarray(hp_props['hos_max_capa'], float)
        tier3 = (np.asarray(hp_props.get('hos_tier', np.zeros(H))) == 3).astype(int)
        os.makedirs(OUT, exist_ok=True)
        tag = f"{axis}{('%g' % value) if axis != 'base' else ''}"
        dec_path = os.path.join(OUT, f"decisions_{region}_{tag}_{gate}_{policy}.csv.gz")
        arrivals = np.zeros(H); peak_q = np.zeros(H); peak_occ = np.zeros(H); sat_steps = np.zeros(H)
        ep_rows = []
        with gzip.open(dec_path, "wt", newline="", encoding="utf-8") as fdec, _suppress_stdout():
            wd = csv.writer(fdec)
            wd.writerow(["ep", "step", "time", "cls", "dest", "mode", "hosp", "cap_remain", "eta_amb", "eta_uav", "eta_rank", "n_avail", "r_woG"])
            for ep in range(n_ep):
                env = fac(seed=SEED + ep); ro, _ = env.reset(seed=SEED + ep); done = False
                wog = 0.0; raw = 0.0; step = 0; ntr = 0; nuav = 0; namb = 0; eta_ranks = []; ep_arr = np.zeros(H)
                while not done:
                    mask = np.asarray(env.action_masks(), bool)
                    a = act(ro, mask, env.unwrapped)
                    c = a // (ND * NM); rem = a % (ND * NM); dst = rem // NM; m = rem % NM
                    rr = np.asarray(ro, np.float32); HF = rr[:H * 4].reshape(H, 4)
                    erk = -1; navail = 0; hosp = -1; capr = eta_a = eta_u = -1
                    if dst > 0:
                        hosp = dst - 1; capr = float(HF[hosp, 1]); eta_a = float(HF[hosp, 2]); eta_u = float(HF[hosp, 3])
                        cap = HF[:, 1]; tier = HF[:, 0]; eta = HF[:, 2] if m == 0 else HF[:, 3]
                        av = cap > 0
                        if c == 0: av = av & (tier > 0.5)
                        navail = int(av.sum())
                        if navail > 0: erk = int((eta[av] < eta[hosp]).sum()) + 1; eta_ranks.append(erk)
                        arrivals[hosp] += 1; ep_arr[hosp] += 1; ntr += 1
                        if m == 1: nuav += 1
                        else: namb += 1
                    ro, r, te, tr, info = env.step(a); rw = info.get("r_woG", 0.0); wog += rw; raw += r; done = te or tr; step += 1
                    hs = env.unwrapped.en_manager.get_full_obs()['h_states']
                    q = hs[:, 1]; occ = hs[:, 2]
                    peak_q = np.maximum(peak_q, q); peak_occ = np.maximum(peak_occ, occ)
                    sat_steps += (occ >= max_capa).astype(float)
                    wd.writerow([ep, step, round(float(info.get('time', 0)), 2), int(c), int(dst), int(m), int(hosp),
                                 round(capr, 3), round(eta_a, 3), round(eta_u, 3), erk, navail, round(rw, 4)])
                used = int((ep_arr > 0).sum())
                # PDR_woG = 1 - woG/preventable_woG (예방가능 총합 정규화 → 사고규모 무관 비교).
                pv = float(getattr(env.unwrapped, "preventable", 0.0)); pvw = float(getattr(env.unwrapped, "preventable_woG", 0.0))
                pdr = round(1 - raw / pv, 4) if pv > 0 else 0.0
                pdr_woG = round(1 - wog / pvw, 4) if pvw > 0 else 0.0
                ep_rows.append(dict(region=region, gate=gate, axis=axis, value=value, policy=policy, ep=ep,
                    woG=round(wog, 3), raw=round(raw, 2), PDR_woG=pdr_woG, PDR=pdr, preventable_woG=round(pvw, 2),
                    time_end=round(float(info.get('time', 0)), 2),
                    n_transport=ntr, n_used_hosp=used, gini=round(gini(ep_arr), 3),
                    max_share=round(float(ep_arr.max() / max(ep_arr.sum(), 1)), 3),
                    mean_eta_rank=round(float(np.mean(eta_ranks)) if eta_ranks else 0, 2), n_uav=nuav, n_amb=namb))
        hosp_rows = [dict(region=region, gate=gate, axis=axis, value=value, policy=policy, hosp=h, tier3=int(tier3[h]),
                          arrivals=int(arrivals[h]), peak_queue=int(peak_q[h]), peak_occ=int(peak_occ[h]),
                          sat_steps=int(sat_steps[h]), max_capa=int(max_capa[h])) for h in range(H)]
        return dict(ok=True, region=region, gate=gate, axis=axis, value=value, policy=policy, ep_rows=ep_rows, hosp_rows=hosp_rows)
    except Exception as e:
        import traceback
        return dict(ok=False, region=region, gate=gate, axis=axis, value=value, policy=policy,
                    err=(str(e) + "|" + traceback.format_exc())[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="서울,부산,대구,충북,강원,제주")
    ap.add_argument("--gates", default="occ")
    ap.add_argument("--policies", default="heur,lb4,rl,tree_d6")
    ap.add_argument("--axes", default="incident:200,350,500 capa:0.5,0.3 amb:10,20 uav:5,15")
    ap.add_argument("--n_ep", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    # 시나리오축 오버라이드(Kakao 등; 기본=시도 OSRM). fork 로 워커 전파.
    ap.add_argument("--model_base", default=""); ap.add_argument("--manifest", default="")
    ap.add_argument("--tree_tag", default=""); ap.add_argument("--heur_occ", default=""); ap.add_argument("--heur_psent", default="")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    global OUT, MANIFEST, MODEL_BASE, TREE_TAG, HEUR
    if A.model_base: MODEL_BASE = A.model_base
    if A.manifest: MANIFEST = os.path.join(REPO, "scenarios/manifests", A.manifest) if not os.path.isabs(A.manifest) else A.manifest
    if A.tree_tag: TREE_TAG = A.tree_tag
    if A.heur_occ: HEUR = dict(HEUR); HEUR["occ"] = A.heur_occ
    if A.heur_psent: HEUR = dict(HEUR); HEUR["site"] = A.heur_psent
    if A.out: OUT = os.path.join(REPO, "results/viper", A.out) if not os.path.isabs(A.out) else A.out
    os.makedirs(OUT, exist_ok=True)
    ep_csv = os.path.join(OUT, "episodes.csv"); hl_csv = os.path.join(OUT, "hospital_loads.csv")
    done = set()
    if os.path.exists(ep_csv):
        for r in csv.DictReader(open(ep_csv, encoding="utf-8")):
            done.add((r["region"], r["gate"], r["axis"], r["value"], r["policy"]))
    regions = A.regions.split(","); gates = A.gates.split(","); policies = A.policies.split(",")
    # 축점: base(전부 기본) + 각 축의 비기본값
    points = [("base", 0.0)]
    for spec in A.axes.split():
        name, vals = spec.split(":")
        for v in vals.split(","):
            points.append((name, float(v)))
    jobs = [(r, g, ax, v, p, A.n_ep) for g in gates for r in regions for (ax, v) in points for p in policies
            if (r, g, ax, str(v), p) not in done]
    print(f"[tradeoff-simlog] jobs={len(jobs)} (done={len(done)}, regions={len(regions)}, points={len(points)}, "
          f"policies={policies}, n_ep={A.n_ep}, out={OUT})", flush=True)
    if not jobs:
        print("[tradeoff-simlog] 완료"); return
    ep_f = open(ep_csv, "a", newline="", encoding="utf-8"); hl_f = open(hl_csv, "a", newline="", encoding="utf-8")
    epw = hlw = None; t0 = time.time(); nok = nf = 0
    with Pool(A.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                if epw is None:
                    epw = csv.DictWriter(ep_f, fieldnames=list(r["ep_rows"][0].keys()))
                    if os.path.getsize(ep_csv) == 0: epw.writeheader()
                    hlw = csv.DictWriter(hl_f, fieldnames=list(r["hosp_rows"][0].keys()))
                    if os.path.getsize(hl_csv) == 0: hlw.writeheader()
                epw.writerows(r["ep_rows"]); hlw.writerows(r["hosp_rows"]); ep_f.flush(); hl_f.flush(); nok += 1
                mw = np.mean([x["woG"] for x in r["ep_rows"]]); mg = np.mean([x["gini"] for x in r["ep_rows"]])
                print(f"  [{k}/{len(jobs)}] {r['region']} {r['axis']}={r['value']} {r['policy']}: woG{mw:.1f} gini{mg:.2f} ({time.time()-t0:.0f}s)", flush=True)
            else:
                nf += 1; print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['axis']}={r['value']} {r['policy']}: {r['err'][:110]}", flush=True)
    print(f"[tradeoff-simlog] done ok={nok} fail={nf} wall={time.time()-t0:.0f}s out={OUT}", flush=True)


if __name__ == "__main__":
    main()
