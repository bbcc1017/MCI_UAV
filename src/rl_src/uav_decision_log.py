"""UAV 운용규칙 추출용 결정로그 수집 (플랜 v2 Phase 3-A).

레벨별 RL 모델(uav{0,5,10,15,26})을 각 시나리오(MCI_UAV_NUM=k)에서 롤아웃하며 **모든 이송 결정**을
컨텍스트와 함께 기록 → "언제·누구를·어디로 UAV 를 쓰는가, 대수가 늘면 어떻게 변하는가" 분석.

설계 원칙(인프라 매핑 결론):
  - obs 슬라이싱(`ro[:H*4]`, F=4/7 버그원천) 대신 dest 특징은 `get_static_eta()`(en_properties)와
    `en_properties['hospital']`(tier/helipad)에서 취득 → obs 레이아웃 완전 비의존.
  - action 디코드는 mask 길이 기반(192=2·48·2 / 96=2·48 uav0 auto-pin) → 레벨 무관.
  - 컨텍스트(p_sent·in_flight·대기환자)는 결정 시점 dict obs(en_manager.get_full_obs)에서.

기록 단위 = 1 이송 결정(dest≥1). 컬럼:
  level,region,ep,step,time, cls(0R/1Y), dest, mode(0AMB/1UAV), dest_tier3, dest_helipad,
  eta_amb, eta_uav (정규화 최근접=1), eta_gap(=eta_amb−eta_uav; +면 UAV가 빠름),
  eta_amb_rank(적격 중 순위 1=최근접), p_sent_dest, in_flight_dest, occ_dest,
  n_red_wait, n_yellow_wait, rho, n_uav_avail
출력: results/rl/redesign/uav_decisions.csv.gz

예: PYTHONIOENCODING=utf-8 python src/rl_src/uav_decision_log.py --n_eps 200 --workers 34
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import gzip
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
LEVELS = {0: "uav0_deepsets", 5: "uav5_pointer", 10: "uav10_pointer",
          15: "uav15_pointer", 26: "L3_pointer"}
COLS = ["level", "region", "ep", "step", "time", "cls", "dest", "mode", "dest_tier3",
        "dest_helipad", "eta_amb", "eta_uav", "eta_gap", "eta_amb_rank", "p_sent_dest",
        "in_flight_dest", "occ_dest", "n_red_wait", "n_yellow_wait", "rho", "n_uav_avail"]


def _decode(a, mask_len, H):
    """flat action → (c, dest, mode). mask 길이로 192/96 구분."""
    nd = H + 1
    if mask_len == 2 * nd * 2:      # 192: c·48·2
        return a // (nd * 2), (a % (nd * 2)) // 2, a % 2
    return a // nd, a % nd, 0        # 96: c·48, mode=AMB 고정


def worker(job):
    level, region, cfg, n_eps = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    os.environ["MCI_OBS_VARIANT"] = "essential+load"
    os.environ["MCI_UAV_NUM"] = str(level)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from evaluate import ppo_policy
    from loadbalance_heuristic import get_static_eta
    from EntityManager import EntityManager
    try:
        mdir = os.path.join(REPO, "results/rl/redesign", f"{LEVELS[level]}_s0")
        rows = []
        with _suppress_stdout():
            model = MaskablePPO.load(os.path.join(mdir, "final_model.zip"), device="cpu")
            vn = os.path.join(mdir, "vecnormalize.pkl")
            norm = load_vecnorm(vn) if os.path.exists(vn) else None
            fac = make_feature_env(cfg, norm)
            env = fac(seed=SEED)
            H = env.unwrapped.en_manager.en_properties['hospital']['hos_num']
            hp = env.unwrapped.en_manager.en_properties['hospital']
            tier3 = (np.asarray(hp['hos_tier']).reshape(-1) == 3).astype(int)
            helip = np.zeros(H, int)
            hi = np.asarray(hp.get('hos_helipad_idx', []), int)
            if hi.size:
                helip[hi] = 1
            eta_amb, eta_uav = get_static_eta(env.unwrapped, H)
            pol = ppo_policy(model)
            for ep in range(n_eps):
                obs, _ = env.reset(seed=SEED + ep)
                done = False
                step = 0
                while not done:
                    mask = np.asarray(env.action_masks(), bool)
                    a = pol(obs, mask, env.unwrapped)
                    c, dest, mode = _decode(int(a), len(mask), H)
                    if dest >= 1:  # 이송 결정만 기록(stay 제외)
                        u = env.unwrapped
                        dobs = u.en_manager.get_full_obs()
                        p_sent = np.asarray(dobs['p_sent'], float)
                        infl = EntityManager.in_flight_by_hospital(dobs, H)
                        hs = np.asarray(dobs['h_states'], float)
                        ps = np.asarray(dobs['p_states'])
                        # 대기(현장) R/Y: class 0/1 & rescued & 미이송(move_start==0)
                        rw = int(np.sum((ps[:, 0] == 0) & (ps[:, 1] == 1) & (ps[:, 2] == 0)))
                        yw = int(np.sum((ps[:, 0] == 1) & (ps[:, 1] == 1) & (ps[:, 2] == 0)))
                        di = dest - 1
                        eta = eta_amb if mode == 0 else eta_uav
                        # 선택 dest 의 ETA 순위(선택 mode 기준, 전체 병원 중 1=최근접) — 해석용
                        rank = int(np.sum(eta < eta[di]) + 1)
                        cap_tot = float(np.maximum(np.asarray(hp['hos_max_send']) - (hs[:, 2] + infl), 0).sum())
                        rho = (rw + yw) / (cap_tot + 1.0)
                        n_uav_avail = int(np.sum((np.asarray(dobs.get('uav_states', np.zeros((0, 3))))[:, 1] <= 1e-6)) if level > 0 else 0)
                        rows.append([level, region, ep, step, round(float(u.ev_manager.time), 2),
                                     c, dest, mode, int(tier3[di]), int(helip[di]),
                                     round(float(eta_amb[di]), 4), round(float(eta_uav[di]), 4),
                                     round(float(eta_amb[di] - eta_uav[di]), 4), rank,
                                     round(float(p_sent[di]), 1), int(infl[di]), round(float(hs[di, 2]), 1),
                                     rw, yw, round(rho, 4), n_uav_avail])
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    step += 1
        return dict(level=level, region=region, ok=True, rows=rows)
    except Exception as e:
        import traceback
        return dict(level=level, region=region, ok=False, err=(str(e) + traceback.format_exc())[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--levels", default="0,5,10,15,26")
    ap.add_argument("--regions", default="")
    ap.add_argument("--n_eps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=34)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/uav_decisions.csv.gz"))
    A = ap.parse_args()

    manifest = json.load(open(A.manifest, encoding="utf-8"))
    levels = [int(x) for x in A.levels.split(",")]
    regions = A.regions.split(",") if A.regions else [r for r in SIDO17 if r in manifest]
    jobs = [(lv, rg, manifest[rg], A.n_eps) for lv in levels for rg in regions]
    print(f"[uav_dlog] levels={levels} regions={len(regions)} jobs={len(jobs)} n_eps={A.n_eps}", flush=True)

    t0 = time.time()
    n_rows = 0
    with gzip.open(A.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
            for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
                if r["ok"]:
                    w.writerows(r["rows"]); n_rows += len(r["rows"])
                    print(f"  [{k}/{len(jobs)}] uav{r['level']} {r['region']}: {len(r['rows'])}결정 "
                          f"(누적 {n_rows}, {time.time()-t0:.0f}s)", flush=True)
                else:
                    print(f"  [{k}/{len(jobs)}] FAIL uav{r['level']} {r['region']}: {r['err'][:180]}", flush=True)
    print(f"\n저장 {A.out}  총 {n_rows}결정  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
