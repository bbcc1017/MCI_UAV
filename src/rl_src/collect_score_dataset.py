"""스코어 정책 학습용 데이터셋 수집 (플랜 v2 추출 트랙 B1).

최강 RL(포인터 head)을 deterministic 롤아웃하며, 각 **이송 결정**(dest≥1)마다 후보 (h,m) 의
특징 φ(score_features)·복원 스코어 S·criticality 를 기록한다. 후속 단계(B2 조건부로짓 MLE)가
이 데이터로 `S ≈ w·φ` 의 w 를 적합하고, B3 CEM 이 closed-loop 로 미세조정한다.

S 복원(핵심): 포인터 head 의 로짓은 `L[c,d,m]=f_class(ctx)+S[d,m]+g_mode(ctx)`. 마스크 미적용
로짓에서 **(c,m) 고정 차분** `L[c,d,m]−L[c,0,m] = S[d,m]−S[0,m]` 로 f_class·g_mode 를 상쇄해
목적지 랭킹 S 를 (모드별 상수 오프셋 제외) 복원한다. get_distribution(마스크 없이) 이 반환하는
정규화 로짓(=log-prob)은 전 액션 공통 상수만 다르므로 이 차분에는 영향이 없다(불변).

설계(uav_decision_log 원형 승계):
  - obs 슬라이싱 금지 — φ 는 score_features(dict obs·en_properties 재계산), 코덱은 mask 길이 기반.
  - deterministic 롤아웃은 ppo_policy(=model.predict deterministic), 정규화 obs(_NormObs) 사용.
  - loggap criticality = viper_distill.make_weight_fn(model,"loggap")(마스크 valid action 로짓폭).

출력: long-format npz(가변 적격수). 배열:
  phi (N_cand,K) · S (N_cand,) · cand_h (N_cand,) · cand_m (N_cand,) · chosen (N_cand, bool)
  offsets (n_dec+1,)  — 결정 d 의 후보 = [offsets[d]:offsets[d+1]] (CSR)
  region_id/ep/step/cls/mode_chosen/loggap (각 n_dec,) · region_names · phi_names

--sanity: 선택 병원이 '복원 S 의 (선택 모드 내) argmax' 인 비율 ≥99% 검증(복원 정합).

예 스모크: PYTHONIOENCODING=utf-8 python src/rl_src/collect_score_dataset.py \
  --model_dir results/rl/redesign/L3_pointer_s0 --regions 서울,강원 --n_eps 2 --workers 2 \
  --out /tmp/claude-1002/score_ds_smoke.npz --sanity
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()


def _codec(mask_len, H):
    """flat encode/decode 쌍 (mask 길이로 192/96 자동)."""
    from loadbalance_heuristic import _codec_from_mask
    enc = _codec_from_mask(mask_len, H)
    nd = H + 1
    if mask_len == 2 * nd * 2:
        dec = lambda a: (a // (nd * 2), (a % (nd * 2)) // 2, a % 2)
    else:
        dec = lambda a: (a // nd, a % nd, 0)
    return enc, dec


def worker(job):
    region, cfg, model_dir, n_eps, uav_num = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    os.environ["MCI_OBS_VARIANT"] = "essential+load"
    if uav_num is not None:
        os.environ["MCI_UAV_NUM"] = str(uav_num)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (로드 전 필수)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, make_weight_fn, _suppress_stdout
    from evaluate import ppo_policy
    from score_features import build_phi, build_ctx, compute_static, K_PHI
    try:
        rows_phi, rows_S, rows_h, rows_m, rows_ch = [], [], [], [], []
        d_ep, d_step, d_cls, d_mode, d_loggap, d_ncand = [], [], [], [], [], []
        with _suppress_stdout():
            model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
            vn = os.path.join(model_dir, "vecnormalize.pkl")
            norm = load_vecnorm(vn) if os.path.exists(vn) else None
            fac = make_feature_env(cfg, norm)
            env = fac(seed=SEED)
            u = env.unwrapped
            H = int(u.en_manager.en_properties['hospital']['hos_num'])
            static = compute_static(u)
            pol = ppo_policy(model)
            wfn = make_weight_fn(model, "loggap")

            def _raw_logits(obs):
                # 마스크 미적용 정규화 로짓(=log-prob) — (c,m)차분에 공통상수 소거되어 S 복원 무영향
                ot = th.as_tensor(np.asarray(obs, np.float32), device=model.device).unsqueeze(0)
                with th.no_grad():
                    dist = model.policy.get_distribution(ot)
                    return dist.distribution.logits.squeeze(0).cpu().numpy()

            for ep in range(n_eps):
                obs, _ = env.reset(seed=SEED + ep)
                done = False
                step = 0
                while not done:
                    mask = np.asarray(env.action_masks(), bool)
                    a = int(pol(obs, mask, u))
                    enc, dec = _codec(len(mask), H)
                    c, dest, m_ch = dec(a)
                    if dest >= 1:   # 이송 결정만
                        has_uav = (len(mask) == 2 * (H + 1) * 2)
                        elig_amb = [i for i in range(H) if mask[enc(c, i + 1, 0)]]
                        elig_uav = [i for i in range(H) if has_uav and mask[enc(c, i + 1, 1)]]
                        cand = [(i, 0) for i in elig_amb] + [(i, 1) for i in elig_uav]
                        cand = np.asarray(cand, dtype=int)
                        ctx = build_ctx(u, static=static)
                        phi = build_phi(u, c, None, cand, ctx=ctx)          # (n_cand, K)
                        logits = _raw_logits(obs)
                        # S_rel[h,m] = L[c,d=h+1,m] − L[c,0,m]  (모드별 stay 기준 상대 스코어)
                        S = np.array([logits[enc(c, hh + 1, mm)] - logits[enc(c, 0, mm)]
                                      for hh, mm in cand], dtype=np.float32)
                        chosen = np.array([(hh == dest - 1 and mm == m_ch) for hh, mm in cand],
                                          dtype=bool)
                        rows_phi.append(phi.astype(np.float32))
                        rows_S.append(S)
                        rows_h.append(cand[:, 0].astype(np.int16))
                        rows_m.append(cand[:, 1].astype(np.int8))
                        rows_ch.append(chosen)
                        d_ep.append(ep); d_step.append(step); d_cls.append(int(c))
                        d_mode.append(int(m_ch)); d_loggap.append(float(wfn(obs, mask)))
                        d_ncand.append(int(cand.shape[0]))
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    step += 1
        pack = dict(
            region=region, ok=True,
            phi=(np.vstack(rows_phi) if rows_phi else np.zeros((0, K_PHI), np.float32)),
            S=(np.concatenate(rows_S) if rows_S else np.zeros(0, np.float32)),
            cand_h=(np.concatenate(rows_h) if rows_h else np.zeros(0, np.int16)),
            cand_m=(np.concatenate(rows_m) if rows_m else np.zeros(0, np.int8)),
            chosen=(np.concatenate(rows_ch) if rows_ch else np.zeros(0, bool)),
            ep=np.asarray(d_ep, np.int32), step=np.asarray(d_step, np.int32),
            cls=np.asarray(d_cls, np.int8), mode_chosen=np.asarray(d_mode, np.int8),
            loggap=np.asarray(d_loggap, np.float32), ncand=np.asarray(d_ncand, np.int32),
        )
        return pack
    except Exception as e:
        import traceback
        return dict(region=region, ok=False, err=(str(e) + traceback.format_exc())[:500])


def _sanity(offsets, cand_m, S, chosen, mode_chosen):
    """선택 병원이 '복원 S 의 (선택 모드 내) argmax' 인 결정 비율.

    포인터 head 는 valid (d,m) 전체서 argmax → 선택 모드 m* 내에서는 d* 가 S[·,m*] 최대여야
    한다(모드간 g_mode·S[0,m] 오프셋은 복원 S 에서 소거되므로 모드 내 비교만 정합)."""
    import numpy as np
    n_dec = len(offsets) - 1
    ok = tot = 0
    for d in range(n_dec):
        s, e = offsets[d], offsets[d + 1]
        if e <= s:
            continue
        mc = mode_chosen[d]
        sel = np.flatnonzero(cand_m[s:e] == mc)
        if sel.size == 0:
            continue
        tot += 1
        Ssel = S[s:e][sel]
        ch = chosen[s:e][sel]
        if ch.any() and sel[int(np.argmax(Ssel))] == sel[int(np.flatnonzero(ch)[0])]:
            ok += 1
    return ok, tot


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/L3_pointer_s0"))
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--regions", default="", help="쉼표 구분(미지정 시 매니페스트 내 SIDO17 전부)")
    ap.add_argument("--n_eps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--uav_num", type=int, default=None, help="MCI_UAV_NUM 오버라이드(기본 미설정=시나리오값)")
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/score_dataset.npz"))
    ap.add_argument("--sanity", action="store_true")
    A = ap.parse_args()

    manifest = json.load(open(A.manifest, encoding="utf-8"))
    regions = [r for r in A.regions.split(",") if r] if A.regions else [r for r in SIDO17 if r in manifest]
    jobs = [(rg, manifest[rg], A.model_dir, A.n_eps, A.uav_num) for rg in regions]
    print(f"[collect_score] model={A.model_dir} regions={len(regions)} n_eps={A.n_eps} jobs={len(jobs)}",
          flush=True)

    t0 = time.time()
    packs = []
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                packs.append(r)
                print(f"  [{k}/{len(jobs)}] {r['region']}: 결정 {len(r['ncand'])} "
                      f"후보 {r['phi'].shape[0]} ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}", flush=True)

    if not packs:
        print("❌ 수집된 결정 없음", flush=True)
        sys.exit(1)

    # 지역 id 순서(잡 순서 무관하게 안정) + 결정별 region_id 부여
    rname = [r for r in regions if any(p["region"] == r for p in packs)]
    rid = {r: i for i, r in enumerate(rname)}
    packs.sort(key=lambda p: rid[p["region"]])

    phi = np.vstack([p["phi"] for p in packs])
    S = np.concatenate([p["S"] for p in packs])
    cand_h = np.concatenate([p["cand_h"] for p in packs])
    cand_m = np.concatenate([p["cand_m"] for p in packs])
    chosen = np.concatenate([p["chosen"] for p in packs])
    ncand = np.concatenate([p["ncand"] for p in packs])
    offsets = np.concatenate([[0], np.cumsum(ncand)]).astype(np.int64)
    ep = np.concatenate([p["ep"] for p in packs])
    step = np.concatenate([p["step"] for p in packs])
    cls = np.concatenate([p["cls"] for p in packs])
    mode_chosen = np.concatenate([p["mode_chosen"] for p in packs])
    loggap = np.concatenate([p["loggap"] for p in packs])
    region_id = np.concatenate([np.full(len(p["ncand"]), rid[p["region"]], np.int16) for p in packs])

    from score_features import PHI_NAMES
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    np.savez_compressed(
        A.out, phi=phi, S=S, cand_h=cand_h, cand_m=cand_m, chosen=chosen,
        offsets=offsets, region_id=region_id, ep=ep, step=step, cls=cls,
        mode_chosen=mode_chosen, loggap=loggap,
        region_names=np.array(rname), phi_names=np.array(PHI_NAMES),
    )
    n_dec = len(offsets) - 1
    print(f"\n저장 {A.out}", flush=True)
    print(f"  결정 {n_dec}  후보총 {phi.shape[0]}  φ차원 {phi.shape[1]}  "
          f"평균후보/결정 {phi.shape[0]/max(n_dec,1):.1f}  wall={time.time()-t0:.0f}s", flush=True)
    print(f"  스키마: phi{phi.shape} S{S.shape} cand_h/m/chosen{cand_h.shape} "
          f"offsets{offsets.shape} 결정메타(region_id/ep/step/cls/mode_chosen/loggap)={n_dec}", flush=True)
    print(f"  이송모드 분포: AMB={int((mode_chosen==0).sum())} UAV={int((mode_chosen==1).sum())} | "
          f"등급: R={int((cls==0).sum())} Y={int((cls==1).sum())}", flush=True)

    if A.sanity:
        ok, tot = _sanity(offsets, cand_m, S, chosen, mode_chosen)
        pct = 100.0 * ok / max(tot, 1)
        print(f"\n[sanity] 선택=복원S(모드내) argmax: {ok}/{tot} = {pct:.2f}% "
              f"{'✅ (≥99%)' if pct >= 99.0 else '❌ (<99% — S 복원 점검 필요)'}", flush=True)


if __name__ == "__main__":
    main()
