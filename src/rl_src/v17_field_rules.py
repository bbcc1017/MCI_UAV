# -*- coding: utf-8 -*-
"""v17 현장 규칙집 — 물리 단위(km·분·명)로 교사 결정에서 규칙을 통계적으로 도출한다.

Yan et al. (2026) 은 환자콜을 몇 개의 고정 위치(우편번호 중심)로 동일시하고, 논문
6절에서 "distilled tree 로 해석 가능한 실시간 휴리스틱을 설계하는 것"을 후속연구로
남겼다. 우리는 250 시군구 × 무작위 4좌표 = 1,000 좌표에서 학습한 정책을 갖고 있으므로,
그 결정로그에서 **좌표에 의존하지 않는 물리 임계값**을 뽑을 수 있다.

정규화 ETA·순위 대신 다음을 쓴다.

* 도로거리 `d_HtoS_road` (km)  — AMB 이송
* 직선거리 `d_HtoS_euc`  (km)  — UAV 이송
* 이송시간 `amb_HtoS_t[0]`, `uav_HtoS_t[0]` (분)
* 누적 발송 `p_sent` (명), 점유 `occ` (명)

subcommands
  static : 좌표별 병원 정적표(거리·시간·tier·헬기장·용량)를 npz 로 덤프
  table  : 교사 결정 npz + 정적표 → 결정·후보 단위 물리량 테이블
  mine   : 임계값 추정(로지스틱 · 결정스텀프 · 지역별 안정성)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
DECISIONS = REPO / "results/scoreboard/v10/distill/data/ppo_train1000_seed5000.npz"
H_PAD = 47


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ static
def _static_worker(job):
    keys, entries = job
    try:
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD=str(H_PAD),
            MCI_REWARD_MODE="woG",
        )
        from viper_distill import _suppress_stdout, make_feature_env

        out = []
        with _suppress_stdout():
            for key, cfg in zip(keys, entries):
                env = make_feature_env(cfg, None)(seed=0)
                u = env.unwrapped
                props = u.en_manager.en_properties
                hp = props["hospital"]
                H = int(hp["hos_num"])
                d_road = np.asarray(hp.get("d_HtoS_road", hp.get("d_HtoS_euc")), float).reshape(-1)
                d_euc = np.asarray(hp.get("d_HtoS_euc", d_road), float).reshape(-1)
                t_amb = np.asarray(props["ambulance"]["amb_HtoS_t"][0], float).reshape(-1)
                t_uav = np.asarray(props["uav"]["uav_HtoS_t"][0], float).reshape(-1)
                tier = np.asarray(hp["hos_tier"], float).reshape(-1)
                heli = np.zeros(H)
                idx = np.asarray(hp.get("hos_helipad_idx", []), int).reshape(-1)
                heli[idx[(idx >= 0) & (idx < H)]] = 1.0
                out.append({
                    "key": key, "H": H,
                    "d_road": d_road[:H], "d_euc": d_euc[:H],
                    "t_amb": t_amb[:H], "t_uav": t_uav[:H],
                    "tier": tier[:H], "heli": heli[:H],
                    "max_send": np.asarray(hp["hos_max_send"], float).reshape(-1)[:H],
                })
        return {"ok": True, "rows": out}
    except Exception as exc:
        import traceback

        return {"ok": False, "err": (str(exc) + traceback.format_exc())[:1500]}


def static_main(args) -> None:
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    keys = sorted(manifest)
    if args.limit:
        keys = keys[: args.limit]
    chunks = [keys[i:i + args.chunk] for i in range(0, len(keys), args.chunk)]
    jobs = [(c, [manifest[k] for k in c]) for c in chunks]
    print(f"[static] coords={len(keys)} chunks={len(chunks)} workers={min(args.workers,len(jobs))}",
          flush=True)
    rows, t0 = [], time.time()
    with Pool(min(args.workers, len(jobs)), maxtasksperchild=4) as pool:
        for i, res in enumerate(pool.imap_unordered(_static_worker, jobs), 1):
            if not res["ok"]:
                raise RuntimeError(res["err"])
            rows.extend(res["rows"])
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] coords={len(rows)} wall={time.time()-t0:.0f}s",
                      flush=True)
    rows.sort(key=lambda r: r["key"])
    order = [r["key"] for r in rows]
    Hs = np.asarray([r["H"] for r in rows], int)
    n, Hmax = len(rows), int(Hs.max())

    def pad(name):
        a = np.full((n, Hmax), np.nan)
        for i, r in enumerate(rows):
            a[i, : r["H"]] = r[name]
        return a

    np.savez_compressed(
        args.out,
        keys=np.asarray(order), H=Hs,
        d_road=pad("d_road"), d_euc=pad("d_euc"),
        t_amb=pad("t_amb"), t_uav=pad("t_uav"),
        tier=pad("tier"), heli=pad("heli"), max_send=pad("max_send"),
    )
    print(f"[static] 저장 {args.out}  coords={n} Hmax={Hmax} "
          f"H범위={Hs.min()}~{Hs.max()} wall={(time.time()-t0)/60:.1f}분", flush=True)



# ------------------------------------------------------------------- table
_COL = {
    "max_send": 10, "red_at_site": 12, "yellow_at_site": 13,
    "amb_available": 14, "uav_available": 15, "time_min": 16,
    "red_unrescued": 17, "yellow_unrescued": 18,
    "cand_p_sent": 21, "cand_in_flight": 23, "total_p_sent": 24,
    "cand_cap_remain": 34, "cand_occ": 35, "cand_occ_ratio": 36, "rho": 37,
}


def decode(a: int, mask_len: int = 192, h_pad: int = H_PAD):
    n_dest = h_pad + 1
    return a // (n_dest * 2), (a % (n_dest * 2)) // 2, a % 2


def table_main(args) -> None:
    """결정 1건 = 1행. 두 등급 × 두 수단의 적격집합·최근접거리를 모두 기록해
    '수단 선택의 자유가 있었는가', '등급 선택의 자유가 있었는가'를 분리할 수 있게 한다."""
    st = np.load(args.static, allow_pickle=False)
    skeys = {str(k): i for i, k in enumerate(st["keys"])}
    z = np.load(args.decisions, allow_pickle=False)
    X, off, cand, teach = z["X"], z["offsets"], z["cand_action"], z["teacher_action"]
    keys = np.asarray([str(x) for x in z["state_key"]])
    n = len(off) - 1
    miss = sorted(set(keys) - set(skeys))
    if miss:
        raise ValueError(f"정적표에 없는 좌표 {len(miss)}개: {miss[:3]}")
    dec = {int(a): decode(int(a)) for a in np.unique(cand)}
    CAP = float(args.cap)
    V_AMB, V_UAV = 50.0, 200.0          # km/h — 시나리오 상수(정적표로 검증됨)

    base = ["key", "sigcd", "point", "state", "cls", "mode", "hosp"]
    chosen = ["d_road_km", "d_euc_km", "t_amb_min", "t_uav_min", "tier3", "heli",
              "max_send", "p_sent", "in_flight", "occ", "occ_ratio",
              "d_rank_same_mode", "extra_km_vs_near", "extra_km_vs_near_free"]
    grid, near = [], []
    for cn in ("R", "Y"):
        for mn in ("amb", "uav"):
            grid.append(f"n_elig_{cn}_{mn}")
            near += [f"near_{cn}_{mn}_km", f"near_{cn}_{mn}_free_km"]
    ctxf = ["free_mode", "free_class", "switch_gain_min",
            "red_at_site", "yellow_at_site", "amb_available", "uav_available",
            "time_min", "total_p_sent", "rho",
            "site_near_hosp_km", "site_near_tier3_km", "site_near_heli_km",
            "n_hosp_within_20km", "n_tier3_within_20km"]
    fields = base + chosen + grid + near + ctxf
    rec = {f: [] for f in fields}
    skipped_stay = 0

    for s_ in range(n):
        a, b = int(off[s_]), int(off[s_ + 1])
        acts, Xs = cand[a:b], X[a:b]
        key = str(keys[s_])
        si = skeys[key]
        Hn = int(st["H"][si])
        d_road, d_euc = st["d_road"][si, :Hn], st["d_euc"][si, :Hn]
        tier, heli = st["tier"][si, :Hn], st["heli"][si, :Hn]
        tc, td, tm = dec[int(teach[s_])]
        if td == 0:
            skipped_stay += 1
            continue
        h = td - 1
        rows_of = {}
        for j, act in enumerate(acts):
            c, dd, m = dec[int(act)]
            if dd > 0:
                rows_of[(c, dd - 1, m)] = j
        elig = {}
        for c in (0, 1):
            for m in (0, 1):
                elig[(c, m)] = np.asarray(
                    sorted(hh for (cc, hh, mm) in rows_of if cc == c and mm == m), int)
        p_sent_all = {}
        for (c, hh, m), j in rows_of.items():
            p_sent_all[hh] = float(Xs[j, _COL["cand_p_sent"]])

        def near_of(c, m, capped):
            e = elig[(c, m)]
            if capped:
                e = np.asarray([hh for hh in e if p_sent_all.get(hh, 0.0) < CAP], int)
            if e.size == 0:
                return np.nan
            dist = d_euc if m == 1 else d_road
            return float(dist[e].min())

        jrow = rows_of[(tc, h, tm)]
        dist_same = d_euc if tm == 1 else d_road
        e_same = elig[(tc, tm)]
        free_same = np.asarray(
            [hh for hh in e_same if p_sent_all.get(hh, 0.0) < CAP], int)
        na = near_of(tc, 0, False)
        nu = near_of(tc, 1, False)

        rec["key"].append(key)
        rec["sigcd"].append(key.rsplit("_", 2)[-2])
        rec["point"].append(key.rsplit("_", 1)[-1])
        rec["state"].append(s_)
        rec["cls"].append(tc)
        rec["mode"].append(tm)
        rec["hosp"].append(h)
        rec["d_road_km"].append(float(d_road[h]))
        rec["d_euc_km"].append(float(d_euc[h]))
        rec["t_amb_min"].append(float(st["t_amb"][si, h]))
        rec["t_uav_min"].append(float(st["t_uav"][si, h]))
        rec["tier3"].append(float(tier[h] == 3))
        rec["heli"].append(float(heli[h]))
        for f, col in (("max_send", "max_send"), ("p_sent", "cand_p_sent"),
                       ("in_flight", "cand_in_flight"), ("occ", "cand_occ"),
                       ("occ_ratio", "cand_occ_ratio")):
            rec[f].append(float(Xs[jrow, _COL[col]]))
        rec["d_rank_same_mode"].append(float((dist_same[e_same] < dist_same[h]).sum()))
        rec["extra_km_vs_near"].append(float(dist_same[h] - dist_same[e_same].min()))
        rec["extra_km_vs_near_free"].append(
            float(dist_same[h] - dist_same[free_same].min()) if free_same.size else np.nan)
        for cn, c in (("R", 0), ("Y", 1)):
            for mn, m in (("amb", 0), ("uav", 1)):
                rec[f"n_elig_{cn}_{mn}"].append(float(elig[(c, m)].size))
                rec[f"near_{cn}_{mn}_km"].append(near_of(c, m, False))
                rec[f"near_{cn}_{mn}_free_km"].append(near_of(c, m, True))
        rec["free_mode"].append(float(elig[(tc, 0)].size > 0 and elig[(tc, 1)].size > 0))
        rec["free_class"].append(float(
            (elig[(0, 0)].size + elig[(0, 1)].size) > 0
            and (elig[(1, 0)].size + elig[(1, 1)].size) > 0))
        rec["switch_gain_min"].append(
            (na * 60.0 / V_AMB - nu * 60.0 / V_UAV) if (na == na and nu == nu) else np.nan)
        for f in ("red_at_site", "yellow_at_site", "amb_available", "uav_available",
                  "time_min", "total_p_sent", "rho"):
            rec[f].append(float(Xs[jrow, _COL[f]]))
        t3 = np.flatnonzero(tier == 3)
        hl = np.flatnonzero(heli > 0.5)
        rec["site_near_hosp_km"].append(float(d_road.min()))
        rec["site_near_tier3_km"].append(float(d_road[t3].min()) if t3.size else np.nan)
        rec["site_near_heli_km"].append(float(d_euc[hl].min()) if hl.size else np.nan)
        rec["n_hosp_within_20km"].append(float((d_road <= 20.0).sum()))
        rec["n_tier3_within_20km"].append(float((d_road[t3] <= 20.0).sum()) if t3.size else 0.0)

    import csv

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(rec["key"])):
            w.writerow([rec[f][i] for f in fields])
    m = np.asarray(rec["mode"])
    c = np.asarray(rec["cls"])
    print(f"[table] 결정 {len(m)}/{n} 행 (현장대기 제외 {skipped_stay}) → {out}", flush=True)
    print(f"  Red {int((c == 0).sum())} / Yellow {int((c == 1).sum())} · "
          f"UAV {int(m.sum())} / AMB {int(len(m) - m.sum())} · "
          f"수단자유 {float(np.mean(rec['free_mode']))*100:.1f}% · "
          f"등급자유 {float(np.mean(rec['free_class']))*100:.1f}%", flush=True)



# ------------------------------------------------------- 조건부 로짓 (목적지)
LOGIT_FEATURES = ["d_km", "p_sent", "in_flight", "occ_ratio", "is_tier3"]


def _logit_design(args):
    """교사의 (등급, 수단) 을 고정한 뒤 목적지 선택만 남긴 후보 단위 설계행렬.

    이렇게 조건화하면 tier3·헬기장·용량 마스크가 이미 반영된 선택집합 안에서의
    '어느 병원' 결정만 남으므로, 계수비가 곧 물리 단위 교환율이 된다.
    """
    st = np.load(args.static, allow_pickle=False)
    skeys = {str(k): i for i, k in enumerate(st["keys"])}
    z = np.load(args.decisions, allow_pickle=False)
    X, off, cand, teach = z["X"], z["offsets"], z["cand_action"], z["teacher_action"]
    keys = np.asarray([str(x) for x in z["state_key"]])
    dec = {int(a): decode(int(a)) for a in np.unique(cand)}
    rows, ys, grp, meta = [], [], [], []
    for s_ in range(len(off) - 1):
        a, b = int(off[s_]), int(off[s_ + 1])
        acts, Xs = cand[a:b], X[a:b]
        tc, td, tm = dec[int(teach[s_])]
        if td == 0:
            continue
        si = skeys[str(keys[s_])]
        Hn = int(st["H"][si])
        dist = st["d_euc"][si, :Hn] if tm == 1 else st["d_road"][si, :Hn]
        tier = st["tier"][si, :Hn]
        sel = [(j, dd - 1) for j, act in enumerate(acts)
               for (c, dd, m) in [dec[int(act)]] if c == tc and m == tm and dd > 0]
        if len(sel) < 2:
            continue
        blk = np.empty((len(sel), len(LOGIT_FEATURES)))
        yy = np.zeros(len(sel), dtype=np.int8)
        for i, (j, hh) in enumerate(sel):
            blk[i] = (dist[hh], Xs[j, _COL["cand_p_sent"]], Xs[j, _COL["cand_in_flight"]],
                      Xs[j, _COL["cand_occ_ratio"]], float(tier[hh] == 3))
            if hh == td - 1:
                yy[i] = 1
        if yy.sum() != 1:
            continue
        rows.append(blk)
        ys.append(yy)
        grp.append(len(sel))
        meta.append((str(keys[s_]).rsplit("_", 2)[-2], tc, tm))
    Xd = np.vstack(rows)
    yd = np.concatenate(ys)
    gd = np.asarray(grp, int)
    return Xd, yd, gd, meta


def _cond_logit(Xd, yd, gd, l2: float = 1e-6):
    """조건부 로짓 MLE. 그룹별 softmax 우도. 반환 = 계수(피처 순서)."""
    from scipy.optimize import minimize

    offs = np.concatenate([[0], np.cumsum(gd)])
    gid = np.repeat(np.arange(len(gd)), gd)
    chosen_rows = np.flatnonzero(yd == 1)
    mu = Xd.mean(0)
    sd = Xd.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xd - mu) / sd

    def nll(beta):
        v = Z @ beta
        mx = np.maximum.reduceat(v, offs[:-1])
        e = np.exp(v - mx[gid])
        den = np.add.reduceat(e, offs[:-1])
        ll = (v[chosen_rows] - mx - np.log(den)).sum()
        return -ll / len(gd) + l2 * float(beta @ beta)

    res = minimize(nll, np.zeros(Z.shape[1]), method="L-BFGS-B",
                   options={"maxiter": 500})
    beta_std = res.x
    beta = beta_std / sd                      # 원 단위 계수로 환산
    return beta, float(res.fun), res


def logit_main(args) -> None:
    t0 = time.time()
    Xd, yd, gd, meta = _logit_design(args)
    print(f"[logit] 선택집합 {len(gd)}개 · 후보행 {len(Xd)} · 평균 후보수 {len(Xd)/len(gd):.1f} "
          f"(설계 {time.time()-t0:.0f}s)", flush=True)
    sig = np.asarray([m[0] for m in meta])
    cls = np.asarray([m[1] for m in meta])
    mode = np.asarray([m[2] for m in meta])
    offs = np.concatenate([[0], np.cumsum(gd)])

    def subset(mask):
        rmask = np.repeat(mask, gd)
        return Xd[rmask], yd[rmask], gd[mask]

    def report(label, Xs, ys, gs, boot_sig=None):
        beta, fun, _ = _cond_logit(Xs, ys, gs)
        bd = beta[LOGIT_FEATURES.index("d_km")]
        km = {f: (beta[i] / bd if abs(bd) > 1e-12 else np.nan)
              for i, f in enumerate(LOGIT_FEATURES)}
        # 적합도: 그룹별 top-1 정확도
        o = np.concatenate([[0], np.cumsum(gs)])
        v = Xs @ beta
        hit = 0
        for k in range(len(gs)):
            a, b = int(o[k]), int(o[k + 1])
            hit += int(ys[a:b][int(np.argmax(v[a:b]))] == 1)
        print(f"\n== {label} (선택집합 {len(gs)}) ==")
        print(f"  top-1 재현율 {hit/len(gs):.4f}  (우연 {np.mean(1/gs):.4f})")
        print(f"  거리 계수 {bd:+.4f} /km → 부호 {'가까울수록 선호' if bd<0 else '이상(먼 곳 선호)'}")
        for f in LOGIT_FEATURES[1:]:
            print(f"  {f:11s} = {km[f]:+7.2f} km 상당 (계수 {beta[LOGIT_FEATURES.index(f)]:+.4f})")
        return beta, km

    out = {}
    out["all"] = report("전체", Xd, yd, gd)
    for c, cn in ((0, "Red"), (1, "Yellow")):
        for m, mn in ((0, "AMB"), (1, "UAV")):
            mask = (cls == c) & (mode == m)
            if mask.sum() >= 500:
                out[f"{cn}_{mn}"] = report(f"{cn} · {mn}", *subset(mask))

    if args.bootstrap:
        rng = np.random.default_rng(0)
        codes = np.unique(sig)
        idx_of = {k: np.flatnonzero(sig == k) for k in codes}
        vals = []
        for _ in range(args.bootstrap):
            pick = rng.choice(codes, size=len(codes), replace=True)
            sel = np.concatenate([idx_of[k] for k in pick])
            m = np.zeros(len(gd), bool)
            # 클러스터 부트스트랩: 중복 선택은 한 번만 반영(보수적)
            m[np.unique(sel)] = True
            b, _ = report_silent(Xd, yd, gd, m)
            vals.append(b)
        v = np.asarray(vals)
        lo, hi = np.percentile(v, [2.5, 97.5], axis=0)
        print("\n== 시군구 클러스터 부트스트랩 95% CI (km 상당) ==")
        for i, f in enumerate(LOGIT_FEATURES[1:], start=1):
            print(f"  {f:11s} [{lo[i]:+.2f}, {hi[i]:+.2f}]")

    js = {k: {"beta": list(map(float, v[0])), "km_equiv": {a: float(b) for a, b in v[1].items()}}
          for k, v in out.items()}
    Path(args.out).write_text(json.dumps(
        {"features": LOGIT_FEATURES, "fits": js,
         "n_choice_sets": int(len(gd)), "decisions": args.decisions},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[logit] 저장 {args.out}  wall={(time.time()-t0)/60:.1f}분", flush=True)


def report_silent(Xd, yd, gd, mask):
    rmask = np.repeat(mask, gd)
    beta, _, _ = _cond_logit(Xd[rmask], yd[rmask], gd[mask])
    bd = beta[LOGIT_FEATURES.index("d_km")]
    return (beta / bd if abs(bd) > 1e-12 else beta * np.nan), None



# ------------------------------------------------------------- 현장 규칙집 정책
FIELD_CARD_DEFAULT = {"lam_km_per_patient": 6.0, "red_uav_km": 12.0, "yellow_hold": 14.0}


def make_field_card_policy(lam_km_per_patient: float = 6.0,
                           red_uav_km: float = 12.0,
                           yellow_hold: float = 14.0,
                           h_pad: int = H_PAD,
                           dist_mode: str = "raw"):
    """교사 결정 1,000좌표 37,000건에서 통계적으로 도출한 3단 현장 규칙집.

    1단 등급 : Yellow 현장대기가 `yellow_hold` 명 이하이고 UAV 가 현장에 있으면 Red,
               그 밖에는 Yellow. (등급자유 구간 재현율 0.842)
    2단 목적지: 적격 병원 중 `거리(km) + lam × 그 병원 현재 부하(명)` 최소.
               부하 = 입원 census + 이송 중. 조건부 로짓 추정 lam = 6.4 km/명.
    3단 수단 : 현장에 한 종류만 대기하면 그것(결정의 85.3%). 둘 다 대기하면
               Red 는 최근접 tier3 도로거리가 `red_uav_km` 를 넘을 때 UAV, 그 밖에는 AMB.
               Yellow 는 AMB. (Red 자유선택 균형정확도 0.828)

    거리는 AMB=도로거리, UAV=직선거리를 쓴다(시나리오 원 단위).

    ``dist_mode="normclip"`` 은 PPO 관측과 같은 정보 제약을 건다 — 거리를 최근접=1 로
    정규화한 뒤 10배에서 클립한다(``get_static_eta`` 와 동일). 규칙 가족을 그대로 두고
    정보만 PPO 수준으로 낮춰, 규칙이 이기는 원인이 정보인지 구조인지 분리하는 진단용이다.
    """
    if dist_mode not in ("raw", "norm", "normclip"):
        raise ValueError(dist_mode)
    from aggregate_obs import AggregateObsWrapper
    from loadbalance_heuristic import _codec_from_mask
    from score_features import build_ctx, compute_static

    cache = {"mid": None}

    def _static(env):
        mid = id(env.en_manager)
        if cache["mid"] != mid:
            base = compute_static(env)
            hp = env.en_manager.en_properties["hospital"]
            H = int(base["H"])
            heli = np.zeros(H)
            idx = np.asarray(hp.get("hos_helipad_idx", []), int).reshape(-1)
            heli[idx[(idx >= 0) & (idx < H)]] = 1.0
            cache.update(
                mid=mid, base=base, H=H, heli=heli,
                d_road=np.asarray(hp.get("d_HtoS_road", hp.get("d_HtoS_euc")), float).reshape(-1)[:H],
                d_euc=np.asarray(hp.get("d_HtoS_euc"), float).reshape(-1)[:H],
                tier3=np.asarray(base["is_tier3"], float),
            )
        return cache

    def fn(obs, mask, env_unwrapped):
        u = env_unwrapped
        mask = np.asarray(mask, dtype=bool)
        st = _static(u)
        H = st["H"]
        encode = _codec_from_mask(len(mask), h_pad)
        dobs = u.en_manager.get_full_obs()
        dobs["time"] = u.ev_manager.time
        ctx = build_ctx(u, static=st["base"], dobs=dobs)
        load = np.asarray(ctx["occ"], float) + np.asarray(ctx["in_flight"], float)
        pa = AggregateObsWrapper._patient_agg(np.asarray(dobs["p_states"]))[:10]
        red_wait, yellow_wait = float(pa[1]), float(pa[6])

        def elig(c, m):
            idx = [h for h in range(H) if mask[encode(c, h + 1, m)]]
            return np.asarray(idx, int)

        sets = {(c, m): elig(c, m) for c in (0, 1) for m in (0, 1)}

        # --- 1단: 등급 ---
        can = {c: (sets[(c, 0)].size + sets[(c, 1)].size) > 0 for c in (0, 1)}
        uav_here = sets[(0, 1)].size > 0 or sets[(1, 1)].size > 0
        want_red = (yellow_wait <= yellow_hold) and uav_here
        c = 0 if (want_red and can[0]) else (1 if can[1] else (0 if can[0] else None))
        if c is None:                                  # 이송 불가 → 현장대기
            return int(encode(0, 0, 0))

        # --- 3단: 수단 (등급 확정 후) ---
        has_a, has_u = sets[(c, 0)].size > 0, sets[(c, 1)].size > 0
        if has_a and has_u:
            if c == 0:
                t3 = sets[(c, 0)][st["tier3"][sets[(c, 0)]] > 0.5]
                near_t3 = float(st["d_road"][t3].min()) if t3.size else float(
                    st["d_road"][sets[(c, 0)]].min())
                m = 1 if near_t3 > red_uav_km else 0
            else:
                m = 0
        else:
            m = 0 if has_a else 1

        # --- 2단: 목적지 ---
        cand = sets[(c, m)]
        dist = st["d_euc"] if m == 1 else st["d_road"]
        if dist_mode != "raw":
            pos = dist[dist > 0]
            dist = dist / (float(pos.min()) if pos.size else 1.0)
            if dist_mode == "normclip":
                dist = np.minimum(dist, 10.0)
        score = dist[cand] + lam_km_per_patient * load[cand]
        best = cand[int(np.argmin(score))]
        return int(encode(c, int(best) + 1, m))

    fn.policy_name = (f"FIELD_CARD[{dist_mode}] lam={lam_km_per_patient:g} "
                      f"red_km={red_uav_km:g} yhold={yellow_hold:g}")
    return fn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--manifest", default=str(TRAIN_MANIFEST))
    s.add_argument("--workers", type=int, default=48)
    s.add_argument("--chunk", type=int, default=10)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--static", default=str(REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"))
    t.add_argument("--decisions", default=str(DECISIONS))
    t.add_argument("--cap", type=float, default=3.0, help="발송상한 T (near_amb_free 정의)")
    t.add_argument("--out", required=True)
    g = sub.add_parser("logit")
    g.add_argument("--static", default=str(REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"))
    g.add_argument("--decisions", default=str(DECISIONS))
    g.add_argument("--bootstrap", type=int, default=0)
    g.add_argument("--out", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    {"static": static_main, "table": table_main, "logit": logit_main}[args.cmd](args)


if __name__ == "__main__":
    main()
