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

# ★ CARD 함수형과 정합한 2특징 집합 (v18 E1).
#   CARD 는 `거리 + λ × (occ + in_flight)` 인데 위 5특징 적합은 부하를 p_sent/in_flight/
#   occ_ratio 로 쪼개 넣어 함수형이 어긋난다. λ 를 산출할 때는 반드시 이쪽을 쓴다.
LOGIT_FEATURES_CARD = ["d_km", "load"]

_FEAT_FN = {
    "d_km":       lambda dist, hh, xs, tier: dist[hh],
    "p_sent":     lambda dist, hh, xs, tier: xs[_COL["cand_p_sent"]],
    "in_flight":  lambda dist, hh, xs, tier: xs[_COL["cand_in_flight"]],
    "occ":        lambda dist, hh, xs, tier: xs[_COL["cand_occ"]],
    "occ_ratio":  lambda dist, hh, xs, tier: xs[_COL["cand_occ_ratio"]],
    "cap_remain": lambda dist, hh, xs, tier: xs[_COL["cand_cap_remain"]],
    "load":       lambda dist, hh, xs, tier: xs[_COL["cand_occ"]] + xs[_COL["cand_in_flight"]],
    "is_tier3":   lambda dist, hh, xs, tier: float(tier[hh] == 3),
}


def _logit_design(args, features=None):
    """교사의 (등급, 수단) 을 고정한 뒤 목적지 선택만 남긴 후보 단위 설계행렬.

    이렇게 조건화하면 tier3·헬기장·용량 마스크가 이미 반영된 선택집합 안에서의
    '어느 병원' 결정만 남으므로, 계수비가 곧 물리 단위 교환율이 된다.
    """
    feats = list(features or LOGIT_FEATURES)
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
        blk = np.empty((len(sel), len(feats)))
        yy = np.zeros(len(sel), dtype=np.int8)
        for i, (j, hh) in enumerate(sel):
            blk[i] = [_FEAT_FN[f](dist, hh, Xs[j], tier) for f in feats]
            if hh == td - 1:
                yy[i] = 1
        if yy.sum() != 1:
            continue
        rows.append(blk)
        ys.append(yy)
        grp.append(len(sel))
        meta.append((str(keys[s_]).rsplit("_", 2)[-2], tc, tm,
                     str(keys[s_]).rsplit("_", 1)[-1]))
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
# 두 파라미터 집합을 명시적으로 분리한다. 혼동이 v17 발표자료의 카드 오기를 낳았다.
#   BEHAVIOUR = 교사 행동을 가장 잘 재현하는 추정치 (조건부 로짓 / BA 컷 / 등급 트리)
#   ADOPTED   = dev40 폐루프 PDR argmin. 실제로 평가·배포된 구성.
# 둘은 다르다 — 특히 yellow_hold 는 14(행동) vs 0(성능)로 갈린다.
FIELD_CARD_BEHAVIOUR = {"lam_km_per_patient": 6.4, "red_uav_km": 11.75, "yellow_hold": 14.0}
FIELD_CARD_ADOPTED = {"lam_km_per_patient": 12.0, "red_uav_km": 12.0, "yellow_hold": 0.0}


# 부하항 후보 (v18 E3 변수선택 어블레이션).
#   "load" 가 CARD 채택값 = 입원 census + 이송 중. 나머지는 "RL 이 고른 변수가 맞는가"를
#   묻는 대조군이다 — 특히 p_sent 는 LB-T3 계열이 쓰는 축이고, in_flight 단독은 병원
#   실시간 연계 없이 현장에서 셀 수 있는(I1) 신호다.
LOAD_TERMS = ("load", "p_sent", "in_flight", "occ", "occ_ratio", "cap_deficit", "zero")


def _load_vector(ctx, term: str, base):
    o = np.asarray(ctx["occ"], float)
    f = np.asarray(ctx["in_flight"], float)
    if term == "load":
        return o + f
    if term == "occ":
        return o
    if term == "in_flight":
        return f
    if term == "p_sent":
        return np.asarray(ctx["p_sent"], float)
    if term == "occ_ratio":                # build_ctx 는 비율을 주지 않는다 → 여기서 계산
        ms = np.maximum(np.asarray(base["max_send"], float), 1.0)
        return (o + f) / ms
    if term == "cap_deficit":          # 남은 여력이 적을수록 벌점 (여력의 음수)
        return -np.asarray(ctx["cap_remain"], float)
    if term == "zero":
        return np.zeros_like(o)
    raise ValueError(f"미지 load_term: {term}")


def make_field_card_policy(lam_km_per_patient: float = 6.0,
                           red_uav_km: float = 12.0,
                           yellow_hold: float = 14.0,
                           h_pad: int = H_PAD,
                           dist_mode: str = "raw",
                           load_term: str = "load"):
    """교사 결정 1,000좌표 37,000건에서 통계적으로 도출한 3단 현장 규칙집.

    ⚠️ **이 함수의 기본 인자는 채택값이 아니라 행동추정치에 가깝다.** 폐루프에서 평가·보고된
    구성은 ``FIELD_CARD_ADOPTED`` = (lam 12.0, red_uav_km 12.0, **yellow_hold 0.0**) 이다.
    특히 ``yellow_hold`` 는 행동추정 14 와 dev40 성능최적 0 이 갈린다(등급 규칙은 폐루프
    성능에 기여하지 않는다 — 항상 Yellow 우선이 최적). 기본값에 의존하지 말고 명시할 것.

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
    if load_term not in LOAD_TERMS:
        raise ValueError(f"load_term 은 {LOAD_TERMS} 중 하나 (got {load_term})")
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
        load = _load_vector(ctx, load_term, st["base"])
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

    fn.policy_name = (f"FIELD_CARD[{dist_mode}/{load_term}] lam={lam_km_per_patient:g} "
                      f"red_km={red_uav_km:g} yhold={yellow_hold:g}")
    return fn


def sigcd_of(region_key: str) -> str:
    """매니페스트 지역키 → 시군구 법정코드 5자리.

    대표점250 은 ``종로구_11110``, train1000 은 ``종로구_11110_p0`` 라 접미가 다르다.
    두 형태 모두에서 같은 코드를 뽑아야 지역별 파라미터표가 학습·평가 양쪽에 붙는다.
    """
    parts = str(region_key).split("_")
    for tok in reversed(parts):
        if tok.isdigit() and len(tok) == 5:
            return tok
    raise ValueError(f"시군구 코드를 못 찾음: {region_key}")


def make_field_card_policy_local(params: dict, region_key: str, h_pad: int = H_PAD,
                                 dist_mode: str = "raw", load_term: str = "load"):
    """지역별 파라미터를 쓰는 CARD (v18 E4).

    ``params`` 는 ``{"_default": {...}, "<sigcd>": {...}}`` 형태이고 각 값은
    ``{"lam":..., "red_km":..., "yhold":...}`` 이다. 지역이 표에 없으면 ``_default`` 로
    떨어진다 — leave-province-out 이나 미보유 지역에서 조용히 실패하지 않게 하기 위함이다.

    ⚠️ 전국 단일 파라미터를 쓰는 기존 경로(`make_field_card_policy`)는 그대로 두었다.
    이 함수는 지역 인자를 받는 별도 진입점이라 구 경로 동작에 영향이 없다.
    """
    d = dict(params.get("_default") or {})
    try:
        d.update(params.get(sigcd_of(region_key)) or {})
    except ValueError:
        pass
    if not d:
        raise ValueError(f"{region_key}: 파라미터도 _default 도 없다")
    fn = make_field_card_policy(float(d["lam"]), float(d["red_km"]), float(d["yhold"]),
                                h_pad=h_pad, dist_mode=dist_mode, load_term=load_term)
    fn.policy_name = (f"FIELD_CARD_LOCAL[{dist_mode}/{load_term}] {region_key} "
                      f"lam={d['lam']:g} red_km={d['red_km']:g} yhold={d['yhold']:g}")
    return fn


# =========================================================== mine (v18 E1)
# v17 의 임계값 3개는 서로 다른 절차로 나왔고 그중 둘은 생성 코드가 커밋되지 않았다.
# 방법론 절로 승격하려면 절차가 재현 가능해야 하므로 흩어진 조각을 여기로 모은다.
#   lambda : (거리, 부하) 2특징 조건부 로짓  — CARD 함수형과 정합 (5특징 적합은 어긋난다)
#   redkm  : 자유선택 Red 부분집합의 균형정확도 컷 스윕
#   yhold  : 자유선택 등급 부분집합의 균형정확도 컷 스윕
# 부트스트랩은 **정식** 클러스터 부트스트랩이다. v17 구현은 복원추출 후 np.unique 로
# 중복을 지워 사실상 63% 서브샘플 반복이었다(CI 폭 왜곡).

def cluster_boot_index(groups, rng, idx_of=None):
    """클러스터 복원추출 → 행 인덱스. 중복 클러스터는 **중복 그대로** 포함한다."""
    codes = np.unique(groups)
    if idx_of is None:
        idx_of = {k: np.flatnonzero(groups == k) for k in codes}
    pick = rng.choice(codes, size=len(codes), replace=True)
    return np.concatenate([idx_of[k] for k in pick]), idx_of


def ba_cut(x, y, ge_is_positive=True):
    """균형정확도 최대 임계. 반환 (cut, ba, n_pos, n_neg).

    ge_is_positive=True  : x >= cut 을 양성(1)으로 예측  — Red 의 UAV 전환(먼 곳일수록 UAV)
    ge_is_positive=False : x <= cut 을 양성(1)으로 예측  — 등급(Yellow 가 적을수록 Red)
    """
    x = np.asarray(x, float); y = np.asarray(y, int)
    P, N = int(y.sum()), int(len(y) - y.sum())
    if P == 0 or N == 0:
        return float("nan"), float("nan"), P, N
    best = (float("nan"), -1.0)
    for c in np.unique(x):
        pr = (x >= c) if ge_is_positive else (x <= c)
        ba = 0.5 * ((pr & (y == 1)).sum() / P + ((~pr) & (y == 0)).sum() / N)
        if ba > best[1]:
            best = (float(c), float(ba))
    return best[0], best[1], P, N


def _read_table(path):
    import pandas as pd
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["sigcd"] = d["sigcd"].astype(str)
    return d


def _lambda_fit(Xd, yd, gd, feats):
    """계수비 = 부하 1명이 몇 km 에 해당하는가."""
    beta, _, _ = _cond_logit(Xd, yd, gd)
    bd = beta[feats.index("d_km")]
    if abs(bd) < 1e-12:
        return float("nan"), beta
    return float(beta[feats.index("load")] / bd), beta


def _topk_acc(Xd, yd, gd, beta):
    o = np.concatenate([[0], np.cumsum(gd)])
    v = Xd @ beta
    hit = sum(int(yd[int(o[k]):int(o[k + 1])][int(np.argmax(v[int(o[k]):int(o[k + 1])]))] == 1)
              for k in range(len(gd)))
    return hit / len(gd), float(np.mean(1.0 / gd))


def mine_main(args) -> None:
    t0 = time.time()
    res = {"date": time.strftime("%Y-%m-%d"), "what": args.what,
           "decisions": getattr(args, "decisions", None), "table": getattr(args, "table", None),
           "min_n": args.min_n, "bootstrap": args.bootstrap}
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- lambda
    if args.what in ("lambda", "stability"):
        feats = LOGIT_FEATURES_CARD
        Xd, yd, gd, meta = _logit_design(args, features=feats)
        sig = np.asarray([m[0] for m in meta])
        cls = np.asarray([m[1] for m in meta])
        pnt = np.asarray([m[3] for m in meta])
        print(f"[mine.lambda] 선택집합 {len(gd)} · 후보행 {len(Xd)} · 특징 {feats} "
              f"({time.time()-t0:.0f}s)", flush=True)

        def fit(mask):
            rm = np.repeat(mask, gd)
            return _lambda_fit(Xd[rm], yd[rm], gd[mask], feats)

        lam_all, beta_all = fit(np.ones(len(gd), bool))
        acc, chance = _topk_acc(Xd, yd, gd, beta_all)
        lam = {"all": lam_all, "beta": [float(b) for b in beta_all],
               "top1_acc": acc, "chance": chance, "n_choice_sets": int(len(gd))}
        for lab, mask in (("p0p1", np.isin(pnt, ["p0", "p1"])),
                          ("p2p3", np.isin(pnt, ["p2", "p3"])),
                          ("Red", cls == 0), ("Yellow", cls == 1)):
            if mask.sum() >= args.min_n:
                lam[lab] = fit(mask)[0]
        if args.bootstrap:
            idx_of, vals = None, []
            off = np.concatenate([[0], np.cumsum(gd)])
            for _ in range(args.bootstrap):
                sel, idx_of = cluster_boot_index(sig, rng, idx_of)
                # 정식 클러스터 부트: 중복 선택된 클러스터의 행을 **중복 그대로** 넣는다
                rm = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])
                try:
                    v, _ = _lambda_fit(Xd[rm], yd[rm], gd[sel], feats)
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                lam["cluster_boot95"] = [float(np.percentile(vals, 2.5)),
                                         float(np.percentile(vals, 97.5))]
                lam["cluster_boot_n"] = len(vals)
        # 지역별
        by = {}
        for code in np.unique(sig):
            m = sig == code
            if m.sum() >= args.min_n:
                v = fit(m)[0]
                if np.isfinite(v):
                    by[str(code)] = float(v)
        if by:
            vv = np.array(list(by.values()))
            lam["by_district"] = {"n": len(by), "median": float(np.median(vv)),
                                  "q25": float(np.percentile(vv, 25)),
                                  "q75": float(np.percentile(vv, 75)),
                                  "values": by if args.dump_groups else None}
        res["lambda"] = lam
        print(f"  λ 전체 {lam_all:.4f} km/명 · top-1 {acc:.4f}(우연 {chance:.4f})"
              + (f" · 부트 95% [{lam['cluster_boot95'][0]:.2f}, {lam['cluster_boot95'][1]:.2f}]"
                 if "cluster_boot95" in lam else "")
              + (f" · 시군구 {lam['by_district']['n']}곳 중위 {lam['by_district']['median']:.2f}"
                 if "by_district" in lam else ""), flush=True)

    # ------------------------------------------------------------ redkm/yhold
    if args.what in ("redkm", "yhold", "stability"):
        d = _read_table(args.table)

    if args.what in ("redkm", "stability"):
        f2 = d[(d.free_mode == 1) & (d.cls == 0)]
        x, y = np.asarray(f2.near_R_amb_km, float), np.asarray(f2["mode"], int)
        cut, ba, P, N = ba_cut(x, y, ge_is_positive=True)
        rk = {"n": int(len(f2)), "n_uav": P, "n_amb": N, "cut_km": cut, "ba": ba}
        # 좌표군 교차검증
        rk["split"] = []
        for fitp, tstp in (("p0p1", "p2p3"), ("p2p3", "p0p1")):
            grp = {"p0p1": ["p0", "p1"], "p2p3": ["p2", "p3"]}
            fm = f2.point.isin(grp[fitp])
            tm = f2.point.isin(grp[tstp])
            if fm.sum() < args.min_n or tm.sum() < args.min_n:
                continue
            c, b, _, _ = ba_cut(x[fm.to_numpy()], y[fm.to_numpy()], True)
            xt, yt = x[tm.to_numpy()], y[tm.to_numpy()]
            pr = xt >= c
            Pt, Nt = int(yt.sum()), int(len(yt) - yt.sum())
            bt = 0.5 * ((pr & (yt == 1)).sum() / max(Pt, 1) + ((~pr) & (yt == 0)).sum() / max(Nt, 1))
            rk["split"].append({"fit": fitp, "cut_km": c, "ba_fit": b,
                                "test": tstp, "n_test": int(tm.sum()), "ba_test": float(bt)})
        # 클러스터 부트
        if args.bootstrap:
            sg = np.asarray(f2.sigcd)
            idx_of, vals = None, []
            for _ in range(args.bootstrap):
                sel, idx_of = cluster_boot_index(sg, rng, idx_of)
                c, b, P2, N2 = ba_cut(x[sel], y[sel], True)
                if np.isfinite(c):
                    vals.append(c)
            if vals:
                rk["cluster_boot95"] = [float(np.percentile(vals, 2.5)),
                                        float(np.percentile(vals, 97.5))]
        # 지역별
        cuts = {}
        for sgc, gg in f2.groupby("sigcd"):
            if len(gg) >= args.min_n and gg["mode"].nunique() == 2:
                c, b, _, _ = ba_cut(gg.near_R_amb_km, gg["mode"], True)
                if np.isfinite(c):
                    cuts[str(sgc)] = float(c)
        if cuts:
            cv = np.array(list(cuts.values()))
            rk["by_district"] = {"n": len(cuts), "median": float(np.median(cv)),
                                 "q25": float(np.percentile(cv, 25)),
                                 "q75": float(np.percentile(cv, 75)),
                                 "q10": float(np.percentile(cv, 10)),
                                 "q90": float(np.percentile(cv, 90)),
                                 "values": cuts if args.dump_groups else None}
        res["red_km"] = rk
        print(f"  red_km 전역 {cut:.2f} km (BA {ba:.4f}, 자유선택 {len(f2)}건)"
              + (f" · 시군구 {rk['by_district']['n']}곳 중위 {rk['by_district']['median']:.2f}"
                 if "by_district" in rk else ""), flush=True)

    if args.what in ("yhold", "stability"):
        fc = d[d.free_class == 1]
        x, y = np.asarray(fc.yellow_at_site, float), (np.asarray(fc.cls, int) == 0).astype(int)
        cut, ba, P, N = ba_cut(x, y, ge_is_positive=False)     # Yellow 적을수록 Red
        yh = {"n": int(len(fc)), "n_red": P, "n_yellow": N, "cut": cut, "ba": ba,
              "acc_at_cut": float(((x <= cut).astype(int) == y).mean()),
              "acc_always_yellow": float((y == 0).mean())}
        # 카드 규칙(등급 임계 ∧ UAV 현장대기) 재현율
        pred = ((x <= cut) & (np.asarray(fc.uav_available, float) > 0)).astype(int)
        yh["acc_with_uav_gate"] = float((pred == y).mean())
        cuts = {}
        for sgc, gg in fc.groupby("sigcd"):
            if len(gg) >= args.min_n and gg.cls.nunique() == 2:
                c, b, _, _ = ba_cut(gg.yellow_at_site, (gg.cls == 0).astype(int), False)
                if np.isfinite(c):
                    cuts[str(sgc)] = float(c)
        if cuts:
            cv = np.array(list(cuts.values()))
            yh["by_district"] = {"n": len(cuts), "median": float(np.median(cv)),
                                 "q25": float(np.percentile(cv, 25)),
                                 "q75": float(np.percentile(cv, 75)),
                                 "values": cuts if args.dump_groups else None}
        res["yhold"] = yh
        print(f"  yhold 전역 {cut:.0f}명 (BA {ba:.4f}) · 재현율 {yh['acc_at_cut']:.4f} "
              f"(항상 Yellow {yh['acc_always_yellow']:.4f}, UAV 게이트 결합 "
              f"{yh['acc_with_uav_gate']:.4f}) · 등급자유 {len(fc)}건", flush=True)

    res["wall_min"] = round((time.time() - t0) / 60, 2)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[mine] 저장 {args.out}  wall={res['wall_min']}분", flush=True)


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

    # ---- mine (v18 E1): 임계값 도출 절차를 코드로 고정 ----
    m = sub.add_parser("mine", help="임계값 추정(λ 조건부로짓 · red_km/yhold BA 컷 · 안정성)")
    m.add_argument("what", choices=["lambda", "redkm", "yhold", "stability"])
    m.add_argument("--static", default=str(REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"))
    m.add_argument("--decisions", default=str(DECISIONS))
    m.add_argument("--table", default=str(REPO / "results/scoreboard/v17/fieldrules/decisions_train1000.csv"))
    m.add_argument("--bootstrap", type=int, default=0, help="시군구 클러스터 부트스트랩 반복수")
    m.add_argument("--min_n", type=int, default=15, help="지역별·분할별 추정 최소 표본")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--dump_groups", action="store_true", help="지역별 개별 추정값을 JSON 에 포함")
    m.add_argument("--out", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    {"static": static_main, "table": table_main, "logit": logit_main,
     "mine": mine_main}[args.cmd](args)


if __name__ == "__main__":
    main()
