# -*- coding: utf-8 -*-
"""v12 Track B 집계 — LB-T 전수 스윕(T=2..40 + 상한없음) 해석.

산출 3개 (계획 §Track B):
  1) T–PDR 곡선 — 전국 평균 PDR_woG(T). T=4 가 실제 최적인지, 어디서 포화하는지.
  2) 지역별 argmin_T 산포 — **T 동적화 여지의 직접 측정**. 전부 4 근처면 동적 T 는 학습 없이
     기각되고, 넓게 퍼지면 T-메타 RL(t_meta_wrapper.py)을 되살릴 근거가 된다.
  3) 두 기준선 — (a) 전국 단일 최적 T(배포 가능), (b) 지역별 argmin_T = 평가후 발췌 oracle
     (v10 프로토콜의 HEUR64 Best-of-64 와 동일 관례로 라벨).

W/T/L 은 v10/v11 관례대로 **지역별 에피소드 차이의 95% CI**(지역평균 임계값 금지).

입력: paired_eval_ladder 의 `--out` CSV + `--dump_pe` NPZ(정책×지역×에피소드 PDR).
사용: python tools/v12_lbT_report.py [--pe <npz>] [--csv <csv>] [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "results/scoreboard/v12/lbT_sweep"


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    """개선 = b−a (양수 = a 가 PDR 낮음=우수), 95% CI 로 승/무/패.

    paired_eval_ladder._paired 와 동일 규약(에피소드 차이 배열의 평균과 95% CI).
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    m = float(d.mean())
    n = d.size
    ci = 1.96 * float(d.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return m, ci, ("win" if m > ci else "loss" if m < -ci else "tie")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default=str(DEFAULT_DIR / "lbT_sweep_eval250_30ep_pe.npz"))
    ap.add_argument("--csv", default=str(DEFAULT_DIR / "lbT_sweep_eval250_30ep.csv"))
    ap.add_argument("--out_dir", default=str(DEFAULT_DIR))
    ap.add_argument("--ref", default="lb_T4", help="비교 기준 정책명")
    ap.add_argument("--train_pe", default="",
                   help="train1000 스윕 NPZ(쉼표구분 다중 — p0..p3 청크 자동 병합). 주면 시군구별 "
                        "T 를 학습좌표(시군구당 4점)에서 적합해 대표점250 에 전이한 **배포 가능** "
                        "수치를 계산한다(좌표 무중복=누수 없음).")
    A = ap.parse_args()
    out_dir = Path(A.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(A.pe, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    regions = [str(x) for x in z["regions"]]
    pdr = np.asarray(z["pdr"], dtype=np.float64)          # (지역, 정책, 에피소드)
    assert pdr.shape[:2] == (len(regions), len(names)), (pdr.shape, len(regions), len(names))
    n_eps = pdr.shape[2]

    def T_of(name: str):
        if not name.startswith("lb_T"):
            return None
        tail = name[4:]
        return np.inf if tail == "inf" else float(tail)

    t_cols = [(T_of(n), i, n) for i, n in enumerate(names) if T_of(n) is not None]
    t_cols.sort(key=lambda x: x[0])
    Ts = np.array([t for t, _, _ in t_cols])
    idx = np.array([i for _, i, _ in t_cols])
    finite = np.isfinite(Ts)

    region_mean = pdr[:, idx, :].mean(axis=2)             # (지역, T)
    nat = region_mean.mean(axis=0)                        # 전국 평균 PDR(T)

    # ---- 1) T–PDR 곡선 ----
    best_g = int(np.argmin(nat))
    print(f"[입력] 지역={len(regions)} 정책={len(names)} 에피소드/지역={n_eps} "
          f"T격자={int(finite.sum())}개 + 상한없음")
    print("\n=== 1) T–PDR 곡선 (전국 평균 PDR_woG, 낮을수록 좋음) ===")
    for k, (t, _, nm) in enumerate(t_cols):
        tag = "  ← 전국 최적" if k == best_g else ""
        label = "inf" if not np.isfinite(t) else f"{t:g}"
        print(f"  T={label:>4s}  {nat[k]:.6f}{tag}")
    print(f"\n전국 단일 최적 T = {'inf' if not np.isfinite(Ts[best_g]) else int(Ts[best_g])} "
          f"(PDR {nat[best_g]:.6f})")

    # 포화 지점: 상한없음과의 차이가 1e-6 이하가 되는 최소 T
    inf_val = nat[~finite][0] if (~finite).any() else np.nan
    sat = None
    if np.isfinite(inf_val):
        for k, t in enumerate(Ts):
            if np.isfinite(t) and abs(nat[k] - inf_val) <= 1e-6:
                sat = int(t)
                break
        print(f"포화(상한없음과 차이 ≤1e-6) 최소 T = {sat}   (상한없음 PDR {inf_val:.6f})")

    # ---- 2) 지역별 argmin_T 산포 ----
    arg = np.argmin(region_mean, axis=1)
    best_T_region = Ts[arg]
    fin = np.isfinite(best_T_region)
    print("\n=== 2) 지역별 최적 T 산포 (T 동적화 여지의 직접 측정) ===")
    vals, cnts = np.unique(best_T_region[fin], return_counts=True)
    for v, c in sorted(zip(vals, cnts), key=lambda x: -x[1])[:12]:
        print(f"  T={int(v):>3d} : {c:3d}개 지역 ({100*c/len(regions):.1f}%)")
    if (~fin).any():
        print(f"  T=inf : {int((~fin).sum())}개 지역")
    if fin.any():
        q = np.percentile(best_T_region[fin], [10, 25, 50, 75, 90])
        print(f"  분위(유한): p10={q[0]:.0f} p25={q[1]:.0f} 중앙={q[2]:.0f} "
              f"p75={q[3]:.0f} p90={q[4]:.0f}")

    # ---- 3) 두 기준선 ----
    oracle = region_mean[np.arange(len(regions)), arg]     # 지역별 argmin (평가후 발췌)
    ref_i = names.index(A.ref) if A.ref in names else None
    print("\n=== 3) 기준선 비교 (전국 평균 PDR_woG) ===")
    print(f"  전국 단일 최적 T={'inf' if not np.isfinite(Ts[best_g]) else int(Ts[best_g])}"
          f" : {nat[best_g]:.6f}   (배포 가능: 좌표 정보 불필요)")
    print(f"  지역별 argmin_T          : {oracle.mean():.6f}   "
          f"(★평가후 발췌 oracle — 배포 정책 아님)")
    print(f"  → 지역화 상한 이득       : {nat[best_g]-oracle.mean():+.6f}")
    if ref_i is not None:
        print(f"  {A.ref:22s} : {pdr[:, ref_i, :].mean():.6f}")

    # ---- 3b) 선택편향 분리: 시드 분할 정직 추정 ----
    # 지역별 argmin_T 는 30ep 에서 40여 후보의 최솟값이라 winner's curse 가 섞인다.
    # 앞 절반 시드로 T 를 고르고 뒤 절반에서 평가하면 '실제 지역 이질성' 만 남는다.
    half = n_eps // 2
    if half >= 5:
        rm_a = pdr[:, idx, :half].mean(axis=2)          # 선택용(앞 절반)
        rm_b = pdr[:, idx, half:].mean(axis=2)          # 평가용(뒤 절반)
        pick = np.argmin(rm_a, axis=1)
        honest = rm_b[np.arange(len(regions)), pick].mean()
        fixed = rm_b[:, best_g].mean()
        insample = rm_b.min(axis=1).mean()              # 뒤 절반 자체 argmin(상한)
        print(f"\n=== 3b) 시드 분할 정직 추정 (앞 {half}ep 로 T 선택 → 뒤 {n_eps-half}ep 평가) ===")
        print(f"  전국 고정 T={'inf' if not np.isfinite(Ts[best_g]) else int(Ts[best_g])}"
              f"            : {fixed:.6f}")
        print(f"  지역별 T(앞 절반서 선택)   : {honest:.6f}   → 정직 이득 {fixed-honest:+.6f}")
        print(f"  뒤 절반 자체 argmin(상한)  : {insample:.6f}   "
              f"(같은 데이터 선택 = 낙관 편향)")
        print(f"  선택편향 크기              : {(fixed-insample)-(fixed-honest):+.6f} "
              f"(= 낙관 이득 − 정직 이득)")
        picked_same = int((Ts[pick] == Ts[best_g]).sum())
        print(f"  앞 절반 선택이 전국최적과 같은 지역: {picked_same}/{len(regions)}")

    # 최적 T vs 기준 T4 의 지역별 유의성(에피소드 95% CI)
    if ref_i is not None:
        gi = idx[best_g]
        w = t_ = l_ = 0
        for r in range(len(regions)):
            _, _, verdict = paired(pdr[r, gi, :], pdr[r, ref_i, :])
            w += verdict == "win"; t_ += verdict == "tie"; l_ += verdict == "loss"
        m, ci, _ = paired(pdr[:, gi, :].ravel(), pdr[:, ref_i, :].ravel())
        print(f"\n  전국최적T vs {A.ref}: 개선 {m:+.6f} (에피소드 CI95 ±{ci:.6f}) "
              f"승/무/패 {w}/{t_}/{l_}  ※지역별 에피소드 95% CI 기준")

    # ---- 3c) train1000 적합 → 대표점250 전이 (배포 가능) ----
    transfer = None
    t_paths = [p for p in A.train_pe.split(",") if p.strip() and os.path.exists(p.strip())]
    if t_paths:
        # p0..p3 청크 병합. 정책 목록(names)과 에피소드 수가 청크 간 동일해야 한다.
        t_names, t_regions, chunks = None, [], []
        for p in t_paths:
            tz = np.load(p.strip(), allow_pickle=False)
            nm = [str(x) for x in tz["names"]]
            if t_names is None:
                t_names = nm
            elif nm != t_names:
                raise ValueError(f"청크 정책목록 불일치: {p}")
            t_regions += [str(x) for x in tz["regions"]]
            chunks.append(np.asarray(tz["pdr"], dtype=np.float64))
        if len({c.shape[2] for c in chunks}) != 1:
            raise ValueError(f"청크 에피소드 수 불일치: {[c.shape for c in chunks]}")
        t_pdr = np.concatenate(chunks, axis=0)
        t_map = {n: i for i, n in enumerate(t_names)}
        print(f"\n[train_pe] 청크 {len(t_paths)}개 병합 → {t_pdr.shape[0]}좌표 × "
              f"{len(t_names)}정책 × {t_pdr.shape[2]}ep")
        # 학습 매니페스트 키 '<이름>_<sigcd>_p<k>' → sigcd 로 묶어 4점 평균
        def sigcd_of(key: str) -> str:
            digits = [t for t in key.split("_") if t.isdigit() and len(t) == 5]
            return digits[0] if digits else ""
        by_sig: dict[str, list[int]] = {}
        for r, k in enumerate(t_regions):
            by_sig.setdefault(sigcd_of(k), []).append(r)
        # 두 스윕이 공유하는 T 만 사용(train 격자를 좁혔을 수 있음 — 생략분 명시)
        shared = [(t, k) for k, (t, _, nm) in enumerate(t_cols) if nm in t_map]
        drop = [("inf" if not np.isfinite(t) else int(t))
                for t, _, nm in t_cols if nm not in t_map]
        Ts_s = np.array([t for t, _ in shared])
        eval_k = np.array([k for _, k in shared])
        tr_i = np.array([t_map[t_cols[k][2]] for _, k in shared])
        pick_sig, n_pts = {}, {}
        for sig, rows in by_sig.items():
            prof = t_pdr[np.ix_(rows, tr_i)].mean(axis=(0, 2))   # (공유T,) 4점×30ep 평균
            pick_sig[sig] = int(np.argmin(prof))
            n_pts[sig] = len(rows)
        picked_k, hit = [], 0
        for r, k in enumerate(regions):
            sig = sigcd_of(k) or k.split("_")[-1]
            j = pick_sig.get(sig)
            picked_k.append(eval_k[j] if j is not None else best_g)
            hit += j is not None
        picked_k = np.asarray(picked_k)
        deploy = region_mean[np.arange(len(regions)), picked_k].mean()
        print(f"\n=== 3c) train1000 적합 → 대표점250 전이 (★배포 가능, 좌표 무중복) ===")
        print(f"  학습좌표 {len(t_regions)}점 / 시군구 {len(by_sig)}개 "
              f"(시군구당 {min(n_pts.values())}~{max(n_pts.values())}점 × {t_pdr.shape[2]}ep) "
              f"| 전이 매칭 {hit}/{len(regions)}")
        if drop:
            print(f"  ⚠️ train 격자에 없어 제외된 T: {drop}  (eval250 250지역 argmin 이 전부 "
                  f"≤14 이고 T≥15 는 전국 단조 악화라는 근거로 좁힘)")
        print(f"  전국 고정 T={'inf' if not np.isfinite(Ts[best_g]) else int(Ts[best_g])}"
              f"                 : {nat[best_g]:.6f}")
        print(f"  시군구별 T(학습좌표 적합)      : {deploy:.6f}   "
              f"→ **배포 가능 이득 {nat[best_g]-deploy:+.6f}**")
        print(f"  시군구별 T(대표점 자체 argmin) : {oracle.mean():.6f}   "
              f"(평가후 발췌 oracle, 이득 {nat[best_g]-oracle.mean():+.6f})")
        same = int((picked_k == best_g).sum())
        print(f"  전이 선택이 전국최적과 같은 시군구: {same}/{len(regions)}")
        gi = idx[best_g]
        w = t_ = l_ = 0
        for r in range(len(regions)):
            a = pdr[r, idx[picked_k[r]], :]
            _, _, v = paired(a, pdr[r, gi, :])
            w += v == "win"; t_ += v == "tie"; l_ += v == "loss"
        allm, allci, _ = paired(
            np.concatenate([pdr[r, idx[picked_k[r]], :] for r in range(len(regions))]),
            pdr[:, gi, :].ravel())
        print(f"  전이T vs 전국고정T: 개선 {allm:+.6f} (에피소드 CI95 ±{allci:.6f}) "
              f"승/무/패 {w}/{t_}/{l_}  ※지역별 에피소드 95% CI")
        transfer = dict(deploy_pdr=float(deploy), deploy_gain=float(nat[best_g] - deploy),
                        wtl=f"{w}/{t_}/{l_}", n_train_points=len(t_regions),
                        dropped_T=drop, same_as_national=same)

    # ---- CSV 저장 ----
    curve = out_dir / "lbT_curve_eval250.csv"
    with open(curve, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["T", "policy", "pdr_wog_mean_national", "pdr_wog_ci95_regions"])
        for k, (t, i, nm) in enumerate(t_cols):
            col = region_mean[:, k]
            ci = 1.96 * col.std(ddof=1) / np.sqrt(col.size)
            wr.writerow(["inf" if not np.isfinite(t) else int(t), nm, f"{nat[k]:.9f}", f"{ci:.9f}"])
    per = out_dir / "lbT_per_region_eval250.csv"
    with open(per, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["region", "best_T", "pdr_at_best_T", "pdr_at_T4",
                     "gain_vs_T4", "pdr_at_national_best_T"])
        t4_k = next((k for k, (t, _, _) in enumerate(t_cols) if t == 4), None)
        for r, reg in enumerate(regions):
            bt = best_T_region[r]
            row = [reg, "inf" if not np.isfinite(bt) else int(bt), f"{oracle[r]:.9f}"]
            if t4_k is not None:
                row += [f"{region_mean[r, t4_k]:.9f}", f"{region_mean[r, t4_k]-oracle[r]:.9f}"]
            else:
                row += ["", ""]
            row.append(f"{region_mean[r, best_g]:.9f}")
            wr.writerow(row)
    meta = out_dir / "lbT_report_meta.json"
    json.dump({
        "pe": os.path.abspath(A.pe), "csv": os.path.abspath(A.csv),
        "train_pe": [os.path.abspath(p) for p in t_paths],
        "n_regions": len(regions), "n_eps": n_eps,
        "T_grid": ["inf" if not np.isfinite(t) else int(t) for t, _, _ in t_cols],
        "national_best_T": "inf" if not np.isfinite(Ts[best_g]) else int(Ts[best_g]),
        "national_best_pdr": float(nat[best_g]),
        "regional_argmin_mean_pdr": float(oracle.mean()),
        "saturation_T": sat,
        "wtl_protocol": "지역별 에피소드 차이 95% CI (v10/v11 _paired 관례)",
        "transfer_train1000_to_eval250": transfer,
    }, open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {curve.name} / {per.name} / {meta.name}  (dir={out_dir})")


if __name__ == "__main__":
    main()
