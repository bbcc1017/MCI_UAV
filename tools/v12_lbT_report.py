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
        ref_rm = pdr[:, ref_i, :].mean(axis=2)
        print(f"  {A.ref:22s} : {ref_rm.mean():.6f}")

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
        "n_regions": len(regions), "n_eps": n_eps,
        "T_grid": ["inf" if not np.isfinite(t) else int(t) for t, _, _ in t_cols],
        "national_best_T": "inf" if not np.isfinite(Ts[best_g]) else int(Ts[best_g]),
        "national_best_pdr": float(nat[best_g]),
        "regional_argmin_mean_pdr": float(oracle.mean()),
        "saturation_T": sat,
        "wtl_protocol": "지역별 에피소드 차이 95% CI (v10/v11 _paired 관례)",
    }, open(meta, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {curve.name} / {per.name} / {meta.name}  (dir={out_dir})")


if __name__ == "__main__":
    main()
