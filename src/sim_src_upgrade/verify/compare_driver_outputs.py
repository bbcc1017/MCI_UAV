"""드라이버 산출물 대조 — 같은 드라이버를 구/신 코어로 돌린 결과가 비트동일한지 본다.

G1/G2/G4/G5 는 sim·obs 계층의 동치를 본다. 이건 그 위에서 **실제 전수평가 드라이버가
쓰는 체크포인트 NPZ 와 CSV** 가 같은지를 본다 — 논문 산출물과 직접 이어지는 마지막 관문이다.

    # 1) 원본
    python src/rl_src/v16_baseline_alignment.py --limit 1 --n_eps 20 --no_strict \\
        --workers 2 --phase run --out_dir results/sim_upgrade/cmp_orig
    # 2) 고속 코어(같은 인자)
    python src/sim_src_upgrade/drivers/run_fast.py --target v16_baseline_alignment --mask_only -- \\
        --limit 1 --n_eps 20 --no_strict --workers 2 --phase run --out_dir results/sim_upgrade/cmp_fast
    # 3) 대조
    python src/sim_src_upgrade/verify/compare_driver_outputs.py \\
        results/sim_upgrade/cmp_orig results/sim_upgrade/cmp_fast
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def compare_npz(a_root: str, b_root: str, verbose: bool) -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(a_root, "**", "*.npz"), recursive=True))
    n_bad = 0
    for fa in files:
        rel = os.path.relpath(fa, a_root)
        fb = os.path.join(b_root, rel)
        if not os.path.exists(fb):
            print(f"  MISSING {rel}")
            n_bad += 1
            continue
        with np.load(fa, allow_pickle=False) as A, np.load(fb, allow_pickle=False) as B:
            va, vb = A["values"], B["values"]
            ok_shape = va.dtype == vb.dtype and va.shape == vb.shape
            # NaN(미완료 칸) 위치까지 같아야 하고, 유한값은 비트 단위로 같아야 한다
            ok_nan = ok_shape and np.array_equal(np.isnan(va), np.isnan(vb))
            fin = np.isfinite(va) if ok_shape else None
            ok_val = ok_shape and np.array_equal(va[fin], vb[fin])
            ok = bool(ok_shape and ok_nan and ok_val)
            if not ok:
                n_bad += 1
            if verbose or not ok:
                d = float(np.nanmax(np.abs(va - vb))) if ok_shape and fin.any() else float("nan")
                print(f"  {'OK  ' if ok else 'FAIL'} {rel:<52s} shape={va.shape} maxΔ={d}")
    return len(files), n_bad


def compare_csv(a_root: str, b_root: str, verbose: bool) -> tuple[int, int]:
    """CSV/CSV.GZ 는 바이트 비교가 아니라 수치 비교(파일에 타임스탬프·경로가 섞일 수 있음)."""
    import pandas as pd

    pats = ("*.csv", "*.csv.gz")
    files = sorted(f for p in pats for f in glob.glob(os.path.join(a_root, "**", p), recursive=True))
    n_bad = 0
    for fa in files:
        rel = os.path.relpath(fa, a_root)
        fb = os.path.join(b_root, rel)
        if not os.path.exists(fb):
            print(f"  MISSING {rel}")
            n_bad += 1
            continue
        da = pd.read_csv(fa)
        db = pd.read_csv(fb)
        ok = da.shape == db.shape and list(da.columns) == list(db.columns)
        if ok:
            num = da.select_dtypes("number").columns
            ok = (da.drop(columns=num).equals(db.drop(columns=num))
                  and np.array_equal(da[num].to_numpy(), db[num].to_numpy(), equal_nan=True))
        if not ok:
            n_bad += 1
        if verbose or not ok:
            print(f"  {'OK  ' if ok else 'FAIL'} {rel}")
    return len(files), n_bad


def main() -> int:
    ap = argparse.ArgumentParser(description="드라이버 산출물 비트동일 대조")
    ap.add_argument("orig")
    ap.add_argument("fast")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"[NPZ] {args.orig}  vs  {args.fast}")
    n_npz, bad_npz = compare_npz(args.orig, args.fast, not args.quiet)
    print(f"[CSV] {args.orig}  vs  {args.fast}")
    n_csv, bad_csv = compare_csv(args.orig, args.fast, not args.quiet)

    bad = bad_npz + bad_csv
    print(f"\n[드라이버 산출물] {'PASS' if bad == 0 else 'FAIL'} — "
          f"NPZ {n_npz}개(실패 {bad_npz}) · CSV {n_csv}개(실패 {bad_csv})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
