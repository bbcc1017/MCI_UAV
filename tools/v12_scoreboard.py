# -*- coding: utf-8 -*-
"""v12 Track A 집계 — GOPT 계열 6팔 vs v10 기준선 (대표점250 seed 0–29 paired).

관례(v10/v11 승계):
  * 지표 = PDR_woG, 낮을수록 좋음.
  * 승/무/패 = **지역별 에피소드 차이의 95% CI**(`_paired` 규약). 지역평균 임계값 금지 —
    region-mean 1e-9 로 세면 tie 수가 불일치한다(v5 NCRP holdout 사례).
  * 하드게이트 = 이 실행의 V10 행이 기존 cube 와 일치. cube 는 float32 저장이므로 허용오차는
    float32 정밀도(≈1e-8 스케일)로 본다.
  * 이 라운드는 **스크리닝**이다(시드 대조군 없음) — 표에 그대로 표기한다.

사용: python tools/v12_scoreboard.py --pe <npz> [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CUBE_SIGUNGU = REPO / "results/scoreboard/v10/full1000/scoreboard_common30_sigungu.csv"
# 파라미터 수 실측(gopt_smoke [7]) — 용량 귀속을 표에서 바로 읽도록 동봉한다.
PARAMS = {
    "V10": 923_720, "X1_bilinear": 999_845, "X2_xattn1": 1_199_781,
    "X3_gopt3": 1_599_653, "X4_attn0": 907_080, "X5_cap518": 999_770,
    "X6_poolcritic": 174_216,
}
NOTE = {
    "X1_bilinear": "GOPT bilinear head (인코더=v10) ★핵심",
    "X2_xattn1": "+ 수요↔목적지 크로스어텐션 1블록",
    "X3_gopt3": "풀 GOPT 3블록·heads8",
    "X4_attn0": "attention 제거(하한)",
    "X5_cap518": "X1 용량 대조군(파라미터 −75)",
    "X6_poolcritic": "v10 actor + 순열불변 critic (1/5.3 크기)",
}


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    """개선 = b−a (양수 = a 가 PDR 낮음=우수), 에피소드 차이의 95% CI 로 승/무/패."""
    d = np.asarray(b, float) - np.asarray(a, float)
    m = float(d.mean())
    ci = 1.96 * float(d.std(ddof=1)) / np.sqrt(d.size) if d.size > 1 else 0.0
    return m, ci, ("win" if m > ci else "loss" if m < -ci else "tie")


def wtl(pdr: np.ndarray, i: int, j: int) -> tuple[int, int, int]:
    """정책 i 가 j 대비 지역별로 몇 승/무/패 (지역별 에피소드 95% CI)."""
    w = t = l = 0
    for r in range(pdr.shape[0]):
        _, _, v = paired(pdr[r, i, :], pdr[r, j, :])
        w += v == "win"; t += v == "tie"; l += v == "loss"
    return w, t, l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--ref", default="V10", help="비교 기준(하드게이트 대상)")
    A = ap.parse_args()
    out_dir = Path(A.out_dir) if A.out_dir else Path(A.pe).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(A.pe, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    regions = [str(x) for x in z["regions"]]
    pdr = np.asarray(z["pdr"], dtype=np.float64)          # (지역, 정책, 에피소드)
    n_eps = pdr.shape[2]
    region_mean = pdr.mean(axis=2)                        # (지역, 정책)
    nat = region_mean.mean(axis=0)

    # ---- 하드게이트: V10 행 vs 기존 cube ----
    gate = None
    if A.ref in names and CUBE_SIGUNGU.exists():
        ri = names.index(A.ref)
        cube = {}
        with open(CUBE_SIGUNGU, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cube[row["coordinate_key"]] = float(row["pdr_wog_PPO_POINTER_V10"])
        diffs = [abs(region_mean[r, ri] - cube[k]) for r, k in enumerate(regions) if k in cube]
        if diffs:
            gate = max(diffs)
            verdict = "PASS(float32 정밀도 이내)" if gate <= 5e-8 else "FAIL"
            print(f"[하드게이트] {A.ref} vs cube PPO_POINTER_V10: 최대오차 {gate:.3e} "
                  f"({len(diffs)}지역) → {verdict}")
            print("             ※cube 는 float32 저장 → 1e-9 이 아니라 float32 정밀도(≈1e-8)로 판정")

    # ---- 표 ----
    order = np.argsort(nat)
    ri = names.index(A.ref) if A.ref in names else int(order[0])
    print(f"\n=== v12 Track A 스크리닝: 대표점250 {len(regions)}지역 × seed 0–{n_eps-1} paired ===")
    print("※시드 대조군 없음 — 개선폭에서 학습 시드 잡음을 분리할 수 없다. 승자는 seed 1·2 복제 필요.\n")
    hdr = (f"{'정책':<16s}{'파라미터':>11s}{'PDR_woG':>10s}{'CI95(지역)':>11s}"
           f"{'vs '+A.ref:>11s}{'CI95(에피)':>11s}{'승/무/패':>12s}  설명")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for i in order:
        nm = names[i]
        col = region_mean[:, i]
        ci_reg = 1.96 * col.std(ddof=1) / np.sqrt(col.size)
        if i == ri:
            imp, ci_ep, w_t_l = 0.0, 0.0, "—"
        else:
            imp, ci_ep, _ = paired(pdr[:, i, :].ravel(), pdr[:, ri, :].ravel())
            w, t, l = wtl(pdr, i, ri)
            w_t_l = f"{w}/{t}/{l}"
        par = PARAMS.get(nm)
        print(f"{nm:<16s}{(f'{par:,}' if par else '-'):>11s}{nat[i]:10.5f}{ci_reg:11.5f}"
              f"{imp:+11.5f}{ci_ep:11.5f}{w_t_l:>12s}  {NOTE.get(nm,'')}")
        rows.append(dict(policy=nm, n_parameters=par or "", pdr_wog_mean=f"{nat[i]:.9f}",
                         pdr_wog_ci95_regions=f"{ci_reg:.9f}",
                         improvement_vs_ref=f"{imp:.9f}", ci95_episodes=f"{ci_ep:.9f}",
                         wtl_vs_ref=w_t_l, note=NOTE.get(nm, "")))

    # ---- 구조 귀속: X1 vs X5(용량 동수), X4 vs V10(attention 기여) ----
    print("\n=== 구조 귀속 (직접 paired) ===")
    for a, b, why in (("X1_bilinear", "X5_cap518", "bilinear head vs 같은 파라미터 예산의 넓은 scorer"),
                      ("V10", "X4_attn0", "현 attention 1층의 기여도"),
                      ("X2_xattn1", "X1_bilinear", "크로스어텐션 1블록의 순효과"),
                      ("X3_gopt3", "X2_xattn1", "블록 3개까지 증축의 순효과")):
        if a in names and b in names:
            ia, ib = names.index(a), names.index(b)
            m, ci, _ = paired(pdr[:, ia, :].ravel(), pdr[:, ib, :].ravel())
            w, t, l = wtl(pdr, ia, ib)
            print(f"  {a:<14s} vs {b:<14s} {m:+.5f} ±{ci:.5f}  {w}/{t}/{l}   ({why})")

    out_csv = out_dir / "v12_scoreboard_eval250.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"pe": os.path.abspath(A.pe), "n_regions": len(regions), "n_eps": n_eps,
               "ref": A.ref, "hard_gate_max_abs_err": gate,
               "round_type": "screening (시드 대조군 없음)",
               "wtl_protocol": "지역별 에피소드 차이 95% CI"},
              open(out_dir / "v12_scoreboard_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
