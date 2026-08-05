# -*- coding: utf-8 -*-
"""v12 wave 2 집계 — 학습 시드 잡음 바닥 대비 구조 효과의 유의성.

핵심 질문: wave 1 에서 X4_attn0(+0.00305)·X6_poolcritic(+0.00233)이 v10 을 앞섰지만 시드
대조군이 없어 '시드 운인지 구조인지' 판별할 수 없었다(v9 에서 같은 문제로 막힘). 여기서는
동일 아키텍처 3시드(V10, V10ctrl_s1, V10ctrl_s2)로 **잡음 바닥**을 세우고, 각 팔의 3시드 평균을
그 바닥과 비교한다.

판정 규칙:
  * 시드 내 산포(within-arm sd across seeds)와 아키텍처 간 차이를 함께 제시한다.
  * 구조 효과가 유의하려면 (팔 3시드 평균 − v10 3시드 평균) 이 **시드 표준오차보다 커야** 한다.
  * 지역별 W/T/L 은 v10/v11 관례대로 에피소드 차이의 95% CI(`_paired`) — 다만 이건 에피소드
    잡음만 반영하므로 **시드 판정의 근거로 쓰지 않는다**(보조 정보).

사용: python tools/v12_wave2_report.py --pe <npz> [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CUBE_SIGUNGU = REPO / "results/scoreboard/v10/full1000/scoreboard_common30_sigungu.csv"
PARAMS = {"V10": 923_720, "X4_attn0": 907_080, "X6_pool": 174_216, "X7_a0pool": 157_576}
ARM_NOTE = {
    "V10": "기준 아키텍처(attention 1층 + flat vf) — 잡음 바닥",
    "X4_attn0": "attention 제거",
    "X6_pool": "순열불변 pooled critic",
    "X7_a0pool": "결합: attention 제거 + pooled critic",
}


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    d = np.asarray(b, float) - np.asarray(a, float)
    m = float(d.mean())
    ci = 1.96 * float(d.std(ddof=1)) / np.sqrt(d.size) if d.size > 1 else 0.0
    return m, ci, ("win" if m > ci else "loss" if m < -ci else "tie")


def arm_of(name: str) -> str:
    """'X4_attn0_s1' → 'X4_attn0', 'V10ctrl_s1'/'V10' → 'V10'."""
    if name == "V10" or name.startswith("V10ctrl"):
        return "V10"
    return re.sub(r"_s\d+$", "", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--out_dir", default="")
    A = ap.parse_args()
    out_dir = Path(A.out_dir) if A.out_dir else Path(A.pe).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(A.pe, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    regions = [str(x) for x in z["regions"]]
    pdr = np.asarray(z["pdr"], dtype=np.float64)
    n_eps = pdr.shape[2]
    region_mean = pdr.mean(axis=2)
    nat = region_mean.mean(axis=0)

    # ---- 하드게이트 ----
    gate = None
    if "V10" in names and CUBE_SIGUNGU.exists():
        ri = names.index("V10")
        cube = {}
        with open(CUBE_SIGUNGU, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cube[row["coordinate_key"]] = float(row["pdr_wog_PPO_POINTER_V10"])
        diffs = [abs(region_mean[r, ri] - cube[k]) for r, k in enumerate(regions) if k in cube]
        if diffs:
            gate = max(diffs)
            print(f"[하드게이트] V10 vs cube: 최대오차 {gate:.3e} ({len(diffs)}지역) → "
                  f"{'PASS' if gate <= 5e-8 else 'FAIL'}")

    # ---- 시드별 표 ----
    print(f"\n=== 시드별 PDR_woG (대표점250 {len(regions)}지역 × seed 0–{n_eps-1}) ===")
    arms: dict[str, list[tuple[str, int]]] = {}
    for i, n in enumerate(names):
        arms.setdefault(arm_of(n), []).append((n, i))
    order = ["V10", "X4_attn0", "X6_pool", "X7_a0pool"]
    order += [a for a in arms if a not in order]
    for a in order:
        if a not in arms:
            continue
        vals = [(n, nat[i]) for n, i in arms[a]]
        s = "  ".join(f"{n}={v:.5f}" for n, v in vals)
        print(f"  {a:12s} {s}")

    # ---- 잡음 바닥 ----
    base_vals = np.array([nat[i] for _, i in arms["V10"]])
    k = len(base_vals)
    base_mean, base_sd = base_vals.mean(), base_vals.std(ddof=1) if k > 1 else 0.0
    base_se = base_sd / np.sqrt(k) if k > 1 else 0.0
    print(f"\n=== 잡음 바닥 (V10 아키텍처 {k}시드) ===")
    print(f"  평균 {base_mean:.5f}  시드 sd {base_sd:.5f}  시드 SE {base_se:.5f}  "
          f"범위 {base_vals.min():.5f}~{base_vals.max():.5f} (폭 {base_vals.ptp():.5f})")

    # ---- 팔별 판정 ----
    print(f"\n=== 구조 효과 판정 (양수 = v10 아키텍처 대비 PDR 낮춤=개선) ===")
    hdr = (f"{'팔':12s}{'파라미터':>10s}{'3시드 평균':>11s}{'시드 sd':>9s}"
           f"{'vs 바닥':>10s}{'결합 SE':>9s}{'배수':>7s}  판정")
    print(hdr); print("-" * len(hdr))
    rows = []
    for a in order:
        if a not in arms or a == "V10":
            continue
        v = np.array([nat[i] for _, i in arms[a]])
        m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        se = np.sqrt(base_se**2 + (sd / np.sqrt(len(v)))**2) if len(v) > 1 else base_se
        diff = base_mean - m
        ratio = diff / se if se > 0 else np.inf
        verdict = ("유의(구조)" if ratio >= 2 else
                   "시사적" if ratio >= 1 else
                   "잡음 내" if ratio > -1 else "열세")
        print(f"{a:12s}{PARAMS.get(a,0):>10,}{m:11.5f}{sd:9.5f}{diff:+10.5f}{se:9.5f}"
              f"{ratio:7.2f}  {verdict}")
        rows.append(dict(arm=a, n_parameters=PARAMS.get(a, ""), seeds=len(v),
                         pdr_mean=f"{m:.9f}", seed_sd=f"{sd:.9f}",
                         diff_vs_base=f"{diff:.9f}", combined_se=f"{se:.9f}",
                         ratio=f"{ratio:.3f}", verdict=verdict, note=ARM_NOTE.get(a, "")))
    print("\n  ※'배수' = 차이 / 결합 시드 SE. ≥2 를 구조 효과로 본다(≈95%).")
    print("  ※지역별 W/T/L 은 에피소드 잡음만 반영하므로 시드 판정 근거로 쓰지 않는다(아래는 보조).")

    # ---- 보조: 각 팔 최선시드 vs V10(s0) 지역별 W/T/L ----
    if "V10" in names:
        ri = names.index("V10")
        print(f"\n=== 보조: 팔별 각 시드 vs V10(s0) 지역별 W/T/L (에피소드 95% CI) ===")
        for a in order:
            if a not in arms or a == "V10":
                continue
            for n, i in arms[a]:
                w = t = l = 0
                for r in range(len(regions)):
                    _, _, v = paired(pdr[r, i, :], pdr[r, ri, :])
                    w += v == "win"; t += v == "tie"; l += v == "loss"
                m, ci, _ = paired(pdr[:, i, :].ravel(), pdr[:, ri, :].ravel())
                print(f"  {n:16s} {m:+.5f} ±{ci:.5f}  {w}/{t}/{l}")

    out_csv = out_dir / "v12_wave2_seed_judgment.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    json.dump({"pe": os.path.abspath(A.pe), "n_regions": len(regions), "n_eps": n_eps,
               "hard_gate_max_abs_err": gate, "noise_floor_mean": float(base_mean),
               "noise_floor_seed_sd": float(base_sd), "noise_floor_seeds": int(k),
               "verdict_rule": "차이/결합시드SE ≥ 2 = 구조 효과"},
              open(out_dir / "v12_wave2_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
