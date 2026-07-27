# -*- coding: utf-8 -*-
"""v11 NCRP/MILP 사다리 집계 — dev40(또는 임의 planner_eval CSV 묶음) 팔 비교표.

planner_eval.py 산출 CSV(region,ep,pdr_planner,pdr_base,n_dec,n_switch,ms_per_dec,sec)를
디렉터리 단위로 읽어 다음을 만든다.

1. 팔별 요약: PDR_woG 평균, base 대비 Δ(지역평균 95% CI), 지역 W/T/L, 결정당 ms, 스위치율
2. **팔 대 팔 paired 비교**: 모든 팔이 같은 (region, ep) 시드에서 같은 base 를 재주행하므로
   두 팔의 pdr_planner 차이를 직접 paired 로 볼 수 있다(v6 가 놓쳤던 검정 — h20 vs h10 이
   여기서만 유의로 드러난다). 기준 팔(--ref)에 대한 열을 따로 뽑는다.
3. base 일치 검증: 모든 팔의 pdr_base 가 (region,ep)별로 동일해야 한다(다르면 시드/모델/
   env 가 어긋난 것 → 비교 무효). 최대 절대오차를 meta 에 남긴다.

W/T/L·CI 는 `tools/v10_scoreboard.py` 와 동일 정의(지역별 episode 차이 배열의 95% CI).

사용:
  python tools/v11_ladder_report.py --dir results/scoreboard/v11/dev40 \
      --ref ref_K8h10m16 --out_prefix results/scoreboard/v11/dev40/ladder
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

# 팔 이름 → 롤아웃 예산 단위(K·m·h 상대비용, 표시용). 없으면 공백.
BUDGET_HINT = {
    "ref_K8h10m16": 1.0, "K8h20m16": 2.0, "K8h10m32": 2.0, "K8h40m16": 4.0,
    "K8h20m32": 4.0, "K8h20m16_sh": 2.0, "K8h20m16_z1": 2.0,
    "K8h20m16_milpinj": 2.25, "clair_h20m1": 0.125, "clair_hinfm1": 0.4,
    "milp": 0.0, "milp_future": 0.0,
}


def _ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def _paired_label(reference, candidate) -> tuple[str, float, float]:
    """reference − candidate 가 양수면 candidate 개선(PDR 낮음)."""
    d = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    mean, ci = float(d.mean()), _ci95(d)
    if mean > ci:
        return "W", mean, ci
    if mean < -ci:
        return "L", mean, ci
    return "T", mean, ci


def load_arm(path: Path) -> dict:
    """planner_eval 스키마 CSV 만 읽는다(집계 산출물·baseline CSV 는 None 반환)."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or "pdr_planner" not in rd.fieldnames:
            return None
        for r in rd:
            rows[(r["region"], int(r["ep"]))] = {
                "pdr": float(r["pdr_planner"]), "base": float(r["pdr_base"]),
                "n_dec": float(r["n_dec"]), "n_switch": float(r["n_switch"]),
                "ms": float(r["ms_per_dec"]), "sec": float(r["sec"]),
            }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default="ref_K8h10m16")
    ap.add_argument("--out_prefix", default="")
    ap.add_argument("--min_rows", type=int, default=200,
                    help="이 행수 미만인 팔은 미완료로 보고 표에서 제외")
    A = ap.parse_args()

    d = Path(A.dir)
    arms = {}
    for p in sorted(d.glob("*.csv")):
        if p.name.endswith(".meta.json"):
            continue
        rows = load_arm(p)
        if rows is None:
            continue
        if len(rows) < A.min_rows:
            print(f"[skip] {p.stem}: {len(rows)}행(미완료)")
            continue
        arms[p.stem] = rows
    if not arms:
        raise SystemExit("완료된 팔 없음")

    # 공통 (region, ep) 교집합에서만 비교
    common = set.intersection(*[set(v) for v in arms.values()])
    common = sorted(common)
    regions = sorted({k[0] for k in common})
    print(f"[v11] 팔 {len(arms)}개, 공통 에피소드 {len(common)}개, 지역 {len(regions)}개")

    # base 일치 검증
    ref_name = A.ref if A.ref in arms else sorted(arms)[0]
    base_err = 0.0
    for name, rows in arms.items():
        for k in common:
            base_err = max(base_err, abs(rows[k]["base"] - arms[ref_name][k]["base"]))
    print(f"[v11] base(PPO greedy) 최대 절대오차 = {base_err:.2e}")

    by_region = {r: [k for k in common if k[0] == r] for r in regions}
    base_arr = {r: np.array([arms[ref_name][k]["base"] for k in by_region[r]]) for r in regions}

    def arr(name, r):
        return np.array([arms[name][k]["pdr"] for k in by_region[r]])

    # ---- 팔별 요약 ----
    summary = []
    for name in sorted(arms):
        pdr_all = np.array([arms[name][k]["pdr"] for k in common])
        reg_delta = np.array([base_arr[r].mean() - arr(name, r).mean() for r in regions])
        labels = [_paired_label(base_arr[r], arr(name, r))[0] for r in regions]
        nd = np.array([arms[name][k]["n_dec"] for k in common])
        ns = np.array([arms[name][k]["n_switch"] for k in common])
        ms = np.array([arms[name][k]["ms"] for k in common])
        sec = np.array([arms[name][k]["sec"] for k in common])
        summary.append({
            "arm": name, "budget_unit": BUDGET_HINT.get(name, ""),
            "n_ep": len(common), "n_region": len(regions),
            "pdr_wog": pdr_all.mean(),
            "delta_vs_base": reg_delta.mean(), "delta_ci95": _ci95(reg_delta),
            "W": labels.count("W"), "T": labels.count("T"), "L": labels.count("L"),
            "ms_per_dec": ms.mean(), "sec_per_ep": sec.mean(),
            "n_dec_per_ep": nd.mean(),
            "switch_rate": float(ns.sum() / nd.sum()) if nd.sum() else 0.0,
        })
    base_pdr = float(np.mean([arms[ref_name][k]["base"] for k in common]))
    summary.append({"arm": "(base) PPO greedy", "budget_unit": 0.0, "n_ep": len(common),
                    "n_region": len(regions), "pdr_wog": base_pdr,
                    "delta_vs_base": 0.0, "delta_ci95": 0.0, "W": 0, "T": len(regions),
                    "L": 0, "ms_per_dec": 0.0, "sec_per_ep": 0.0, "n_dec_per_ep": 0.0,
                    "switch_rate": 0.0})
    summary.sort(key=lambda x: -x["delta_vs_base"])

    # ---- 팔 대 팔 paired (전체 쌍 + ref 열) ----
    names = sorted(arms)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            reg_d = np.array([arr(b, r).mean() - arr(a, r).mean() for r in regions])
            labels = [_paired_label(arr(b, r), arr(a, r))[0] for r in regions]
            pairs.append({"arm_a": a, "arm_b": b, "delta_a_better": reg_d.mean(),
                          "ci95": _ci95(reg_d), "W_a": labels.count("W"),
                          "T": labels.count("T"), "L_a": labels.count("L"),
                          "significant": abs(reg_d.mean()) > _ci95(reg_d)})

    prefix = A.out_prefix or str(d / "ladder")
    with open(prefix + "_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(prefix + "_pairwise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"dir": str(d), "arms": names, "ref": ref_name,
                   "n_common_episodes": len(common), "n_regions": len(regions),
                   "base_max_abs_err": base_err,
                   "paired_definition": "지역별 episode 차이 배열의 95% CI(v10 관례)"},
                  f, ensure_ascii=False, indent=1)

    # ---- 콘솔 표 ----
    print(f"\n{'팔':22s} {'예산':>5s} {'PDR_woG':>9s} {'Δvs base':>10s} {'±CI':>8s} "
          f"{'W/T/L':>11s} {'ms/dec':>8s} {'switch':>7s}")
    for s in summary:
        wtl = f"{s['W']}/{s['T']}/{s['L']}"
        bu = f"{s['budget_unit']:.2f}" if s["budget_unit"] != "" else "-"
        print(f"{s['arm']:22s} {bu:>5s} {s['pdr_wog']:9.5f} {s['delta_vs_base']:+10.5f} "
              f"{s['delta_ci95']:8.5f} {wtl:>11s} {s['ms_per_dec']:8.0f} "
              f"{s['switch_rate']:7.3f}")
    print(f"\n=== {ref_name} 대비 paired (양수 = 그 팔이 우수) ===")
    for p in pairs:
        if ref_name not in (p["arm_a"], p["arm_b"]):
            continue
        # delta_a_better = mean(pdr_b − pdr_a) → 양수면 arm_a 우수. 출력은 'other 관점'으로 뒤집는다.
        a_is_ref = p["arm_a"] == ref_name
        other = p["arm_b"] if a_is_ref else p["arm_a"]
        d_other = -p["delta_a_better"] if a_is_ref else p["delta_a_better"]
        wtl = (f"{p['L_a']}/{p['T']}/{p['W_a']}" if a_is_ref
               else f"{p['W_a']}/{p['T']}/{p['L_a']}")
        print(f"  {other:22s} Δ={d_other:+.5f} ±{p['ci95']:.5f} "
              f"{'*' if p['significant'] else ' '} W/T/L={wtl}")
    print(f"\n산출: {prefix}_{{summary,pairwise,meta}}.*")


if __name__ == "__main__":
    main()
