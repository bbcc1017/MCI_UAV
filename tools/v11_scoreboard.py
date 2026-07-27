# -*- coding: utf-8 -*-
"""v11 대표점250 통합 scoreboard — v10 4행 cube에 신규 방법 행을 덧붙인다.

기존 `results/scoreboard/v10/full1000/scoreboard_common30_episodes.npz`
(250지역 × 4방법 × seed 0..29)는 **무수정 재사용**하고, planner_eval.py 가 만든 신규 팔
CSV(pdr_planner)를 같은 (지역, seed) 격자에 정렬해 방법 축으로 append 한다.

공정성 검증(하드 게이트):
* 신규 팔의 `pdr_base`(같은 시드 PPO greedy 재주행)가 기존 cube 의 PPO 행과 일치해야 한다
  → 최대 절대오차를 meta 에 기록하고 tol 초과 시 실패.
* 지역 250 × seed 0..29 완전 격자만 허용(결측 시 실패).

W/T/L·CI 정의는 v10_scoreboard 와 동일(지역별 episode 차이 배열의 95% CI).

사용:
  python tools/v11_scoreboard.py \
    --extra PPO_NCRP_BEST=results/scoreboard/v11/eval250/<tag>.csv \
    --extra MILP_ROLLING=results/scoreboard/v11/eval250/milp.csv \
    --out_dir results/scoreboard/v11/eval250
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CUBE = REPO / "results/scoreboard/v10/full1000/scoreboard_common30_episodes.npz"
DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best", "LB_T4": "LB-T4",
    "PPO_POINTER_V10": "PPO Pointer v10", "PPO_POINTER_V10_NCRP_M16": "PPO + NCRP-m16",
}
COLORS = ["#9AA0A6", "#6CA0DC", "#1261A0", "#E07A1F", "#C0392B", "#2E8B57", "#8E44AD",
          "#16A085"]
SIDO = {"11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전",
        "31": "울산", "36": "세종", "41": "경기", "42": "강원", "43": "충북", "44": "충남",
        "45": "전북", "46": "전남", "47": "경북", "48": "경남", "50": "제주", "51": "강원",
        "52": "전북"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def _paired_label(reference, candidate):
    d = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    mean, ci = float(d.mean()), _ci95(d)
    return ("W" if mean > ci else "L" if mean < -ci else "T"), mean, ci


def load_extra(path: Path, regions, seeds):
    """planner_eval CSV → (pdr(250,30), base(250,30), 부가통계 dict)."""
    ridx = {r: i for i, r in enumerate(regions)}
    sidx = {int(s): j for j, s in enumerate(seeds)}
    pdr = np.full((len(regions), len(seeds)), np.nan)
    base = np.full_like(pdr, np.nan)
    nd = np.zeros(len(regions)); ns = np.zeros(len(regions)); ms = []
    sec = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i, j = ridx.get(r["region"]), sidx.get(int(r["ep"]))
            if i is None or j is None:
                continue
            pdr[i, j] = float(r["pdr_planner"])
            base[i, j] = float(r["pdr_base"])
            nd[i] += float(r["n_dec"]); ns[i] += float(r["n_switch"])
            ms.append(float(r["ms_per_dec"])); sec.append(float(r["sec"]))
    miss = int(np.isnan(pdr).sum())
    stats = {"missing_cells": miss, "ms_per_dec_mean": float(np.mean(ms)) if ms else 0.0,
             "sec_per_ep_mean": float(np.mean(sec)) if sec else 0.0,
             "switch_rate": float(ns.sum() / nd.sum()) if nd.sum() else 0.0,
             "n_dec_total": float(nd.sum())}
    return pdr, base, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[],
                    help="NAME=경로.csv (반복 가능)")
    ap.add_argument("--cube", type=Path, default=CUBE)
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results/scoreboard/v11/eval250")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="신규 팔 base vs 기존 PPO 행 허용 오차")
    A = ap.parse_args()

    z = np.load(A.cube, allow_pickle=True)
    regions = [str(x) for x in z["regions"]]
    methods = [str(x) for x in z["methods"]]
    seeds = [int(x) for x in z["seeds"]]
    cube = np.asarray(z["pdr_wog"], dtype=float)
    ppo_j = methods.index("PPO_POINTER_V10")

    extra_stats, base_err = {}, {}
    for spec in A.extra:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"신규 팔 CSV 없음: {p}")
        pdr, base, stats = load_extra(p, regions, seeds)
        if stats["missing_cells"]:
            raise SystemExit(f"{name}: 격자 결측 {stats['missing_cells']}칸 — 완주 후 재실행")
        err = float(np.nanmax(np.abs(base - cube[:, ppo_j, :])))
        if err > A.tol:
            raise SystemExit(f"{name}: base 가 기존 PPO 행과 불일치(max_abs_err={err:.3e}) "
                             "— 시드/모델/env 정합성 확인")
        cube = np.concatenate([cube, pdr[:, None, :]], axis=1)
        methods.append(name)
        extra_stats[name] = stats
        base_err[name] = err
        stats["source"] = str(p.relative_to(REPO)) if p.is_absolute() else str(p)
        stats["sha256"] = _sha256(p)

    means = cube.mean(axis=2)                      # (지역, 방법)
    A.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- overall ----
    refs = [m for m in ("HEUR64_BEST", "LB_T4", "PPO_POINTER_V10",
                        "PPO_POINTER_V10_NCRP_M16") if m in methods]
    overall = []
    for j, m in enumerate(methods):
        row = {"method": m, "display_name": DISPLAY.get(m, m),
               "n_regions": len(regions), "n_episodes_per_region": len(seeds),
               "pdr_wog_mean": float(means[:, j].mean()),
               "pdr_wog_ci95_regions": _ci95(means[:, j])}
        for ref in refs:
            rj = methods.index(ref)
            d = means[:, rj] - means[:, j]
            wtl = [_paired_label(cube[i, rj], cube[i, j])[0] for i in range(len(regions))]
            row[f"improvement_vs_{ref}"] = float(d.mean())
            row[f"improvement_vs_{ref}_ci95_regions"] = _ci95(d)
            row[f"relative_reduction_vs_{ref}_pct"] = float(
                100.0 * d.mean() / means[:, rj].mean()) if means[:, rj].mean() else 0.0
            row[f"wtl_vs_{ref}"] = f"{wtl.count('W')}/{wtl.count('T')}/{wtl.count('L')}"
        st = extra_stats.get(m, {})
        row["ms_per_dec"] = st.get("ms_per_dec_mean", "")
        row["sec_per_episode"] = st.get("sec_per_ep_mean", "")
        row["switch_rate"] = st.get("switch_rate", "")
        overall.append(row)
    with open(A.out_dir / "scoreboard_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall[0].keys()))
        w.writeheader(); w.writerows(overall)

    # ---- pairwise(전 쌍) ----
    pairs = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            ja, jb = methods.index(a), methods.index(b)
            d = means[:, ja] - means[:, jb]          # 양수 = b 우수
            wtl = [_paired_label(cube[k, ja], cube[k, jb])[0] for k in range(len(regions))]
            pairs.append({"reference": a, "candidate": b,
                          "delta_candidate_better": float(d.mean()), "ci95": _ci95(d),
                          "W": wtl.count("W"), "T": wtl.count("T"), "L": wtl.count("L"),
                          "significant": bool(abs(d.mean()) > _ci95(d))})
    with open(A.out_dir / "scoreboard_pairwise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader(); w.writerows(pairs)

    # ---- 지역별 ----
    with open(A.out_dir / "scoreboard_sigungu.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["region", "sigcd", "sido"] + [f"pdr_{m}" for m in methods]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(regions):
            sig = r.rsplit("_", 1)[-1]
            row = {"region": r, "sigcd": sig, "sido": SIDO.get(sig[:2], "?")}
            row.update({f"pdr_{m}": float(means[i, j]) for j, m in enumerate(methods)})
            w.writerow(row)

    # ---- 그림 ----
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(methods)), 4.6))
    vals = [means[:, j].mean() for j in range(len(methods))]
    errs = [_ci95(means[:, j]) for j in range(len(methods))]
    ax.bar(range(len(methods)), vals, yerr=errs, capsize=4,
           color=[COLORS[j % len(COLORS)] for j in range(len(methods))])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([DISPLAY.get(m, m).replace("_", "\n") for m in methods],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PDR_woG (낮을수록 좋음)")
    ax.set_title(f"대표점 250 × seed 0–{seeds[-1]} 통합 scoreboard")
    for x, (v, e) in enumerate(zip(vals, errs)):
        ax.text(x, v + e + 0.002, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(A.out_dir / "scoreboard_overall.png", dpi=150)

    np.savez(A.out_dir / "scoreboard_episodes.npz", regions=np.array(regions),
             methods=np.array(methods), seeds=np.array(seeds), pdr_wog=cube)
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    with open(A.out_dir / "scoreboard_meta.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "v10_random4_train__representative250_eval",
                   "metric": "PDR_woG", "lower_is_better": True, "methods": methods,
                   "n_regions": len(regions), "seeds": seeds,
                   "base_cube": str(A.cube.relative_to(REPO)),
                   "base_cube_sha256": _sha256(A.cube),
                   "extra_arms": extra_stats,
                   "extra_base_vs_existing_ppo_max_abs_error": base_err,
                   "paired_definition": "지역별 episode 차이 배열의 95% CI(v10 관례)",
                   "git_sha": sha}, f, ensure_ascii=False, indent=1)

    print(f"{'방법':26s} {'PDR_woG':>9s} {'vs PPO':>9s} {'W/T/L(vs PPO)':>14s}")
    for row in sorted(overall, key=lambda x: x["pdr_wog_mean"]):
        print(f"{row['display_name']:26s} {row['pdr_wog_mean']:9.5f} "
              f"{row.get('improvement_vs_PPO_POINTER_V10', 0):+9.5f} "
              f"{row.get('wtl_vs_PPO_POINTER_V10', '-'):>14s}")
    print(f"산출 → {A.out_dir}")


if __name__ == "__main__":
    main()
