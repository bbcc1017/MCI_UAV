# -*- coding: utf-8 -*-
"""v10 대표점 250개 공통-seed scoreboard 생성.

입력
----
* v10_full_baselines.py가 만든 HEUR64_BEST/LB_T4 episode CSV
* paired_eval_ladder.py가 만든 PPO episode NPZ
* planner_eval.py가 만든 NCRP episode CSV

네 방법이 공통으로 보유한 seed 0..29만 사용한다. PDR_woG는 낮을수록 좋으며,
지역별 승/무/패는 동일 seed episode 차이의 95% 신뢰구간으로 판정한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "HEUR64_BEST",
    "LB_T4",
    "PPO_POINTER_V10",
    "PPO_POINTER_V10_NCRP_M16",
]
DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best",
    "LB_T4": "LB-T4",
    "PPO_POINTER_V10": "PPO Pointer v10",
    "PPO_POINTER_V10_NCRP_M16": "PPO + NCRP-m16",
}
COLORS = {
    "HEUR64_BEST": "#9AA0A6",
    "LB_T4": "#6CA0DC",
    "PPO_POINTER_V10": "#1261A0",
    "PPO_POINTER_V10_NCRP_M16": "#E07A1F",
}
SIDO = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주",
    "30": "대전", "31": "울산", "36": "세종", "41": "경기", "42": "강원",
    "43": "충북", "44": "충남", "45": "전북", "46": "전남", "47": "경북",
    "48": "경남", "50": "제주", "51": "강원", "52": "전북",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def _paired_label(reference: np.ndarray, candidate: np.ndarray) -> tuple[str, float, float]:
    """reference-candidate가 양수면 candidate 개선. 지역별 episode paired CI 판정."""
    d = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    mean, ci = float(d.mean()), _ci95(d)
    if mean > ci:
        return "W", mean, ci
    if mean < -ci:
        return "L", mean, ci
    return "T", mean, ci


def _load_baselines(path: Path, n_eps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = [
        "dataset", "coordinate_key", "region", "sigcd", "lat", "lon",
        "policy", "episode", "seed", "pdr_woG",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=use, chunksize=250_000):
        keep = chunk[
            chunk["dataset"].eq("eval250")
            & chunk["policy"].isin(["HEUR64_BEST", "LB_T4"])
            & chunk["seed"].between(0, n_eps - 1)
        ]
        if not keep.empty:
            parts.append(keep)
    if not parts:
        raise RuntimeError(f"eval250 baseline episode가 없습니다: {path}")
    raw = pd.concat(parts, ignore_index=True)
    dup = raw.duplicated(["coordinate_key", "policy", "seed"])
    if dup.any():
        raise RuntimeError(f"baseline (지역,정책,seed) 중복 {int(dup.sum())}개")
    counts = raw.groupby(["coordinate_key", "policy"])["seed"].agg(["nunique", "min", "max"])
    bad = counts[
        (counts["nunique"] != n_eps) | (counts["min"] != 0) | (counts["max"] != n_eps - 1)
    ]
    if not bad.empty:
        raise RuntimeError(f"baseline seed 불완전:\n{bad.head()}")
    coords = (
        raw[["coordinate_key", "region", "sigcd", "lat", "lon"]]
        .drop_duplicates("coordinate_key")
        .sort_values("coordinate_key")
        .reset_index(drop=True)
    )
    return raw, coords


def _load_ppo(path: Path, n_eps: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    regions = [str(x) for x in z["regions"]]
    seeds = np.asarray(z["seeds"], dtype=int)
    pdr = np.asarray(z["pdr"], dtype=float)
    if pdr.ndim != 3 or pdr.shape[1] != 1:
        raise RuntimeError(f"PPO pdr shape 이상: {pdr.shape}")
    wanted = np.arange(n_eps)
    indices = []
    for seed in wanted:
        hit = np.flatnonzero(seeds == seed)
        if len(hit) != 1:
            raise RuntimeError(f"PPO seed {seed} 개수={len(hit)}")
        indices.append(int(hit[0]))
    return regions, wanted, pdr[:, 0, indices]


def _load_ncrp(path: Path, n_eps: int) -> pd.DataFrame:
    x = pd.read_csv(path)
    required = {
        "region", "ep", "pdr_planner", "pdr_base",
        "n_dec", "n_switch", "ms_per_dec", "sec",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"NCRP 컬럼 누락: {sorted(missing)}")
    x = x[x["ep"].between(0, n_eps - 1)].copy()
    if x.duplicated(["region", "ep"]).any():
        raise RuntimeError("NCRP (지역,ep) 중복")
    counts = x.groupby("region")["ep"].agg(["nunique", "min", "max"])
    bad = counts[
        (counts["nunique"] != n_eps) | (counts["min"] != 0) | (counts["max"] != n_eps - 1)
    ]
    if len(counts) != 250 or not bad.empty or len(x) != 250 * n_eps:
        raise RuntimeError(
            f"NCRP 완전성 실패: rows={len(x)}, regions={len(counts)}, bad={len(bad)}"
        )
    numeric = ["pdr_planner", "pdr_base", "n_dec", "n_switch", "ms_per_dec", "sec"]
    if not np.isfinite(x[numeric].to_numpy(float)).all():
        raise RuntimeError("NCRP 비유한 수치 발견")
    if not x["pdr_planner"].between(0, 1).all() or not x["pdr_base"].between(0, 1).all():
        raise RuntimeError("NCRP PDR 범위 [0,1] 위반")
    return x


def _make_cube(
    baseline: pd.DataFrame,
    coords: pd.DataFrame,
    ppo_regions: list[str],
    ppo_pdr: np.ndarray,
    ncrp: pd.DataFrame,
    n_eps: int,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    coord_by_region = dict(zip(coords["coordinate_key"], coords.index))
    if set(ppo_regions) != set(coord_by_region):
        miss = sorted(set(coord_by_region) - set(ppo_regions))
        extra = sorted(set(ppo_regions) - set(coord_by_region))
        raise RuntimeError(f"PPO 지역 불일치 missing={miss[:3]} extra={extra[:3]}")

    regions = list(ppo_regions)
    coords = coords.set_index("coordinate_key").loc[regions].reset_index()
    cube = np.full((len(regions), len(METHODS), n_eps), np.nan, dtype=float)
    idx = {r: i for i, r in enumerate(regions)}
    midx = {m: i for i, m in enumerate(METHODS)}

    for row in baseline.itertuples(index=False):
        cube[idx[row.coordinate_key], midx[row.policy], int(row.seed)] = float(row.pdr_woG)
    cube[:, midx["PPO_POINTER_V10"], :] = ppo_pdr
    for row in ncrp.itertuples(index=False):
        cube[idx[row.region], midx["PPO_POINTER_V10_NCRP_M16"], int(row.ep)] = float(
            row.pdr_planner
        )
    if not np.isfinite(cube).all():
        where = np.argwhere(~np.isfinite(cube))
        raise RuntimeError(f"공통 cube 결측 {len(where)}개, 첫 위치={where[0].tolist()}")

    # planner가 같은 seed의 PPO를 재주행한 값이 기존 PPO 평가와 같은지 확인한다.
    ppo_lookup = {(r, s): cube[i, midx["PPO_POINTER_V10"], s]
                  for i, r in enumerate(regions) for s in range(n_eps)}
    base_err = np.array([
        abs(float(row.pdr_base) - ppo_lookup[(row.region, int(row.ep))])
        for row in ncrp.itertuples(index=False)
    ])
    if float(base_err.max()) > 1e-10:
        raise RuntimeError(f"NCRP pdr_base↔PPO CRN 불일치 max_abs={base_err.max():.3e}")
    return regions, cube, coords.assign(sido=coords["sigcd"].astype(str).str[:2].map(SIDO))


def _build_tables(
    regions: list[str],
    cube: np.ndarray,
    coords: pd.DataFrame,
    ncrp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = cube.mean(axis=2)
    overall_rows = []
    for j, method in enumerate(METHODS):
        region_means = means[:, j]
        row = {
            "method": method,
            "display_name": DISPLAY[method],
            "n_regions": len(regions),
            "n_episodes_per_region": cube.shape[2],
            "pdr_wog_mean": float(region_means.mean()),
            "pdr_wog_ci95_regions": _ci95(region_means),
        }
        for ref in METHODS[:3]:
            rj = METHODS.index(ref)
            d = means[:, rj] - region_means
            row[f"improvement_vs_{ref}"] = float(d.mean())
            row[f"improvement_vs_{ref}_ci95_regions"] = _ci95(d)
            row[f"relative_reduction_vs_{ref}_pct"] = (
                float(d.mean() / means[:, rj].mean() * 100.0)
            )
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)

    pair_rows = []
    for reference in METHODS:
        rj = METHODS.index(reference)
        for candidate in METHODS:
            cj = METHODS.index(candidate)
            if cj <= rj:
                continue
            labels = []
            deltas = []
            for i, region in enumerate(regions):
                label, delta, ci = _paired_label(cube[i, rj], cube[i, cj])
                labels.append(label)
                deltas.append(delta)
            dreg = means[:, rj] - means[:, cj]
            pair_rows.append({
                "reference": reference,
                "candidate": candidate,
                "interpretation": "positive_improvement_means_candidate_lower_pdr",
                "mean_improvement": float(dreg.mean()),
                "ci95_improvement_across_regions": _ci95(dreg),
                "relative_reduction_pct": float(dreg.mean() / means[:, rj].mean() * 100.0),
                "region_W": labels.count("W"),
                "region_T": labels.count("T"),
                "region_L": labels.count("L"),
                "region_mean_better": int((dreg > 0).sum()),
                "region_mean_equal": int(np.isclose(dreg, 0.0, atol=1e-12).sum()),
                "region_mean_worse": int((dreg < 0).sum()),
            })
    pairs = pd.DataFrame(pair_rows)

    ncrp_group = ncrp.groupby("region", sort=False).agg(
        ncrp_decisions=("n_dec", "sum"),
        ncrp_switches=("n_switch", "sum"),
        ncrp_ms_per_dec=("ms_per_dec", "mean"),
        ncrp_seconds_per_episode=("sec", "mean"),
    )
    rows = []
    ppo_j = METHODS.index("PPO_POINTER_V10")
    ncrp_j = METHODS.index("PPO_POINTER_V10_NCRP_M16")
    t4_j = METHODS.index("LB_T4")
    heur_j = METHODS.index("HEUR64_BEST")
    for i, region in enumerate(regions):
        pn_label, pn_delta, pn_ci = _paired_label(cube[i, ppo_j], cube[i, ncrp_j])
        t4_label, t4_delta, t4_ci = _paired_label(cube[i, t4_j], cube[i, ncrp_j])
        h_label, h_delta, h_ci = _paired_label(cube[i, heur_j], cube[i, ncrp_j])
        meta = coords.iloc[i]
        ng = ncrp_group.loc[region]
        row = {
            "coordinate_key": region,
            "region": meta["region"],
            "sigcd": str(meta["sigcd"]),
            "sido": meta["sido"],
            "lat": float(meta["lat"]),
            "lon": float(meta["lon"]),
            "n_episodes": cube.shape[2],
        }
        for j, method in enumerate(METHODS):
            row[f"pdr_wog_{method}"] = float(means[i, j])
        row.update({
            "improvement_NCRP_vs_PPO": pn_delta,
            "ci95_NCRP_vs_PPO": pn_ci,
            "NCRP_vs_PPO_WTL": pn_label,
            "relative_reduction_NCRP_vs_PPO_pct": (
                pn_delta / means[i, ppo_j] * 100.0 if means[i, ppo_j] else np.nan
            ),
            "improvement_NCRP_vs_T4": t4_delta,
            "ci95_NCRP_vs_T4": t4_ci,
            "NCRP_vs_T4_WTL": t4_label,
            "improvement_NCRP_vs_HEUR64": h_delta,
            "ci95_NCRP_vs_HEUR64": h_ci,
            "NCRP_vs_HEUR64_WTL": h_label,
            "ncrp_decisions": int(ng["ncrp_decisions"]),
            "ncrp_switches": int(ng["ncrp_switches"]),
            "ncrp_switch_rate": (
                float(ng["ncrp_switches"] / ng["ncrp_decisions"])
                if ng["ncrp_decisions"] else 0.0
            ),
            "ncrp_ms_per_dec": float(ng["ncrp_ms_per_dec"]),
            "ncrp_seconds_per_episode": float(ng["ncrp_seconds_per_episode"]),
        })
        rows.append(row)
    sigungu = pd.DataFrame(rows)
    sigungu["rank_NCRP_improvement"] = (
        sigungu["improvement_NCRP_vs_PPO"].rank(method="min", ascending=False).astype(int)
    )
    sigungu = sigungu.sort_values("rank_NCRP_improvement").reset_index(drop=True)
    return overall, pairs, sigungu


def _build_sido_table(sigungu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sido, x in sigungu.groupby("sido", sort=False):
        d = x["improvement_NCRP_vs_PPO"].to_numpy(float)
        row = {
            "sido": sido,
            "n_sigungu": len(x),
            "pdr_wog_HEUR64_BEST": x["pdr_wog_HEUR64_BEST"].mean(),
            "pdr_wog_LB_T4": x["pdr_wog_LB_T4"].mean(),
            "pdr_wog_PPO_POINTER_V10": x["pdr_wog_PPO_POINTER_V10"].mean(),
            "pdr_wog_PPO_POINTER_V10_NCRP_M16": (
                x["pdr_wog_PPO_POINTER_V10_NCRP_M16"].mean()
            ),
            "improvement_NCRP_vs_PPO": d.mean(),
            "ci95_NCRP_vs_PPO_across_sigungu": _ci95(d),
            "NCRP_vs_PPO_W": int(x["NCRP_vs_PPO_WTL"].eq("W").sum()),
            "NCRP_vs_PPO_T": int(x["NCRP_vs_PPO_WTL"].eq("T").sum()),
            "NCRP_vs_PPO_L": int(x["NCRP_vs_PPO_WTL"].eq("L").sum()),
            "ncrp_switch_rate_weighted": (
                x["ncrp_switches"].sum() / x["ncrp_decisions"].sum()
            ),
            "ncrp_ms_per_dec_mean": x["ncrp_ms_per_dec"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "improvement_NCRP_vs_PPO", ascending=False
    ).reset_index(drop=True)


def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "NanumGothic",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 180,
    })


def _plot_overall(overall: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(METHODS))
    vals = overall.set_index("method").loc[METHODS, "pdr_wog_mean"].to_numpy()
    cis = overall.set_index("method").loc[METHODS, "pdr_wog_ci95_regions"].to_numpy()
    axes[0].barh(y, vals, color=[COLORS[m] for m in METHODS], height=0.62)
    axes[0].errorbar(vals, y, xerr=cis, fmt="none", ecolor="#202124", capsize=3, lw=1)
    axes[0].set_yticks(y, [DISPLAY[m] for m in METHODS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("평균 PDR_woG (낮을수록 우수)")
    axes[0].set_title("대표점 250개 · 공통 seed 0–29")
    axes[0].grid(axis="x", alpha=0.2)
    xmax = vals.max() * 1.17
    axes[0].set_xlim(0, xmax)
    for yi, (v, ci) in enumerate(zip(vals, cis)):
        axes[0].text(v + xmax * 0.012, yi, f"{v:.4f} ± {ci:.4f}", va="center", fontsize=9)

    pair_idx = pairs.set_index(["reference", "candidate"])
    comps = [
        ("HEUR64_BEST", "PPO_POINTER_V10", "PPO vs HEUR64"),
        ("LB_T4", "PPO_POINTER_V10", "PPO vs LB-T4"),
        ("PPO_POINTER_V10", "PPO_POINTER_V10_NCRP_M16", "NCRP vs PPO"),
        ("LB_T4", "PPO_POINTER_V10_NCRP_M16", "NCRP vs LB-T4"),
    ]
    labels, gains, gain_ci, colors, annotations = [], [], [], [], []
    for ref, cand, label in comps:
        r = pair_idx.loc[(ref, cand)]
        labels.append(label)
        gains.append(float(r["mean_improvement"]))
        gain_ci.append(float(r["ci95_improvement_across_regions"]))
        colors.append(COLORS[cand])
        annotations.append(f"W/T/L {int(r.region_W)}/{int(r.region_T)}/{int(r.region_L)}")
    yy = np.arange(len(labels))
    axes[1].barh(yy, gains, color=colors, height=0.62)
    axes[1].errorbar(gains, yy, xerr=gain_ci, fmt="none", ecolor="#202124", capsize=3, lw=1)
    axes[1].axvline(0, color="#444", lw=0.8)
    axes[1].set_yticks(yy, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("기준 - 후보 PDR_woG (양수=후보 개선)")
    axes[1].set_title("paired 개선폭과 시군구 W/T/L")
    axes[1].grid(axis="x", alpha=0.2)
    for yi, (v, ci, ann) in enumerate(zip(gains, gain_ci, annotations)):
        axes[1].text(v + max(gains) * 0.025, yi, f"{v:+.4f} ± {ci:.4f}\n{ann}",
                     va="center", fontsize=8.5)
    fig.suptitle("v10 최종 Scoreboard", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_sigungu(sigungu: pd.DataFrame, out: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    d = sigungu["improvement_NCRP_vs_PPO"].to_numpy()
    vmax = max(float(np.quantile(np.abs(d), 0.98)), 1e-6)

    sc = axes[0, 0].scatter(
        sigungu["lon"], sigungu["lat"], c=d, cmap="RdBu", vmin=-vmax, vmax=vmax,
        s=24, edgecolor="white", linewidth=0.25,
    )
    axes[0, 0].set_title("대표좌표별 NCRP 개선폭")
    axes[0, 0].set_xlabel("경도")
    axes[0, 0].set_ylabel("위도")
    cb = fig.colorbar(sc, ax=axes[0, 0], shrink=0.84)
    cb.set_label("PPO - NCRP PDR_woG (양수=개선)")

    x = sigungu["pdr_wog_PPO_POINTER_V10"].to_numpy()
    y = sigungu["pdr_wog_PPO_POINTER_V10_NCRP_M16"].to_numpy()
    lim = [min(x.min(), y.min()) - 0.005, max(x.max(), y.max()) + 0.005]
    axes[0, 1].scatter(x, y, c=d, cmap="RdBu", vmin=-vmax, vmax=vmax,
                       s=26, alpha=0.82, edgecolor="none")
    axes[0, 1].plot(lim, lim, "--", color="#555", lw=1)
    axes[0, 1].set(xlim=lim, ylim=lim, xlabel="PPO Pointer v10 PDR_woG",
                   ylabel="PPO + NCRP-m16 PDR_woG")
    axes[0, 1].set_title("PPO와 NCRP의 시군구별 비교")
    axes[0, 1].grid(alpha=0.15)

    ordered = sigungu.sort_values("improvement_NCRP_vs_PPO")
    od = ordered["improvement_NCRP_vs_PPO"].to_numpy()
    axes[1, 0].bar(
        np.arange(len(od)), od,
        color=np.where(od >= 0, COLORS["PPO_POINTER_V10"], "#C44E52"), width=1.0,
    )
    axes[1, 0].axhline(0, color="#333", lw=0.8)
    axes[1, 0].set_xlabel("시군구 (개선폭 오름차순)")
    axes[1, 0].set_ylabel("PPO - NCRP PDR_woG")
    axes[1, 0].set_title("250개 시군구의 NCRP 개선 분포")

    extremes = pd.concat([
        sigungu.nsmallest(8, "improvement_NCRP_vs_PPO"),
        sigungu.nlargest(8, "improvement_NCRP_vs_PPO").sort_values(
            "improvement_NCRP_vs_PPO"
        ),
    ])
    vals = extremes["improvement_NCRP_vs_PPO"].to_numpy()
    labs = [f"{r.sido} {r.region}" for r in extremes.itertuples()]
    yy = np.arange(len(extremes))
    axes[1, 1].barh(
        yy, vals, color=np.where(vals >= 0, COLORS["PPO_POINTER_V10"], "#C44E52")
    )
    axes[1, 1].axvline(0, color="#333", lw=0.8)
    axes[1, 1].set_yticks(yy, labs, fontsize=8.5)
    axes[1, 1].set_xlabel("PPO - NCRP PDR_woG")
    axes[1, 1].set_title("개선폭 상·하위 시군구")
    axes[1, 1].grid(axis="x", alpha=0.15)

    wtl = sigungu["NCRP_vs_PPO_WTL"].value_counts()
    fig.suptitle(
        "NCRP-m16 지역별 효과 "
        f"(W/T/L={int(wtl.get('W', 0))}/{int(wtl.get('T', 0))}/{int(wtl.get('L', 0))})",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_root = repo / "results/scoreboard/v10/full1000"
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root)
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--ppo", type=Path, default=None)
    ap.add_argument("--ncrp", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    baselines = (args.baselines or root / "baseline_episodes.csv.gz").resolve()
    ppo = (args.ppo or root / "rl_eval250_episodes.npz").resolve()
    ncrp_path = (
        args.ncrp or root / "ncrp_m16_eval250_seed0_29.csv"
    ).resolve()
    for p in (baselines, ppo, ncrp_path):
        if not p.exists():
            raise SystemExit(f"입력 없음: {p}")

    baseline, coords = _load_baselines(baselines, args.n_eps)
    ppo_regions, seeds, ppo_pdr = _load_ppo(ppo, args.n_eps)
    ncrp = _load_ncrp(ncrp_path, args.n_eps)
    regions, cube, coords = _make_cube(
        baseline, coords, ppo_regions, ppo_pdr, ncrp, args.n_eps
    )
    overall, pairs, sigungu = _build_tables(regions, cube, coords, ncrp)
    sido = _build_sido_table(sigungu)

    out_overall = root / "scoreboard_common30_overall.csv"
    out_pairs = root / "scoreboard_common30_pairwise.csv"
    out_sigungu = root / "scoreboard_common30_sigungu.csv"
    out_sido = root / "scoreboard_common30_sido.csv"
    out_npz = root / "scoreboard_common30_episodes.npz"
    out_plot = root / "scoreboard_common30_overall.png"
    out_region_plot = root / "scoreboard_common30_sigungu.png"
    overall.to_csv(out_overall, index=False, encoding="utf-8-sig")
    pairs.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    sigungu.to_csv(out_sigungu, index=False, encoding="utf-8-sig")
    sido.to_csv(out_sido, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        out_npz, regions=np.asarray(regions), methods=np.asarray(METHODS),
        seeds=seeds, pdr_wog=cube,
    )
    _plot_overall(overall, pairs, out_plot)
    _plot_sigungu(sigungu, out_region_plot)

    model_dir = repo / "results/rl/redesign/v10_random4_1000_pointer_s0"
    provenance_files = [
        repo / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json",
        model_dir / "final_model.zip",
        model_dir / "vecnormalize.pkl",
        model_dir / "meta.json",
        repo / "src/rl_src/planner_policy.py",
        repo / "src/rl_src/planner_eval.py",
        Path(__file__).resolve(),
    ]
    meta = {
        "protocol": "v10_random4_train__representative250_eval",
        "metric": "PDR_woG",
        "lower_is_better": True,
        "methods": METHODS,
        "n_regions": len(regions),
        "seeds": [int(x) for x in seeds],
        "common_random_numbers": {
            "realized_environment": "네 방법 모두 동일한 실제 episode seed 0..29",
            "ncrp_imagination": (
                "비천리안 reseed_base=777000, 후보 간 같은 j번째 상상미래 CRN 공유"
            ),
        },
        "ncrp": {
            "K": 8, "h": 10, "m": 16, "leaf": "none",
            "clairvoyant": False, "reseed_base": 777000,
        },
        "validation": {
            "complete_cube_shape": list(cube.shape),
            "all_finite": bool(np.isfinite(cube).all()),
            "pdr_range": [float(cube.min()), float(cube.max())],
            "ncrp_base_vs_existing_ppo_max_abs_error": float(
                np.max(np.abs(
                    ncrp.sort_values(["region", "ep"])["pdr_base"].to_numpy()
                    - np.array([
                        cube[regions.index(r), METHODS.index("PPO_POINTER_V10"), ep]
                        for r, ep in ncrp.sort_values(["region", "ep"])[["region", "ep"]]
                        .itertuples(index=False, name=None)
                    ])
                ))
            ),
        },
        "selection_note": (
            "HEUR64_BEST와 LB_T4의 좌표별 기반 규칙은 seed 0..999 전수 결과로 "
            "사후 선택된 강한 oracle 기준선이다."
        ),
        "inputs": {
            str(p.relative_to(repo)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in (baselines, ppo, ncrp_path, *provenance_files)
        },
        "outputs": [
            str(p.relative_to(repo)) for p in
            (
                out_overall, out_pairs, out_sigungu, out_sido, out_npz,
                out_plot, out_region_plot,
            )
        ],
        "git_sha": _git_sha(repo),
    }
    out_meta = root / "scoreboard_common30_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(overall[[
        "method", "pdr_wog_mean", "pdr_wog_ci95_regions",
    ]].to_string(index=False))
    print("\npaired:")
    print(pairs[[
        "reference", "candidate", "mean_improvement",
        "ci95_improvement_across_regions", "region_W", "region_T", "region_L",
    ]].to_string(index=False))
    print(f"\n저장: {root}")


if __name__ == "__main__":
    main()
