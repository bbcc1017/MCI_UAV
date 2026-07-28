# -*- coding: utf-8 -*-
"""v10 의사결정나무 증류 16개와 기존 4개 정책의 공통-seed scoreboard.

최종 대표점은 정책 선택에 사용하지 않는다. 학습좌표 p2 40곳의
seed 8000..8009 폐루프 검증으로 증류 나무군과 대표 정책을 먼저 확정하고,
대표점 250곳에서는 기존 scoreboard와 동일한 seed 0..29만 결합한다.

지역별 승/무/패는 기존 v10 관례와 동일하게 같은 지역·seed의 episode
차이 95% CI로 판정한다. PDR_woG는 낮을수록 좋다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_METHODS = [
    "HEUR64_BEST",
    "LB_T4",
    "PPO_POINTER_V10",
    "PPO_POINTER_V10_NCRP_M16",
]
BASE_DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best",
    "LB_T4": "LB-T4",
    "PPO_POINTER_V10": "PPO Pointer v10",
    "PPO_POINTER_V10_NCRP_M16": "PPO + NCRP-m16",
}
INFO_ORDER = ["I0_MINIMAL", "I1_FIELD", "I2_TELEMETRY", "I3_CONNECTED"]
INFO_LABEL = {
    "I0_MINIMAL": "I0 현장최소",
    "I1_FIELD": "I1 현장기록",
    "I2_TELEMETRY": "I2 차량텔레메트리",
    "I3_CONNECTED": "I3 병원연계",
}
COMPLEXITY_ORDER = ["C1", "C2", "C3", "C4"]
COMPLEXITY_LABEL = {
    "C1": "C1 깊이3·≤8잎",
    "C2": "C2 깊이5·≤32잎",
    "C3": "C3 깊이7·≤128잎",
    "C4": "C4 깊이10·≤512잎",
}
TREE_COLORS = {
    "I0_MINIMAL": "#8C8C8C",
    "I1_FIELD": "#2E86AB",
    "I2_TELEMETRY": "#4E9F3D",
    "I3_CONNECTED": "#D66A2C",
}
BASE_COLORS = {
    "HEUR64_BEST": "#B0B0B0",
    "LB_T4": "#7AA6C2",
    "PPO_POINTER_V10": "#174A7E",
    "PPO_POINTER_V10_NCRP_M16": "#7E3F98",
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
    if len(x) <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def _paired_label(reference: np.ndarray, candidate: np.ndarray) -> tuple[str, float, float]:
    """reference-candidate > 0이면 candidate 개선."""
    d = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    mean, ci = float(d.mean()), _ci95(d)
    if mean > ci:
        return "W", mean, ci
    if mean < -ci:
        return "L", mean, ci
    return "T", mean, ci


def _load_base(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    regions = [str(x) for x in z["regions"]]
    methods = [str(x) for x in z["methods"]]
    seeds = np.asarray(z["seeds"], dtype=int)
    cube = np.asarray(z["pdr_wog"], dtype=float)
    if methods != BASE_METHODS:
        raise RuntimeError(f"기존 정책 순서 불일치: {methods}")
    if cube.shape != (250, 4, 30):
        raise RuntimeError(f"기존 cube shape 불일치: {cube.shape}")
    if not np.array_equal(seeds, np.arange(30)):
        raise RuntimeError(f"기존 seed 불일치: {seeds.tolist()}")
    if len(set(regions)) != 250:
        raise RuntimeError("기존 대표점 region 중복")
    if not np.isfinite(cube).all() or not ((0 <= cube) & (cube <= 1)).all():
        raise RuntimeError("기존 cube PDR 유한성/범위 오류")
    return regions, seeds, cube


def _load_tree(
    path: Path,
    regions: list[str],
    seeds: np.ndarray,
    cases: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    x = pd.read_csv(path)
    required = {
        "region", "policy", "info_level", "complexity", "episode", "seed",
        "pdr_woG", "n_decisions", "ms_per_decision",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"tree 평가 컬럼 누락: {sorted(missing)}")
    if len(x) != len(regions) * len(cases) * len(seeds):
        raise RuntimeError(
            f"tree 평가 행수 불일치 {len(x)} != "
            f"{len(regions) * len(cases) * len(seeds)}"
        )
    if x.duplicated(["region", "policy", "seed"]).any():
        raise RuntimeError("tree (region,policy,seed) 중복")
    if set(x["region"]) != set(regions):
        raise RuntimeError("tree와 기존 scoreboard 대표점 집합 불일치")
    if set(x["policy"]) != set(cases):
        raise RuntimeError("tree 평가 case 집합 불일치")
    if set(x["seed"]) != set(seeds.tolist()):
        raise RuntimeError("tree와 기존 scoreboard seed 집합 불일치")
    numeric = x[["pdr_woG", "n_decisions", "ms_per_decision"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not x["pdr_woG"].between(0, 1).all():
        raise RuntimeError("tree 수치 유한성/PDR 범위 오류")

    rix = {r: i for i, r in enumerate(regions)}
    cix = {c: i for i, c in enumerate(cases)}
    six = {int(s): i for i, s in enumerate(seeds)}
    cube = np.full((len(regions), len(cases), len(seeds)), np.nan, dtype=float)
    for row in x.itertuples(index=False):
        cube[rix[row.region], cix[row.policy], six[int(row.seed)]] = float(row.pdr_woG)
    if not np.isfinite(cube).all():
        raise RuntimeError("tree cube 결측")
    return cube, x


def _validation_selection(
    initial_path: Path,
    final_path: Path,
    cases: list[str],
) -> tuple[pd.DataFrame, str]:
    a = pd.read_csv(initial_path)
    b = pd.read_csv(final_path)
    keys = ["region", "policy", "episode", "seed"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise RuntimeError("내부 검증 행 중복")
    m = a.merge(b, on=keys, suffixes=("_initial", "_dagger"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise RuntimeError("초기/DAgger 내부 검증 pairing 실패")
    if set(m["policy"]) != set(cases):
        raise RuntimeError("내부 검증 case 집합 불일치")
    rows = []
    for case, g in m.groupby("policy", sort=False):
        init = float(g["pdr_woG_initial"].mean())
        final = float(g["pdr_woG_dagger"].mean())
        d = g["pdr_woG_initial"].to_numpy() - g["pdr_woG_dagger"].to_numpy()
        rows.append({
            "policy": case,
            "initial_pdr_wog": init,
            "dagger_pdr_wog": final,
            "dagger_improvement": float(d.mean()),
            "dagger_improvement_ci95_episodes": _ci95(d),
            "dagger_better": bool(final < init),
        })
    out = pd.DataFrame(rows).sort_values("dagger_pdr_wog").reset_index(drop=True)
    primary = str(out.iloc[0]["policy"])
    return out, primary


def _build_tables(
    regions: list[str],
    methods: list[str],
    cube: np.ndarray,
    tree_long: pd.DataFrame,
    fit: pd.DataFrame,
    coord_meta: pd.DataFrame,
    primary: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = cube.mean(axis=2)
    fit_by = fit.set_index("policy")
    latency = tree_long.groupby("policy").agg(
        ms_per_decision=("ms_per_decision", "mean"),
        n_decisions=("n_decisions", "sum"),
    )
    overall_rows = []
    for j, method in enumerate(methods):
        region_means = means[:, j]
        is_tree = method not in BASE_METHODS
        row = {
            "method": method,
            "display_name": (
                BASE_DISPLAY[method] if not is_tree
                else f"{INFO_LABEL[method.rsplit('_', 1)[0]]} {method.rsplit('_', 1)[1]}"
            ),
            "family": "baseline" if not is_tree else "distilled_tree",
            "info_level": "" if not is_tree else method.rsplit("_", 1)[0],
            "complexity": "" if not is_tree else method.rsplit("_", 1)[1],
            "preselected_primary": method == primary,
            "n_regions": len(regions),
            "n_episodes_per_region": cube.shape[2],
            "pdr_wog_mean": float(region_means.mean()),
            "pdr_wog_ci95_regions": _ci95(region_means),
        }
        for ref in BASE_METHODS:
            rj = methods.index(ref)
            d = means[:, rj] - region_means
            row[f"improvement_vs_{ref}"] = float(d.mean())
            row[f"improvement_vs_{ref}_ci95_regions"] = _ci95(d)
            row[f"relative_reduction_vs_{ref}_pct"] = (
                float(d.mean() / means[:, rj].mean() * 100.0)
            )
        if is_tree:
            fr = fit_by.loc[method]
            row.update({
                "depth": int(fr["depth"]),
                "leaves": int(fr["leaves"]),
                "nodes": int(fr["nodes"]),
                "n_features_available": int(fr["n_features_available"]),
                "n_features_used": int(fr["n_features_used"]),
                "offline_fidelity_full": float(fr["fidelity_full"]),
                "offline_fidelity_class": float(fr["fidelity_class"]),
                "offline_fidelity_dest": float(fr["fidelity_dest"]),
                "offline_fidelity_mode": float(fr["fidelity_mode"]),
                "ms_per_decision": float(latency.loc[method, "ms_per_decision"]),
                "total_decisions": int(latency.loc[method, "n_decisions"]),
            })
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows).sort_values("pdr_wog_mean").reset_index(drop=True)

    pair_rows = []
    for case in methods[len(BASE_METHODS):]:
        cj = methods.index(case)
        for ref in BASE_METHODS:
            rj = methods.index(ref)
            labels = []
            for i in range(len(regions)):
                label, _, _ = _paired_label(cube[i, rj], cube[i, cj])
                labels.append(label)
            dreg = means[:, rj] - means[:, cj]
            pair_rows.append({
                "reference": ref,
                "candidate": case,
                "mean_improvement": float(dreg.mean()),
                "ci95_improvement_across_regions": _ci95(dreg),
                "relative_reduction_pct": float(dreg.mean() / means[:, rj].mean() * 100.0),
                "region_W": labels.count("W"),
                "region_T": labels.count("T"),
                "region_L": labels.count("L"),
                "region_mean_better": int((dreg > 0).sum()),
                "region_mean_worse": int((dreg < 0).sum()),
            })
    pairs = pd.DataFrame(pair_rows)

    coord_meta = coord_meta.set_index("coordinate_key").loc[regions]
    sigungu_rows = []
    for ci, case in enumerate(methods[len(BASE_METHODS):], start=len(BASE_METHODS)):
        for i, region in enumerate(regions):
            row = {
                "coordinate_key": region,
                "region": coord_meta.loc[region, "region"],
                "sigcd": str(coord_meta.loc[region, "sigcd"]),
                "sido": coord_meta.loc[region, "sido"],
                "lat": float(coord_meta.loc[region, "lat"]),
                "lon": float(coord_meta.loc[region, "lon"]),
                "policy": case,
                "info_level": case.rsplit("_", 1)[0],
                "complexity": case.rsplit("_", 1)[1],
                "pdr_wog": float(means[i, ci]),
            }
            for ref in BASE_METHODS:
                rj = methods.index(ref)
                label, delta, ci95 = _paired_label(cube[i, rj], cube[i, ci])
                row[f"improvement_vs_{ref}"] = delta
                row[f"ci95_vs_{ref}"] = ci95
                row[f"wtl_vs_{ref}"] = label
            sigungu_rows.append(row)
    sigungu = pd.DataFrame(sigungu_rows)

    primary_rows = []
    pidx = methods.index(primary)
    for sido, group in coord_meta.reset_index().groupby("sido", sort=False):
        indices = [regions.index(r) for r in group["coordinate_key"]]
        row = {
            "sido": sido,
            "n_sigungu": len(indices),
            "primary_policy": primary,
            "pdr_wog_primary": float(means[indices, pidx].mean()),
        }
        for ref in BASE_METHODS:
            rj = methods.index(ref)
            d = means[indices, rj] - means[indices, pidx]
            row[f"improvement_vs_{ref}"] = float(d.mean())
            row[f"ci95_vs_{ref}_across_sigungu"] = _ci95(d)
        primary_rows.append(row)
    sido = pd.DataFrame(primary_rows).sort_values(
        "improvement_vs_PPO_POINTER_V10", ascending=False
    ).reset_index(drop=True)
    return overall, pairs, sigungu, sido


def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "NanumGothic",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 180,
    })


def _plot_matrix(overall: pd.DataFrame, out: Path) -> None:
    _set_plot_style()
    trees = overall[overall["family"].eq("distilled_tree")].copy()
    mat = np.full((4, 4), np.nan)
    for row in trees.itertuples(index=False):
        i = INFO_ORDER.index(row.info_level)
        j = COMPLEXITY_ORDER.index(row.complexity)
        mat[i, j] = row.pdr_wog_mean
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    im = ax.imshow(mat, cmap="YlGnBu_r", aspect="auto", vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(4), [COMPLEXITY_LABEL[x] for x in COMPLEXITY_ORDER])
    ax.set_yticks(range(4), [INFO_LABEL[x] for x in INFO_ORDER])
    ax.set_title("정보 수준 × 의사결정나무 복잡도: 평균 PDR_woG")
    for i in range(4):
        for j in range(4):
            color = "white" if mat[i, j] < 0.20 else "#202020"
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                    color=color, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.83)
    cb.set_label("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_tradeoff(overall: pd.DataFrame, out: Path) -> None:
    _set_plot_style()
    trees = overall[overall["family"].eq("distilled_tree")].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(4)
    for info in INFO_ORDER:
        g = trees[trees["info_level"].eq(info)].set_index("complexity").loc[COMPLEXITY_ORDER]
        ax.plot(
            x, g["pdr_wog_mean"], marker="o", lw=2, ms=6,
            color=TREE_COLORS[info], label=INFO_LABEL[info],
        )
        # 값이 가까운 I1~I3 표식은 겹치므로 끝점만 직접 표기한다.
        final = g.iloc[-1]
        offset = {
            "I0_MINIMAL": (8, 5),
            "I1_FIELD": (8, 11),
            "I2_TELEMETRY": (8, -1),
            "I3_CONNECTED": (8, -13),
        }[info]
        ax.annotate(
            f"{final.pdr_wog_mean:.3f}",
            (x[-1], final.pdr_wog_mean),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=TREE_COLORS[info],
        )
    base = overall.set_index("method")
    for method, style in [
        ("LB_T4", "--"),
        ("PPO_POINTER_V10", "-."),
        ("PPO_POINTER_V10_NCRP_M16", ":"),
    ]:
        value = float(base.loc[method, "pdr_wog_mean"])
        ax.axhline(value, color=BASE_COLORS[method], linestyle=style, lw=1.5,
                   label=f"{BASE_DISPLAY[method]} {value:.3f}")
    ax.set_xticks(
        x,
        ["C1\n깊이3·≤8잎", "C2\n깊이5·≤32잎",
         "C3\n깊이7·≤128잎", "C4\n깊이10·≤512잎"],
    )
    ax.set_ylabel("평균 PDR_woG (낮을수록 우수)")
    ax.set_xlabel("나무 복잡도")
    ax.set_title("현장 정보와 규칙 복잡도의 성능–해석가능성 절충")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_all20(overall: pd.DataFrame, primary: str, out: Path) -> None:
    _set_plot_style()
    x = overall.sort_values("pdr_wog_mean", ascending=True).copy()
    labels = []
    colors = []
    for row in x.itertuples(index=False):
        if row.method in BASE_METHODS:
            labels.append(BASE_DISPLAY[row.method])
            colors.append(BASE_COLORS[row.method])
        else:
            star = " ★" if row.method == primary else ""
            labels.append(f"{INFO_LABEL[row.info_level]} {row.complexity}{star}")
            colors.append(TREE_COLORS[row.info_level])
    fig, ax = plt.subplots(figsize=(10.6, 9.2))
    y = np.arange(len(x))
    vals = x["pdr_wog_mean"].to_numpy(float)
    cis = x["pdr_wog_ci95_regions"].to_numpy(float)
    ax.barh(y, vals, color=colors, height=0.68)
    ax.errorbar(vals, y, xerr=cis, fmt="none", ecolor="#333333", capsize=2, lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    ax.set_title("공통 seed 0–29 통합 Scoreboard: 기존 4개 + 증류나무 16개")
    ax.grid(axis="x", alpha=0.16)
    xmax = max(vals + cis) * 1.16
    ax.set_xlim(0, xmax)
    for yi, (v, ci) in enumerate(zip(vals, cis)):
        ax.text(v + xmax * 0.009, yi, f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_primary_regions(
    sigungu: pd.DataFrame,
    primary: str,
    out: Path,
) -> None:
    _set_plot_style()
    x = sigungu[sigungu["policy"].eq(primary)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    refs = ["LB_T4", "PPO_POINTER_V10", "PPO_POINTER_V10_NCRP_M16"]
    ref_labels = ["LB-T4", "PPO", "PPO+NCRP"]
    wtl = np.array([
        [
            int(x[f"wtl_vs_{r}"].eq("W").sum()),
            int(x[f"wtl_vs_{r}"].eq("T").sum()),
            int(x[f"wtl_vs_{r}"].eq("L").sum()),
        ]
        for r in refs
    ])
    left = np.zeros(3)
    colors = ["#3478A6", "#B9B9B9", "#C65A54"]
    for j, label in enumerate(["승", "무", "패"]):
        axes[0].barh(ref_labels, wtl[:, j], left=left, color=colors[j], label=label)
        for i, value in enumerate(wtl[:, j]):
            if value >= 8:
                axes[0].text(left[i] + value / 2, i, str(value), ha="center",
                             va="center", color="white" if j != 1 else "#222222")
        left += wtl[:, j]
    axes[0].set_xlim(0, 250)
    axes[0].set_xlabel("시군구 수")
    axes[0].set_title(f"{primary}: paired 95% CI 승/무/패")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")

    d = x["improvement_vs_PPO_POINTER_V10"].sort_values().to_numpy(float)
    axes[1].bar(
        np.arange(len(d)), d,
        color=np.where(d >= 0, "#3478A6", "#C65A54"), width=1,
    )
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xlabel("시군구 (PPO 대비 개선폭 오름차순)")
    axes[1].set_ylabel("PPO - 증류나무 PDR_woG")
    axes[1].set_title("양수는 증류나무가 PPO보다 우수")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_npz",
        default="results/scoreboard/v10/full1000/scoreboard_common30_episodes.npz",
    )
    p.add_argument(
        "--coord_csv",
        default="results/scoreboard/v10/full1000/scoreboard_common30_sigungu.csv",
    )
    p.add_argument(
        "--tree_csv",
        default="results/scoreboard/v10/distill/tree_eval250_seed0_29.csv",
    )
    p.add_argument(
        "--fit_csv",
        default="results/scoreboard/v10/distill/trees_final_prob/fit_summary.csv",
    )
    p.add_argument(
        "--initial_validation",
        default=(
            "results/scoreboard/v10/distill/"
            "initial_prob_closedloop40_seed8000_8009.csv"
        ),
    )
    p.add_argument(
        "--final_validation",
        default=(
            "results/scoreboard/v10/distill/"
            "final_prob_closedloop40_seed8000_8009.csv"
        ),
    )
    p.add_argument(
        "--out_dir",
        default="results/scoreboard/v10/distill/scoreboard",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_path = (repo / args.base_npz).resolve()
    coord_path = (repo / args.coord_csv).resolve()
    tree_path = (repo / args.tree_csv).resolve()
    fit_path = (repo / args.fit_csv).resolve()
    init_val_path = (repo / args.initial_validation).resolve()
    final_val_path = (repo / args.final_validation).resolve()
    out = (repo / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fit = pd.read_csv(fit_path, encoding="utf-8-sig")
    cases = [
        f"{info}_{complexity}"
        for info in INFO_ORDER
        for complexity in COMPLEXITY_ORDER
    ]
    if list(fit["policy"]) != cases:
        raise RuntimeError("fit_summary case 순서/집합 불일치")

    regions, seeds, base_cube = _load_base(base_path)
    tree_cube, tree_long = _load_tree(tree_path, regions, seeds, cases)
    validation, primary = _validation_selection(init_val_path, final_val_path, cases)
    if not validation["dagger_better"].all():
        raise RuntimeError("DAgger가 개선하지 못한 case가 있어 최종 나무군 확정 조건 위반")

    methods = BASE_METHODS + cases
    cube = np.concatenate([base_cube, tree_cube], axis=1)
    if cube.shape != (250, 20, 30):
        raise RuntimeError(f"통합 cube shape 오류: {cube.shape}")

    coords = pd.read_csv(coord_path, dtype={"sigcd": str})
    coords = coords[
        ["coordinate_key", "region", "sigcd", "sido", "lat", "lon"]
    ].drop_duplicates("coordinate_key")
    overall, pairs, sigungu, sido = _build_tables(
        regions, methods, cube, tree_long, fit, coords, primary
    )

    out_overall = out / "scoreboard_all20_overall.csv"
    out_pairs = out / "scoreboard_tree_pairwise.csv"
    out_sigungu = out / "scoreboard_tree_sigungu.csv"
    out_sido = out / "scoreboard_primary_sido.csv"
    out_validation = out / "dagger_validation_comparison.csv"
    out_npz = out / "scoreboard_all20_episodes.npz"
    overall.to_csv(out_overall, index=False, encoding="utf-8-sig")
    pairs.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    sigungu.to_csv(out_sigungu, index=False, encoding="utf-8-sig")
    sido.to_csv(out_sido, index=False, encoding="utf-8-sig")
    validation.to_csv(out_validation, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        out_npz,
        regions=np.asarray(regions),
        methods=np.asarray(methods),
        seeds=seeds,
        pdr_wog=cube,
    )

    plots = {
        "matrix": out / "tree_info_complexity_matrix.png",
        "tradeoff": out / "tree_complexity_tradeoff.png",
        "all20": out / "scoreboard_all20.png",
        "primary_regions": out / "primary_tree_regions.png",
    }
    _plot_matrix(overall, plots["matrix"])
    _plot_tradeoff(overall, plots["tradeoff"])
    _plot_all20(overall, primary, plots["all20"])
    _plot_primary_regions(sigungu, primary, plots["primary_regions"])

    meta = {
        "schema_version": 1,
        "protocol": {
            "final_eval_role": "untouched_representative_250",
            "common_seeds": [int(x) for x in seeds],
            "n_regions": len(regions),
            "n_methods": len(methods),
            "n_tree_cases": len(cases),
            "wtl": "within-region paired episode 95% CI",
            "aggregation": "equal weight per sigungu representative coordinate",
            "lower_is_better": True,
            "tree_family_selection": (
                "train-manifest p2 40 coordinates, seeds 8000..8009; "
                "final representative coordinates not used"
            ),
            "preselected_primary": primary,
            "hard_action_mask": (
                "applied to every tree; I0 is minimum field information "
                "conditional on central feasibility mask, not zero communication"
            ),
            "baseline_selection_caveat": (
                "HEUR64_BEST and LB_T4 are coordinate-local post-hoc selected "
                "with seed 0..999 and are optimistic oracle baselines"
            ),
            "multiplicity_caveat": (
                "16 tree cases are reported as a sensitivity grid; W/T/L is "
                "descriptive and not family-wise-error corrected"
            ),
        },
        "inputs": {
            str(x): _sha256(x)
            for x in [
                base_path, coord_path, tree_path, fit_path,
                init_val_path, final_val_path,
            ]
        },
        "outputs": {
            str(x): _sha256(x)
            for x in [
                out_overall, out_pairs, out_sigungu, out_sido,
                out_validation, out_npz, *plots.values(),
            ]
        },
        "git_sha": _git_sha(repo),
    }
    out_meta = out / "scoreboard_meta.json"
    out_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[scoreboard] primary={primary}")
    print(
        overall[
            ["method", "pdr_wog_mean", "pdr_wog_ci95_regions",
             "improvement_vs_PPO_POINTER_V10"]
        ].head(20).to_string(index=False)
    )
    print(f"[scoreboard] outputs={out}")


if __name__ == "__main__":
    main()
