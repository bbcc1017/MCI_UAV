#!/usr/bin/env python3
"""대표점 250개에서 Shin Threshold 운용방식별 PDR_woG를 그린다."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/scoreboard/v10/shin16_full1000"
NPZ_DIR = RESULT_ROOT / "work/shin/eval250"
SCOREBOARD_CSV = RESULT_ROOT / "analysis/common30_scoreboard.csv"
LB_T_SWEEP = (
    ROOT / "results/scoreboard/v12/lbT_sweep/lbT_eval250_1000ep_pe.npz"
)
OUTPUT_PATH = RESULT_ROOT / "analysis/shin_threshold_mode_comparison_seed0_29.png"

N_EPISODES = 30
METRIC_INDEX = 3
RULES = [
    ("Both UAV First", 1),
    ("Both AMB First", 2),
    ("Only AMB", 3),
    ("Only UAV", 0),
]
REFERENCE_STYLES = [
    ("PPO_POINTER_V10_NCRP_M16", "PPO + NCRP-m16", "#7D3FA0"),
    ("PPO_POINTER_V10", "PPO Pointer v10", "#145A86"),
    ("LB_T3", "LB-T3", "#2E8B57"),
    ("HEUR64_BEST", "HEUR64 Best", "#C7443E"),
]


def set_korean_font() -> None:
    font_path = Path("/home/ryu/.fonts/NanumGothic-Regular.ttf")
    if font_path.exists():
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def read_reference_values() -> dict[str, float]:
    with SCOREBOARD_CSV.open(encoding="utf-8-sig", newline="") as f:
        values = {
            row["method"]: float(row["pdr_wog_mean"])
            for row in csv.DictReader(f)
        }
    with np.load(LB_T_SWEEP, allow_pickle=False) as data:
        names = [str(x) for x in data["names"]]
        seeds = np.asarray(data["seeds"], dtype=int)
        pdr = np.asarray(data["pdr"], dtype=float)
    if pdr.shape != (250, 39, 1000) or not np.array_equal(
        seeds, np.arange(1000)
    ):
        raise RuntimeError(f"LB-T 전수 스윕 정본 형상/seed 불일치: {pdr.shape}")
    values["LB_T3"] = float(pdr[:, names.index("lb_T3"), :N_EPISODES].mean())
    return values


def calculate_rule_statistics() -> tuple[np.ndarray, np.ndarray]:
    files = sorted(NPZ_DIR.glob("*.npz"))
    if len(files) != 250:
        raise RuntimeError(f"대표점 결과가 250개여야 합니다: 현재 {len(files)}개")

    region_means = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            values = data["values"]
        if values.shape[0] < 4 or values.shape[1] < N_EPISODES:
            raise RuntimeError(f"예상하지 못한 결과 배열 크기: {path} {values.shape}")
        region_means.append(
            [values[rule_idx, :N_EPISODES, METRIC_INDEX].mean() for _, rule_idx in RULES]
        )

    region_means_array = np.asarray(region_means, dtype=np.float64)
    means = region_means_array.mean(axis=0)
    ci95 = 1.96 * region_means_array.std(axis=0, ddof=1) / np.sqrt(len(files))
    return means, ci95


def plot() -> None:
    set_korean_font()
    reference_values = read_reference_values()
    means, ci95 = calculate_rule_statistics()
    heur_value = reference_values["HEUR64_BEST"]

    fig, ax = plt.subplots(figsize=(20.48, 13.655), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(RULES))
    colors = ["#2B8CBE", "#5798B7", "#7E8B97", "#9BA6AF"]
    ax.barh(y, means, height=0.57, color=colors, edgecolor="none", zorder=2)
    ax.errorbar(
        means,
        y,
        xerr=ci95,
        fmt="none",
        ecolor="#30363B",
        elinewidth=2.2,
        capsize=8,
        capthick=2.2,
        zorder=4,
    )

    for key, _, color in REFERENCE_STYLES:
        ax.axvline(
            reference_values[key],
            color=color,
            linestyle=(0, (6, 6)),
            linewidth=2.6,
            alpha=0.95,
            zorder=3,
        )

    for idx, mean in enumerate(means):
        pct = (mean / heur_value - 1.0) * 100.0
        direction = "감소" if pct < 0 else "증가"
        ax.text(
            mean - 0.019,
            idx,
            f"{mean:.4f}\nHEUR 대비 {abs(pct):.2f}% {direction}",
            ha="right",
            va="center",
            color="white",
            fontsize=19,
            linespacing=1.45,
            zorder=5,
        )

    ax.set_yticks(y, [name for name, _ in RULES], fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(0.13, 0.405)
    ax.set_xticks([0.13, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.tick_params(axis="x", labelsize=16, colors="#5B6570", length=5, width=1.2)
    ax.tick_params(axis="y", length=0, pad=18)
    ax.set_xlabel(
        "평균 PDR_woG  ← 낮을수록 우수  (x축 0.13부터 확대)",
        fontsize=18,
        labelpad=18,
    )
    ax.xaxis.grid(True, color="#DDE2E7", linewidth=1.1, zorder=0)
    ax.yaxis.grid(False)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#8B959E")
    ax.spines["bottom"].set_linewidth(1.2)

    fig.suptitle(
        "Shin Threshold 휴리스틱: 이송수단 운용방식별 성능",
        fontsize=30,
        y=0.972,
    )
    fig.text(
        0.5,
        0.929,
        "대표점 250개 · 모든 정책 공통 seed 0–29 · PDR_woG (낮을수록 우수)"
        " · 오차막대는 지역평균 95% CI",
        ha="center",
        fontsize=17,
        color="#68737D",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=(0, (6, 6)),
            linewidth=3,
            label=f"{label}  {reference_values[key]:.4f}",
        )
        for key, label, color in REFERENCE_STYLES
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="주요 비교 정책 기준선",
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.52, 1.045),
        frameon=True,
        fontsize=15,
        title_fontsize=17,
        handlelength=3.1,
        columnspacing=2.8,
        borderpad=0.9,
    )
    legend.get_frame().set_facecolor("#F8FAFB")
    legend.get_frame().set_edgecolor("#CFD6DC")
    legend.get_frame().set_linewidth(1.1)

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.105, top=0.74)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)

    print(f"저장: {OUTPUT_PATH}")
    for (name, _), mean, ci in zip(RULES, means, ci95):
        pct = (mean / heur_value - 1.0) * 100.0
        print(f"{name:16s} mean={mean:.6f} ci95={ci:.6f} vs_HEUR={pct:+.2f}%")


if __name__ == "__main__":
    plot()
