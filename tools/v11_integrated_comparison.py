# -*- coding: utf-8 -*-
"""v10 증류 + v11 NCRP/MILP를 같은 대표점250 프로토콜로 통합한다.

입력은 모두 기존 정본 산출물만 사용한다.

* 공통 기준선·PPO·NCRP·MILP:
  ``results/scoreboard/v11/eval250/scoreboard_episodes.npz``
* CART 증류:
  ``results/scoreboard/v10/distill/tree_eval250_seed0_29.csv``
* 현장형 GBDT 증류:
  ``results/scoreboard/v10/distill/student_top4_eval250_seed0_29.csv``
* NCRP 재최적화 사다리:
  ``results/scoreboard/v11/dev40/ladder_summary.csv``

모든 본 비교 정책은 대표점 250 × seed 0..29의 동일 격자다. 외부250은 평가셋이
다르므로 이 그림에 섞지 않고 보고서에서 별도 일반화 근거로만 다룬다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
V11_CUBE = REPO / "results/scoreboard/v11/eval250/scoreboard_episodes.npz"
TREE_RAW = REPO / "results/scoreboard/v10/distill/tree_eval250_seed0_29.csv"
STUDENT_RAW = REPO / "results/scoreboard/v10/distill/student_top4_eval250_seed0_29.csv"
DEV_LADDER = REPO / "results/scoreboard/v11/dev40/ladder_summary.csv"
OUT_DIR = REPO / "results/scoreboard/v11/integrated"


POLICIES = [
    ("HEUR64_BEST", "HEUR64 Best-of-64", "휴리스틱", "64룰 사후좌표 최선"),
    ("LB_T4", "LB-T4", "휴리스틱", "발송상한 규칙"),
    ("I3_CONNECTED_C4", "증류 CART I3-C4", "증류", "병원연계 43특징·512 leaf"),
    ("MILP_ROLLING", "Rolling-horizon MILP", "OR", "I3급 정보·재최적화"),
    ("I1_FIELD_GBDT_L31_SOFT", "현장형 GBDT I1", "증류", "현장·정적 26특징"),
    ("PPO_POINTER_V10", "PPO Pointer v10", "RL", "반응형 정책"),
    ("PPO_POINTER_V10_NCRP_M16", "PPO + NCRP h10m16", "플래너", "구 채택"),
    ("PPO_POINTER_V10_NCRP_H20M16", "PPO + NCRP h20m16", "플래너", "재최적화 채택"),
    (
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        "PPO + NCRP h20m16 + MILP",
        "하이브리드",
        "v11 최종",
    ),
]

FAMILY_COLORS = {
    "휴리스틱": "#8B929A",
    "증류": "#477A69",
    "OR": "#7A5C96",
    "RL": "#1D5D8F",
    "플래너": "#C97A21",
    "하이브리드": "#A84232",
}


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def raw_policy_cube(path: Path, policy: str, regions: list[str], seeds: list[int]) -> np.ndarray:
    """정책 raw CSV를 공통 (지역, seed) 격자로 정렬한다."""
    df = pd.read_csv(path)
    df = df[df["policy"] == policy].copy()
    if df.empty:
        raise ValueError(f"{path}: 정책 없음: {policy}")
    if df.duplicated(["region", "episode"]).any():
        raise ValueError(f"{path}: {policy} 지역×seed 중복")
    p = df.pivot(index="region", columns="episode", values="pdr_woG")
    p = p.reindex(index=regions, columns=seeds)
    if p.isna().any().any():
        raise ValueError(f"{path}: {policy} 공통 격자 결측 {int(p.isna().sum().sum())}칸")
    return p.to_numpy(dtype=float)


def paired_wtl(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int, int]:
    """지역별 episode 차이 95% CI로 W/T/L을 계산한다."""
    out = []
    for i in range(reference.shape[0]):
        d = reference[i] - candidate[i]
        mu, half = float(d.mean()), ci95(d)
        out.append("W" if mu > half else "L" if mu < -half else "T")
    return out.count("W"), out.count("T"), out.count("L")


def load_policy_cubes() -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    z = np.load(V11_CUBE, allow_pickle=True)
    regions = [str(x) for x in z["regions"]]
    seeds = [int(x) for x in z["seeds"]]
    methods = [str(x) for x in z["methods"]]
    cube = np.asarray(z["pdr_wog"], dtype=float)
    if cube.shape != (250, len(methods), 30):
        raise ValueError(f"v11 cube 형상 불일치: {cube.shape}")
    if seeds != list(range(30)):
        raise ValueError(f"평가 seed 불일치: {seeds}")

    out = {m: cube[:, j, :] for j, m in enumerate(methods)}
    out["I3_CONNECTED_C4"] = raw_policy_cube(TREE_RAW, "I3_CONNECTED_C4", regions, seeds)
    out["I1_FIELD_GBDT_L31_SOFT"] = raw_policy_cube(
        STUDENT_RAW, "I1_FIELD_GBDT_L31_SOFT", regions, seeds
    )
    return regions, seeds, out


def build_scoreboard(cubes: dict[str, np.ndarray]) -> pd.DataFrame:
    heur = cubes["HEUR64_BEST"]
    ppo = cubes["PPO_POINTER_V10"]
    heur_mean = float(heur.mean())
    rows = []
    for method, label, family, detail in POLICIES:
        x = cubes[method]
        regional = x.mean(axis=1)
        d_heur = heur.mean(axis=1) - regional
        d_ppo = ppo.mean(axis=1) - regional
        w, t, l = paired_wtl(ppo, x)
        rows.append(
            {
                "method": method,
                "label": label,
                "family": family,
                "detail": detail,
                "n_regions": x.shape[0],
                "n_episodes_per_region": x.shape[1],
                "pdr_wog_mean": float(regional.mean()),
                "pdr_wog_ci95_regions": ci95(regional),
                "improvement_vs_heur": float(d_heur.mean()),
                "improvement_vs_heur_ci95_regions": ci95(d_heur),
                "relative_reduction_vs_heur_pct": float(100.0 * d_heur.mean() / heur_mean),
                "improvement_vs_ppo": float(d_ppo.mean()),
                "improvement_vs_ppo_ci95_regions": ci95(d_ppo),
                "wtl_vs_ppo": f"{w}/{t}/{l}",
            }
        )
    return pd.DataFrame(rows).sort_values("pdr_wog_mean", ignore_index=True)


def build_steps(cubes: dict[str, np.ndarray]) -> pd.DataFrame:
    comparisons = [
        (
            "h10m16 → h20m16",
            "PPO_POINTER_V10_NCRP_M16",
            "PPO_POINTER_V10_NCRP_H20M16",
        ),
        (
            "h20m16 → +MILP 후보",
            "PPO_POINTER_V10_NCRP_H20M16",
            "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        ),
        (
            "PPO → v11 최종",
            "PPO_POINTER_V10",
            "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        ),
    ]
    rows = []
    for label, ref, cand in comparisons:
        d = cubes[ref].mean(axis=1) - cubes[cand].mean(axis=1)
        w, t, l = paired_wtl(cubes[ref], cubes[cand])
        rows.append(
            {
                "comparison": label,
                "reference": ref,
                "candidate": cand,
                "improvement": float(d.mean()),
                "ci95_regions": ci95(d),
                "wtl": f"{w}/{t}/{l}",
                "significant": bool(d.mean() > ci95(d)),
            }
        )
    return pd.DataFrame(rows)


def build_tuning() -> pd.DataFrame:
    df = pd.read_csv(DEV_LADDER)
    wanted = {
        "ref_K8h10m16": ("h10 m16", 1.0, 10),
        "K8h10m32": ("h10 m32", 2.0, 10),
        "K8h20m8": ("h20 m8", 1.0, 20),
        "K8h20m16": ("h20 m16", 2.0, 20),
        "K8h20m32": ("h20 m32", 4.0, 20),
        "K8h40m16": ("h40 m16", 4.0, 40),
        "K8hinfm16": ("h∞ m16", 5.0, 999),
        "K8h20m16_milpinj": ("h20 m16 + MILP", 2.25, 20),
    }
    out = df[df["arm"].isin(wanted)].copy()
    if len(out) != len(wanted):
        missing = sorted(set(wanted) - set(out["arm"]))
        raise ValueError(f"NCRP 사다리 팔 누락: {missing}")
    out["label"] = out["arm"].map(lambda x: wanted[x][0])
    out["budget_plot"] = out["arm"].map(lambda x: wanted[x][1])
    out["horizon"] = out["arm"].map(lambda x: wanted[x][2])
    return out.sort_values(["budget_plot", "pdr_wog"], ignore_index=True)


def render(score: pd.DataFrame, steps: pd.DataFrame, tuning: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "NanumGothic",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#8A929A",
            "text.color": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#4C5660",
            "ytick.color": "#4C5660",
        }
    )
    fig = plt.figure(figsize=(16, 9), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.32, 1.0),
        height_ratios=(1.08, 0.92),
        left=0.065,
        right=0.975,
        top=0.865,
        bottom=0.09,
        wspace=0.25,
        hspace=0.39,
    )
    ax_rank = fig.add_subplot(gs[:, 0])
    ax_tune = fig.add_subplot(gs[0, 1])
    ax_step = fig.add_subplot(gs[1, 1])

    # A. 같은 대표점250 정책 순위
    rank = score.sort_values("pdr_wog_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(rank))
    for i, row in rank.iterrows():
        color = FAMILY_COLORS[row["family"]]
        ax_rank.errorbar(
            row["pdr_wog_mean"],
            i,
            xerr=row["pdr_wog_ci95_regions"],
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="#263238",
            markeredgewidth=0.5,
            markersize=8.5,
            capsize=3.5,
            lw=1.8,
            zorder=3,
        )
        ax_rank.text(
            row["pdr_wog_mean"] + row["pdr_wog_ci95_regions"] + 0.0028,
            i,
            f"{row['pdr_wog_mean']:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if row["family"] == "하이브리드" else "normal",
        )
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(rank["label"])
    ax_rank.invert_yaxis()
    ax_rank.set_xlim(0.12, 0.265)
    ax_rank.set_xlabel("PDR_woG  (낮을수록 좋음)")
    ax_rank.set_title("A. 대표점 250 정책 통합 비교", loc="left", fontweight="bold", pad=10)
    ax_rank.text(
        0,
        0.995,
        "seed 0–29, 정책당 7,500 paired episodes · 오차막대=지역평균 95% CI",
        transform=ax_rank.transAxes,
        fontsize=9,
        color="#59636E",
        va="top",
    )
    ax_rank.grid(axis="x", color="#D9DEE3", lw=0.8, alpha=0.8)
    ax_rank.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markeredgecolor="#263238",
            markersize=7,
            label=k,
        )
        for k, c in FAMILY_COLORS.items()
    ]
    ax_rank.legend(
        handles=handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.125),
        frameon=False,
        fontsize=9,
    )

    # B. NCRP 재최적화: 계산예산과 성능
    hcolors = {10: "#6E9CC3", 20: "#C97A21", 40: "#8B929A", 999: "#4C5660"}
    for _, row in tuning.iterrows():
        is_final_cand = row["arm"] == "K8h20m16_milpinj"
        color = "#A84232" if is_final_cand else hcolors[int(row["horizon"])]
        marker = "D" if is_final_cand else "o"
        ax_tune.scatter(
            row["budget_plot"],
            row["pdr_wog"],
            s=75 if is_final_cand else 55,
            color=color,
            edgecolor="#263238",
            linewidth=0.5,
            marker=marker,
            zorder=3,
        )
        dx, dy = 0.05, 0.00018
        if row["arm"] in {"K8h20m32", "K8h40m16", "K8hinfm16"}:
            dy = -0.00062
        ax_tune.text(
            row["budget_plot"] + dx,
            row["pdr_wog"] + dy,
            row["label"],
            fontsize=8.5,
            va="center",
        )
    ax_tune.set_xlim(0.75, 5.7)
    ax_tune.set_ylim(0.1467, 0.1511)
    ax_tune.set_xlabel("상대 롤아웃 계산예산  (h10·m16=1)")
    ax_tune.set_ylabel("dev40 PDR_woG")
    ax_tune.set_title("B. NCRP (h,m) 재최적화", loc="left", fontweight="bold", pad=10)
    ax_tune.text(
        0,
        0.995,
        "h20·m16이 내부 최적점; 더 깊거나 m을 늘려도 개선 없음",
        transform=ax_tune.transAxes,
        fontsize=9,
        color="#59636E",
        va="top",
    )
    ax_tune.grid(color="#D9DEE3", lw=0.8, alpha=0.8)
    ax_tune.spines[["top", "right"]].set_visible(False)

    # C. 대표점250 paired 개선폭
    ypos = np.arange(len(steps))
    colors = ["#C97A21", "#A84232", "#1D5D8F"]
    ax_step.barh(
        ypos,
        steps["improvement"],
        xerr=steps["ci95_regions"],
        color=colors,
        edgecolor="#263238",
        linewidth=0.5,
        capsize=4,
    )
    ax_step.set_yticks(ypos)
    ax_step.set_yticklabels(steps["comparison"])
    ax_step.invert_yaxis()
    ax_step.axvline(0, color="#263238", lw=0.9)
    ax_step.set_xlim(0, 0.0094)
    ax_step.set_xlabel("PDR_woG 개선량  (양수=오른쪽 정책 우수)")
    ax_step.set_title("C. 재최적화의 순차 기여", loc="left", fontweight="bold", pad=10)
    ax_step.text(
        0,
        0.995,
        "오차막대=지역 paired 개선량 95% CI",
        transform=ax_step.transAxes,
        fontsize=9,
        color="#59636E",
        va="top",
    )
    for i, row in steps.iterrows():
        ax_step.text(
            row["improvement"] + row["ci95_regions"] + 0.00018,
            i,
            f"+{row['improvement']:.5f}  ({row['wtl']})",
            va="center",
            fontsize=8.5,
        )
    ax_step.grid(axis="x", color="#D9DEE3", lw=0.8, alpha=0.8)
    ax_step.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "MCI UAV 정책 통합 비교: 증류 · PPO · NCRP 재최적화 · MILP 결합",
        x=0.065,
        y=0.958,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#19344D",
    )
    fig.text(
        0.065,
        0.91,
        "동일 평가축에서는 v11 하이브리드가 최저 PDR을 기록하고, MILP는 단독 대체보다 NCRP 후보 생성기로 기여한다.",
        ha="left",
        fontsize=11,
        color="#4C5660",
    )
    fig.text(
        0.975,
        0.025,
        "지표: PDR_woG (Green 제외 사망확률, 낮을수록 좋음)  |  출처: v10/v11 scoreboard 정본",
        ha="right",
        fontsize=8.5,
        color="#68727D",
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"v11_integrated_comparison.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_artifact(
    score: pd.DataFrame, steps: pd.DataFrame, tuning: pd.DataFrame, out: Path
) -> None:
    """Data Analytics report renderer와 재사용할 수 있는 정규 artifact를 저장한다."""
    score_rows = []
    for rank, (_, r) in enumerate(score.iterrows(), 1):
        score_rows.append(
            {
                "rank": rank,
                "method": str(r["method"]),
                "label": str(r["label"]),
                "family": str(r["family"]),
                "detail": str(r["detail"]),
                "pdr_wog_mean": float(r["pdr_wog_mean"]),
                "pdr_wog_ci95_regions": float(r["pdr_wog_ci95_regions"]),
                "relative_reduction_vs_heur_pct": float(r["relative_reduction_vs_heur_pct"]),
                "improvement_vs_ppo": float(r["improvement_vs_ppo"]),
                "wtl_vs_ppo": str(r["wtl_vs_ppo"]),
                "n_regions": int(r["n_regions"]),
                "n_episodes_per_region": int(r["n_episodes_per_region"]),
            }
        )
    step_rows = [
        {
            "comparison": str(r["comparison"]),
            "improvement": float(r["improvement"]),
            "ci95_regions": float(r["ci95_regions"]),
            "wtl": str(r["wtl"]),
            "significant": bool(r["significant"]),
        }
        for _, r in steps.iterrows()
    ]
    tuning_rows = [
        {
            "arm": str(r["arm"]),
            "label": str(r["label"]),
            "budget": float(r["budget_plot"]),
            "pdr_wog": float(r["pdr_wog"]),
            "horizon": "∞" if int(r["horizon"]) == 999 else str(int(r["horizon"])),
            "ms_per_dec": float(r["ms_per_dec"]),
            "switch_rate": float(r["switch_rate"]),
        }
        for _, r in tuning.iterrows()
    ]
    now = datetime.now(timezone.utc).isoformat()
    title = "v11 MILP·증류·NCRP 통합 비교"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "대표점250 공통 평가축의 증류·PPO·NCRP·MILP 통합 비교",
            "generatedAt": now,
            "charts": [
                {
                    "id": "policy_rank",
                    "title": "대표점 250 정책별 PDR_woG",
                    "subtitle": "정책당 7,500 paired episodes; 낮을수록 좋음",
                    "type": "horizontalBar",
                    "dataset": "scoreboard",
                    "sourceId": "integrated_scoreboard",
                    "source": {
                        "id": "integrated_scoreboard",
                        "label": "대표점250 통합 scoreboard",
                        "path": "results/scoreboard/v11/integrated/integrated_scoreboard.csv",
                        "query": {
                            "engine": "DuckDB",
                            "language": "sql",
                            "description": "대표점250 통합 정책을 PDR 오름차순으로 조회",
                            "sql": (
                                "SELECT * FROM read_csv_auto("
                                "'results/scoreboard/v11/integrated/integrated_scoreboard.csv') "
                                "ORDER BY pdr_wog_mean ASC"
                            ),
                            "tables_used": [
                                "results/scoreboard/v11/integrated/integrated_scoreboard.csv"
                            ],
                            "metric_definitions": [
                                "pdr_wog_mean: Green 제외 사망확률의 250지역×30seed 평균; 낮을수록 우수"
                            ],
                        },
                    },
                    "intent": "comparison",
                    "encodings": {
                        "x": {"field": "label", "type": "nominal", "label": "정책"},
                        "y": {
                            "field": "pdr_wog_mean",
                            "type": "quantitative",
                            "label": "평균 PDR_woG",
                        },
                        "color": {
                            "field": "family",
                            "type": "nominal",
                            "label": "정책군",
                        },
                        "tooltip": [
                            {
                                "field": "pdr_wog_ci95_regions",
                                "type": "quantitative",
                                "label": "지역 95% CI 반폭",
                            },
                            {
                                "field": "relative_reduction_vs_heur_pct",
                                "type": "quantitative",
                                "label": "HEUR 대비 감소율(%)",
                            },
                            {"field": "wtl_vs_ppo", "type": "text", "label": "vs PPO W/T/L"},
                        ],
                    },
                    "maxRows": 9,
                },
                {
                    "id": "tuning_scatter",
                    "title": "NCRP 계산예산과 PDR_woG",
                    "subtitle": "dev40 공통 난수 평가; h20·m16에서 성능이 포화",
                    "type": "scatter",
                    "dataset": "ncrp_tuning",
                    "sourceId": "ncrp_tuning",
                    "source": {
                        "id": "ncrp_tuning",
                        "label": "v11 NCRP dev40 사다리",
                        "path": "results/scoreboard/v11/integrated/ncrp_tuning_plot_data.csv",
                        "query": {
                            "engine": "DuckDB",
                            "language": "sql",
                            "description": "NCRP 사다리의 계산예산과 PDR을 조회",
                            "sql": (
                                "SELECT * FROM read_csv_auto("
                                "'results/scoreboard/v11/integrated/ncrp_tuning_plot_data.csv') "
                                "ORDER BY budget_plot ASC, pdr_wog ASC"
                            ),
                            "tables_used": [
                                "results/scoreboard/v11/integrated/ncrp_tuning_plot_data.csv"
                            ],
                            "metric_definitions": [
                                "budget_plot: K8·h10·m16 비용을 1로 둔 상대 롤아웃 예산",
                                "pdr_wog: dev40 공통난수 평균 Green 제외 사망확률",
                            ],
                        },
                    },
                    "intent": "relationship",
                    "encodings": {
                        "x": {
                            "field": "budget",
                            "type": "quantitative",
                            "label": "상대 롤아웃 예산",
                        },
                        "y": {
                            "field": "pdr_wog",
                            "type": "quantitative",
                            "label": "dev40 PDR_woG",
                        },
                        "color": {
                            "field": "horizon",
                            "type": "nominal",
                            "label": "계획지평 h",
                        },
                        "label": {"field": "label", "type": "text", "label": "조건"},
                        "tooltip": [
                            {
                                "field": "ms_per_dec",
                                "type": "quantitative",
                                "label": "결정 지연(ms)",
                            },
                            {
                                "field": "switch_rate",
                                "type": "quantitative",
                                "label": "스위치율",
                            },
                        ],
                    },
                    "maxRows": 8,
                },
                {
                    "id": "paired_steps",
                    "title": "재최적화 단계별 PDR_woG 개선량",
                    "subtitle": "대표점250 지역 paired 평균; 양수는 오른쪽 정책 우수",
                    "type": "horizontalBar",
                    "dataset": "paired_steps",
                    "sourceId": "paired_steps",
                    "source": {
                        "id": "paired_steps",
                        "label": "대표점250 paired 개선 단계",
                        "path": "results/scoreboard/v11/integrated/paired_improvement_steps.csv",
                        "query": {
                            "engine": "DuckDB",
                            "language": "sql",
                            "description": "대표점250의 순차 paired 개선량을 조회",
                            "sql": (
                                "SELECT * FROM read_csv_auto("
                                "'results/scoreboard/v11/integrated/paired_improvement_steps.csv') "
                                "ORDER BY improvement ASC"
                            ),
                            "tables_used": [
                                "results/scoreboard/v11/integrated/paired_improvement_steps.csv"
                            ],
                            "metric_definitions": [
                                "improvement: 기준정책 PDR에서 후보정책 PDR을 뺀 지역평균; 양수면 후보 우수"
                            ],
                        },
                    },
                    "intent": "comparison",
                    "encodings": {
                        "x": {
                            "field": "comparison",
                            "type": "nominal",
                            "label": "비교",
                        },
                        "y": {
                            "field": "improvement",
                            "type": "quantitative",
                            "label": "PDR_woG 개선량",
                        },
                        "tooltip": [
                            {
                                "field": "ci95_regions",
                                "type": "quantitative",
                                "label": "95% CI 반폭",
                            },
                            {"field": "wtl", "type": "text", "label": "W/T/L"},
                        ],
                    },
                    "maxRows": 3,
                },
            ],
            "tables": [
                {
                    "id": "scoreboard_table",
                    "title": "대표점250 통합 scoreboard",
                    "subtitle": "공통 seed 0–29와 동일 hard action mask 적용",
                    "dataset": "scoreboard",
                    "sourceId": "integrated_scoreboard",
                    "source": {
                        "id": "integrated_scoreboard",
                        "label": "대표점250 통합 scoreboard",
                        "path": "results/scoreboard/v11/integrated/integrated_scoreboard.csv",
                        "query": {
                            "engine": "DuckDB",
                            "language": "sql",
                            "description": "대표점250 통합 정책을 PDR 오름차순으로 조회",
                            "sql": (
                                "SELECT * FROM read_csv_auto("
                                "'results/scoreboard/v11/integrated/integrated_scoreboard.csv') "
                                "ORDER BY pdr_wog_mean ASC"
                            ),
                            "tables_used": [
                                "results/scoreboard/v11/integrated/integrated_scoreboard.csv"
                            ],
                            "metric_definitions": [
                                "pdr_wog_mean: Green 제외 사망확률의 250지역×30seed 평균; 낮을수록 우수"
                            ],
                        },
                    },
                    "defaultSort": {"field": "pdr_wog_mean", "direction": "asc"},
                    "columns": [
                        {"field": "rank", "label": "순위", "format": "number"},
                        {"field": "label", "label": "정책", "type": "text"},
                        {"field": "family", "label": "유형", "type": "text"},
                        {"field": "pdr_wog_mean", "label": "PDR_woG", "format": "number"},
                        {
                            "field": "pdr_wog_ci95_regions",
                            "label": "지역 95% CI",
                            "format": "number",
                        },
                        {
                            "field": "relative_reduction_vs_heur_pct",
                            "label": "HEUR 대비 감소(%)",
                            "format": "number",
                        },
                        {"field": "wtl_vs_ppo", "label": "vs PPO W/T/L", "type": "text"},
                    ],
                }
            ],
            "sources": [
                {
                    "id": "integrated_scoreboard",
                    "label": "대표점250 통합 scoreboard",
                    "path": "results/scoreboard/v11/integrated/integrated_scoreboard.csv",
                },
                {
                    "id": "ncrp_tuning",
                    "label": "v11 NCRP dev40 사다리",
                    "path": "results/scoreboard/v11/dev40/ladder_summary.csv",
                },
                {
                    "id": "paired_steps",
                    "label": "대표점250 paired 개선 단계",
                    "path": "results/scoreboard/v11/integrated/paired_improvement_steps.csv",
                },
                {
                    "id": "milp_model",
                    "label": "Rolling-horizon MILP 구현",
                    "path": "src/rl_src/milp_policy.py",
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "integrated_scoreboard",
                    "body": (
                        "## 기술 요약\n\n대표점 250의 동일 평가축에서 **PPO + NCRP "
                        "h20m16 + MILP 후보주입**이 PDR_woG **0.140286**으로 가장 "
                        "우수했습니다. PPO 대비 개선은 **0.008104 ±0.000741**, "
                        "W/T/L은 **215/35/0**입니다."
                    ),
                },
                {"id": "policy_chart", "type": "chart", "chartId": "policy_rank"},
                {
                    "id": "milp",
                    "type": "markdown",
                    "sourceId": "milp_model",
                    "body": (
                        "## MILP는 후보생성기로 가장 효과적이었다\n\nMILP는 트립슬롯·환자등급·"
                        "병원 치료기회의 이진 배정을 두고, 예상 치료개시 시각의 생존확률 합을 "
                        "최대화합니다. 슬롯 대수, 현장 환자수, 치료기회, 병원 발송여유를 "
                        "제약하며 적격 조합은 RL과 같은 action mask에서만 만듭니다. 단독 "
                        "MILP는 0.152909였지만, 그 행동을 NCRP 후보로 넣으면 h20m16 대비 "
                        "**0.001189 ±0.000247** 추가 개선했습니다."
                    ),
                },
                {
                    "id": "tuning",
                    "type": "markdown",
                    "sourceId": "ncrp_tuning",
                    "body": (
                        "## NCRP는 h20·m16에서 포화했다\n\nh10에서 h20으로 늘리면 "
                        "유의하게 개선됐지만, h40·h∞ 또는 m32는 계산량만 증가하거나 "
                        "악화했습니다. 깊은 지평에서는 PPO 후속정책의 궤적오차가 누적되는 "
                        "편향이 표본분산 감소 효과보다 커집니다."
                    ),
                },
                {"id": "tuning_chart", "type": "chart", "chartId": "tuning_scatter"},
                {
                    "id": "steps",
                    "type": "markdown",
                    "sourceId": "paired_steps",
                    "body": (
                        "## 재최적화의 두 단계가 모두 재현됐다\n\n대표점250에서 "
                        "h10→h20은 **+0.001762 ±0.000364**, MILP 후보 추가는 "
                        "**+0.001189 ±0.000247**로 모두 95% CI를 넘었습니다."
                    ),
                },
                {"id": "steps_chart", "type": "chart", "chartId": "paired_steps"},
                {"id": "table", "type": "table", "tableId": "scoreboard_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 비교 범위와 한계\n\n- 본 비교는 대표점250 × seed 0–29입니다.\n"
                        "- 증류 외부250은 좌표·seed가 달라 같은 막대에 섞지 않았습니다.\n"
                        "- MILP는 평균 이송시간과 기대 서버 해방간격 근사에 의존합니다.\n"
                        "- I1 GBDT는 현장정보 제한 모델이지만 단일 CART와 같은 직접 규칙집은 "
                        "아닙니다.\n- HEUR64 Best-of-64는 좌표별 사후 최선 발췌 기준입니다."
                    ),
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "body": (
                        "## 다음 단계\n\n1. 논문 본표는 대표점250 통합 scoreboard를 사용합니다.\n"
                        "2. 현장정책은 I1 GBDT, 직접 규칙 제시는 I3-C4를 구분합니다.\n"
                        "3. 최종 고성능 방법은 RL–OR 하이브리드로 기술하고 MILP 단독 결과도 "
                        "강한 OR 기준선으로 함께 보고합니다."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 추가로 답할 질문\n\n- 학습 seed 1·2에서도 정책 순위가 유지되는가?\n"
                        "- 농촌·도서에서 MILP 근사오차를 줄일 수 있는가?\n"
                        "- GBDT의 지역별 이득을 지리·자원특성과 연결할 수 있는가?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": now,
            "status": "ready",
            "datasets": {
                "scoreboard": score_rows,
                "ncrp_tuning": tuning_rows,
                "paired_steps": step_rows,
            },
        },
    }
    with open(out / "artifact.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    regions, seeds, cubes = load_policy_cubes()
    score = build_scoreboard(cubes)
    steps = build_steps(cubes)
    tuning = build_tuning()

    score.to_csv(args.out_dir / "integrated_scoreboard.csv", index=False, encoding="utf-8-sig")
    steps.to_csv(args.out_dir / "paired_improvement_steps.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(args.out_dir / "ncrp_tuning_plot_data.csv", index=False, encoding="utf-8-sig")
    render(score, steps, tuning, args.out_dir)
    write_artifact(score, steps, tuning, args.out_dir)

    print(f"대표점: {len(regions)}, seeds: {seeds[0]}..{seeds[-1]}")
    print(score[["label", "pdr_wog_mean", "pdr_wog_ci95_regions", "wtl_vs_ppo"]].to_string(index=False))
    print(f"산출: {args.out_dir}")


if __name__ == "__main__":
    main()
