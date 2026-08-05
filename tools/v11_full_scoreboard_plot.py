# -*- coding: utf-8 -*-
"""대표점 250 × seed 0..29의 정책을 Shin Best까지 한 장에 통합한다.

천리안 NCRP는 실제 정책이 아니라 성능 상한이므로 제외한다. 나머지 기준선,
PPO/NCRP/MILP, CART, GBDT, EBM, Shin Best는 모두 같은 대표점 250개와
seed 0..29 격자에서 평가된 원시 episode 결과를 사용한다. Shin Best는
각 대표점에서 seed 0..999 평균으로 16개 중 하나를 사후 선택한다.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
V11_CUBE = REPO / "results/scoreboard/v11/eval250/scoreboard_episodes.npz"
LB_T_SWEEP = (
    REPO / "results/scoreboard/v12/lbT_sweep/lbT_eval250_1000ep_pe.npz"
)
TREE_RAW = REPO / "results/scoreboard/v10/distill/tree_eval250_seed0_29.csv"
STUDENT_RAW = REPO / "results/scoreboard/v10/distill/student_top4_eval250_seed0_29.csv"
EBM_RAW = REPO / "results/scoreboard/v10/distill/ebm_best_eval250_seed0_29.csv"
SHIN_ROOT = REPO / "results/scoreboard/v10/shin16_full1000/work/shin/eval250"
OUT_DIR = REPO / "results/scoreboard/v11/integrated"


V11_METHODS = {
    "PPO_POINTER_V10_NCRP_H20M16_MILPINJ": (
        "PPO + NCRP h20m16 + MILP ★",
        "최종 RL–OR",
    ),
    "PPO_POINTER_V10_NCRP_H20M16": ("PPO + NCRP h20m16", "NCRP"),
    "PPO_POINTER_V10_NCRP_M16": ("PPO + NCRP h10m16", "NCRP"),
    "PPO_POINTER_V10": ("PPO Pointer v10", "PPO"),
    "MILP_ROLLING": ("Rolling-horizon MILP", "MILP"),
    "HEUR64_BEST": ("HEUR64 Best-of-64", "규칙 기준선"),
}

TREE_LABELS = {
    "I0_MINIMAL": "I0 현장최소",
    "I1_FIELD": "I1 현장기록",
    "I2_TELEMETRY": "I2 차량텔레메트리",
    "I3_CONNECTED": "I3 병원연계",
}

STUDENT_METHODS = {
    "I1_FIELD_GBDT_L31_SOFT": "증류 GBDT I1 현장기록 L31 ★",
    "I3_CONNECTED_GBDT_L63_BASE": "증류 GBDT I3 병원연계 L63",
    "I3_CONNECTED_GBDT_L31_BASE": "증류 GBDT I3 병원연계 L31",
    "I3_CONNECTED_CART_L384": "증류 CART I3 병원연계 L384",
}

COLORS = {
    "최종 RL–OR": "#A84232",
    "NCRP": "#8045A3",
    "PPO": "#285D88",
    "MILP": "#B07A35",
    "규칙 기준선": "#A8A8A8",
    "문헌 휴리스틱": "#8B7430",
    "I0": "#929292",
    "I1": "#3888AA",
    "I2": "#5A9F45",
    "I3": "#DE7026",
}


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(values.size))


def read_raw_cube(
    path: Path,
    policy: str,
    regions: list[str],
    seeds: list[int],
) -> np.ndarray:
    """원시 CSV를 정확한 (지역 250, seed 30) 격자로 정렬한다."""
    df = pd.read_csv(path)
    value_col = "pdr_woG" if "pdr_woG" in df.columns else "pdr_wog"
    df = df[df["policy"] == policy].copy()
    if len(df) != len(regions) * len(seeds):
        raise ValueError(
            f"{path.name}:{policy} 표본수 불일치 "
            f"{len(df)} != {len(regions) * len(seeds)}"
        )
    if df.duplicated(["region", "seed"]).any():
        raise ValueError(f"{path.name}:{policy} 지역×seed 중복")

    pivot = df.pivot(index="region", columns="seed", values=value_col)
    pivot = pivot.reindex(index=regions, columns=seeds)
    if pivot.isna().any().any():
        raise ValueError(
            f"{path.name}:{policy} 공통 격자 결측 "
            f"{int(pivot.isna().sum().sum())}칸"
        )
    return pivot.to_numpy(dtype=float)


def info_family(policy: str) -> str:
    if policy.startswith("I0_"):
        return "I0"
    if policy.startswith("I1_"):
        return "I1"
    if policy.startswith("I2_"):
        return "I2"
    if policy.startswith("I3_"):
        return "I3"
    raise ValueError(f"정보수준을 알 수 없는 정책: {policy}")


def load_shin_best_cube(regions: list[str], seeds: list[int]) -> np.ndarray:
    """대표점별 seed 0..999 평균 Best-of-16의 공통 seed 구간을 읽는다."""
    if seeds != list(range(30)):
        raise ValueError(f"Shin 공통 seed 불일치: {seeds}")

    rows = []
    expected_rules = None
    for region in regions:
        path = SHIN_ROOT / f"{region}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Shin 체크포인트 누락: {path}")
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["values"], dtype=float)
            done = np.asarray(data["done"], dtype=bool)
            all_seeds = np.asarray(data["seeds"], dtype=int)
            rules = tuple(str(x) for x in data["rule_names"])
        if values.shape != (16, 1000, 5) or not done.all():
            raise ValueError(f"Shin 체크포인트 불완전: {path}")
        if not np.array_equal(all_seeds, np.arange(1000)):
            raise ValueError(f"Shin seed 불일치: {path}")
        if expected_rules is None:
            expected_rules = rules
        elif rules != expected_rules:
            raise ValueError(f"Shin 규칙 순서 불일치: {path}")

        best_idx = int(np.argmin(values[:, :, 3].mean(axis=1)))
        rows.append(values[best_idx, :30, 3])
    return np.stack(rows)


def tree_label(policy: str) -> str:
    for prefix, name in TREE_LABELS.items():
        if policy.startswith(prefix + "_C"):
            complexity = policy.rsplit("_", 1)[-1]
            star = " ★" if policy == "I3_CONNECTED_C4" else ""
            return f"증류 CART {name} {complexity}{star}"
    raise ValueError(f"알 수 없는 CART 정책: {policy}")


def load_rows() -> pd.DataFrame:
    z = np.load(V11_CUBE, allow_pickle=True)
    regions = [str(x) for x in z["regions"]]
    seeds = [int(x) for x in z["seeds"]]
    methods = [str(x) for x in z["methods"]]
    values = np.asarray(z["pdr_wog"], dtype=float)

    if len(regions) != 250 or seeds != list(range(30)):
        raise ValueError(
            f"v11 평가격자 불일치: regions={len(regions)}, seeds={seeds}"
        )
    if values.shape != (250, len(methods), 30):
        raise ValueError(f"v11 배열 형상 불일치: {values.shape}")

    cubes: dict[str, tuple[np.ndarray, str, str]] = {}
    for method, (label, family) in V11_METHODS.items():
        if method not in methods:
            raise ValueError(f"v11 정책 누락: {method}")
        cubes[method] = (
            values[:, methods.index(method), :],
            label,
            family,
        )

    with np.load(LB_T_SWEEP, allow_pickle=False) as lb:
        lb_regions = [str(x) for x in lb["regions"]]
        lb_names = [str(x) for x in lb["names"]]
        lb_seeds = [int(x) for x in lb["seeds"]]
        lb_pdr = np.asarray(lb["pdr"], dtype=float)
    if lb_regions != regions:
        raise ValueError("LB-T3와 v11 scoreboard의 대표점 순서가 다릅니다")
    if lb_seeds != list(range(1000)) or lb_pdr.shape != (250, 39, 1000):
        raise ValueError(
            f"LB-T3 평가격자 불일치: seeds={lb_seeds[:3]}..{lb_seeds[-3:]}, "
            f"shape={lb_pdr.shape}"
        )
    cubes["LB_T3"] = (
        lb_pdr[:, lb_names.index("lb_T3"), :30],
        "LB-T3",
        "규칙 기준선",
    )

    cubes["SHIN_EVAL_ORACLE_BEST16"] = (
        load_shin_best_cube(regions, seeds),
        "Shin Best-of-16 (지역별 사후 최적)",
        "문헌 휴리스틱",
    )

    tree_policies = sorted(pd.read_csv(TREE_RAW, usecols=["policy"])["policy"].unique())
    if len(tree_policies) != 16:
        raise ValueError(f"CART 정책수 불일치: {len(tree_policies)}")
    for policy in tree_policies:
        cubes[policy] = (
            read_raw_cube(TREE_RAW, policy, regions, seeds),
            tree_label(policy),
            info_family(policy),
        )

    for policy, label in STUDENT_METHODS.items():
        cubes[policy] = (
            read_raw_cube(STUDENT_RAW, policy, regions, seeds),
            label,
            info_family(policy),
        )

    ebm_policy = "I3_CONNECTED_EBM_I08"
    cubes[ebm_policy] = (
        read_raw_cube(EBM_RAW, ebm_policy, regions, seeds),
        "증류 EBM I3 병원연계 I8",
        "I3",
    )

    if len(cubes) != 29:
        raise ValueError(f"통합 정책수 불일치: {len(cubes)}")

    rows = []
    for method, (cube, label, family) in cubes.items():
        if cube.shape != (250, 30):
            raise ValueError(f"{method} 배열 형상 불일치: {cube.shape}")
        regional_means = cube.mean(axis=1)
        rows.append(
            {
                "method": method,
                "label": label,
                "family": family,
                "n_regions": cube.shape[0],
                "n_seeds": cube.shape[1],
                "n_episodes": cube.size,
                "pdr_wog_mean": float(regional_means.mean()),
                "pdr_wog_ci95_regions": ci95(regional_means),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["pdr_wog_mean", "label"], ignore_index=True
    )


def render(scoreboard: pd.DataFrame, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "NanumGothic",
            "axes.unicode_minus": False,
            "font.size": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#383838",
            "axes.labelcolor": "#383838",
            "xtick.color": "#505050",
            "ytick.color": "#505050",
        }
    )

    height = max(15.5, 0.53 * len(scoreboard) + 2.5)
    fig, ax = plt.subplots(figsize=(17.5, height))
    y = np.arange(len(scoreboard))
    means = scoreboard["pdr_wog_mean"].to_numpy()
    cis = scoreboard["pdr_wog_ci95_regions"].to_numpy()
    colors = [COLORS[x] for x in scoreboard["family"]]

    ax.barh(
        y,
        means,
        height=0.78,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.6,
        zorder=2,
    )
    ax.errorbar(
        means,
        y,
        xerr=cis,
        fmt="none",
        ecolor="#4D4D4D",
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.2,
        zorder=3,
    )

    for yi, (mean, ci, family) in enumerate(
        zip(means, cis, scoreboard["family"], strict=True)
    ):
        ax.text(
            mean + ci + 0.0030,
            yi,
            f"{mean:.4f}",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold" if family == "최종 RL–OR" else "normal",
            color="#444444",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(scoreboard["label"], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.31)
    ax.set_xticks(np.arange(0.0, 0.31, 0.05))
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)", fontsize=13)
    ax.set_title(
        "공통 seed 0–29 전체 통합 Scoreboard: 기준·RL/OR·증류·Shin 정책 29개",
        fontsize=19,
        pad=34,
    )
    ax.text(
        0.5,
        1.012,
        "오차막대=250개 지역 평균의 95% CI · 정책당 7,500 paired episodes · ★=정책군 대표",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#666666",
    )
    ax.text(
        0.5,
        -0.055,
        "Shin Best=대표점별 사후 Best-of-16(선정 seed 0–999, 표시 seed 0–29) · "
        "CART=단일 나무 · GBDT/EBM=부스팅 앙상블 · 천리안 상한 제외",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#666666",
    )

    ax.xaxis.grid(True, color="#E1E4E8", linewidth=0.9, zorder=0)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    legend_items = [
        ("최종 RL–OR", "최종 RL–OR"),
        ("NCRP", "NCRP"),
        ("PPO", "PPO"),
        ("MILP", "MILP"),
        ("규칙 기준선", "규칙 기준선"),
        ("문헌 휴리스틱", "Shin 문헌 휴리스틱"),
        ("I0", "증류 I0"),
        ("I1", "증류 I1"),
        ("I2", "증류 I2"),
        ("I3", "증류 I3"),
    ]
    handles = [
        Patch(facecolor=COLORS[key], edgecolor="none", label=label)
        for key, label in legend_items
    ]
    ax.legend(
        handles=handles,
        ncol=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.092),
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.2,
    )

    fig.subplots_adjust(left=0.305, right=0.965, top=0.91, bottom=0.13)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scoreboard = load_rows()
    csv_path = OUT_DIR / "v11_full_scoreboard_29_with_shin.csv"
    scoreboard.to_csv(csv_path, index=False, encoding="utf-8-sig")
    render(scoreboard, OUT_DIR / "v11_full_scoreboard_29_with_shin")

    print(scoreboard[["label", "pdr_wog_mean", "pdr_wog_ci95_regions"]].to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Plot: {OUT_DIR / 'v11_full_scoreboard_29_with_shin.png'}")


if __name__ == "__main__":
    main()
