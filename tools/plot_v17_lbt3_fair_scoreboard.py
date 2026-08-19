# -*- coding: utf-8 -*-
"""v17 병원규칙 보정 후 PPO·휴리스틱·LB-T3·Shin 공통30 scoreboard.

PPO가 대표점 250개에서 seed 0..29로 평가되어 있으므로, 1,000회 전수평가
산출물도 같은 seed 0..29만 표시한다. 규칙 선정은 전수 1,000회(또는 학습
random4 1,000좌표)로 먼저 고정하고, 표시 구간을 선정에 다시 쓰지 않는다.
트리·NCRP는 의도적으로 포함하지 않는다.
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
V17 = REPO / "results/scoreboard/v17"
HEUR_ROOT = V17 / "heur64_t4_hospital_rule_fix_full1000"
LB_ROOT = V17 / "lb3_shin_hospital_rule_fix_full1000"
PPO_CUBE = REPO / "results/scoreboard/v11/eval250/scoreboard_episodes.npz"
SHIN_ORIGINAL = REPO / "results/scoreboard/v10/shin16_full1000/work/shin/eval250"
OUT = V17 / "lbt3_common30_scoreboard"

N_SHOW = 30
METRIC = 3  # PDR_woG


def ci95_region(cube: np.ndarray) -> float:
    means = np.asarray(cube, dtype=float).mean(axis=1)
    return float(1.96 * means.std(ddof=1) / math.sqrt(means.size))


def summarize(label: str, family: str, cube: np.ndarray, selection: str) -> dict:
    if cube.shape != (250, N_SHOW):
        raise ValueError(f"{label}: cube shape {cube.shape}")
    region_means = cube.mean(axis=1)
    return {
        "label": label,
        "family": family,
        "pdr_wog_mean": float(region_means.mean()),
        "pdr_wog_ci95_region": ci95_region(cube),
        "n_regions": 250,
        "n_seeds": N_SHOW,
        "n_episodes": int(cube.size),
        "selection": selection,
    }


def npz_map(root: Path) -> dict[str, Path]:
    files = {p.stem: p for p in root.glob("*.npz")}
    if len(files) != 250:
        raise ValueError(f"{root}: eval NPZ {len(files)} != 250")
    return files


def posthoc_best_cube(files: dict[str, Path]) -> np.ndarray:
    rows = []
    for key in sorted(files):
        with np.load(files[key], allow_pickle=False) as z:
            values = np.asarray(z["values"], dtype=float)
            seeds = np.asarray(z["seeds"], dtype=int)
        if not np.array_equal(seeds, np.arange(1000)):
            raise ValueError(f"seed mismatch: {files[key]}")
        best = int(np.argmin(values[:, :, METRIC].mean(axis=1)))
        rows.append(values[best, :N_SHOW, METRIC])
    return np.stack(rows)


def t4_cube(files: dict[str, Path]) -> np.ndarray:
    rows = []
    for key in sorted(files):
        with np.load(files[key], allow_pickle=False) as z:
            values = np.asarray(z["values"], dtype=float)
            seeds = np.asarray(z["seeds"], dtype=int)
        if values.shape != (1000, 5) or not np.array_equal(seeds, np.arange(1000)):
            raise ValueError(f"T4 checkpoint mismatch: {files[key]}")
        rows.append(values[:N_SHOW, METRIC])
    return np.stack(rows)


def load_lb_assets() -> tuple[dict[str, Path], dict[str, Path], pd.DataFrame]:
    train = {
        p.stem: p for p in (LB_ROOT / "work/lb3/train1000").glob("*.npz")
    }
    eval_ = npz_map(LB_ROOT / "work/lb3/eval250")
    if len(train) != 1000:
        raise ValueError(f"LB train NPZ {len(train)} != 1000")
    meta = pd.read_csv(LB_ROOT / "lb3_full64_summary.csv", dtype={"sigcd": str})
    return train, eval_, meta


def lb_fixed_and_regional_cubes(
    train: dict[str, Path], eval_: dict[str, Path], meta: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_meta = meta[meta.dataset == "train1000"][
        ["coordinate_key", "sigcd"]
    ].drop_duplicates()
    eval_meta = meta[meta.dataset == "eval250"][
        ["coordinate_key", "sigcd"]
    ].drop_duplicates()
    if len(train_meta) != 1000 or len(eval_meta) != 250:
        raise ValueError("LB metadata coordinate grain mismatch")

    # 정책명/순서는 모든 좌표에서 같아야 한다.
    probe = next(iter(train.values()))
    with np.load(probe, allow_pickle=False) as z:
        names = [str(x) for x in z["policy_names"]]
    if len(names) != 65 or names[0] != "LB3_AGNOSTIC_RR_FASTEST":
        raise ValueError("LB policy layout mismatch")
    cap_idx = np.arange(1, 65)
    start_idx = np.array(
        [i for i, n in enumerate(names) if n.startswith("LB3_CAP_F64 | START,")],
        dtype=int,
    )
    if len(start_idx) != 32:
        raise ValueError(f"START candidate count {len(start_idx)} != 32")

    # 전국 단일 규칙: 학습 1,000좌표 전체 평균으로 한 번만 선택.
    nationwide_sum = np.zeros(65, dtype=float)
    for key in train_meta.coordinate_key:
        with np.load(train[key], allow_pickle=False) as z:
            nationwide_sum += np.asarray(z["values"], dtype=float)[:, :, METRIC].mean(axis=1)
    nationwide_mean = nationwide_sum / len(train_meta)
    fixed_all = int(cap_idx[np.argmin(nationwide_mean[cap_idx])])
    fixed_start = int(start_idx[np.argmin(nationwide_mean[start_idx])])

    # 지역별 규칙: 해당 시군구의 random4 네 좌표만으로 선택.
    regional_all: dict[str, int] = {}
    regional_start: dict[str, int] = {}
    for sigcd, group in train_meta.groupby("sigcd"):
        if len(group) != 4:
            raise ValueError(f"{sigcd}: train points {len(group)} != 4")
        score = np.zeros(65, dtype=float)
        for key in group.coordinate_key:
            with np.load(train[key], allow_pickle=False) as z:
                score += np.asarray(z["values"], dtype=float)[:, :, METRIC].mean(axis=1)
        score /= len(group)
        regional_all[sigcd] = int(cap_idx[np.argmin(score[cap_idx])])
        regional_start[sigcd] = int(start_idx[np.argmin(score[start_idx])])

    cubes = {
        "agnostic": [],
        "fixed_start": [],
        "fixed_all": [],
        "regional_start": [],
        "regional_all": [],
    }
    for row in eval_meta.sort_values("coordinate_key").itertuples(index=False):
        with np.load(eval_[row.coordinate_key], allow_pickle=False) as z:
            v = np.asarray(z["values"], dtype=float)
        cubes["agnostic"].append(v[0, :N_SHOW, METRIC])
        cubes["fixed_start"].append(v[fixed_start, :N_SHOW, METRIC])
        cubes["fixed_all"].append(v[fixed_all, :N_SHOW, METRIC])
        cubes["regional_start"].append(v[regional_start[row.sigcd], :N_SHOW, METRIC])
        cubes["regional_all"].append(v[regional_all[row.sigcd], :N_SHOW, METRIC])

    out = {k: np.stack(v) for k, v in cubes.items()}
    audit = {
        "fixed_start_row": fixed_start,
        "fixed_start_policy": names[fixed_start],
        "fixed_all_row": fixed_all,
        "fixed_all_policy": names[fixed_all],
        "regional_sigcd_count": len(regional_all),
    }
    return out, audit


def lb_heurbest_cap_cube(
    heur_files: dict[str, Path], lb_files: dict[str, Path]
) -> np.ndarray:
    rows = []
    if set(heur_files) != set(lb_files):
        raise ValueError("HEUR/LB eval coordinate mismatch")
    for key in sorted(heur_files):
        with np.load(heur_files[key], allow_pickle=False) as h:
            hv = np.asarray(h["values"], dtype=float)
        best = int(np.argmin(hv[:, :, METRIC].mean(axis=1)))
        with np.load(lb_files[key], allow_pickle=False) as l:
            lv = np.asarray(l["values"], dtype=float)
        rows.append(lv[best + 1, :N_SHOW, METRIC])
    return np.stack(rows)


def ppo_cube() -> np.ndarray:
    with np.load(PPO_CUBE, allow_pickle=True) as z:
        methods = [str(x) for x in z["methods"]]
        seeds = [int(x) for x in z["seeds"]]
        values = np.asarray(z["pdr_wog"], dtype=float)
    if seeds != list(range(N_SHOW)):
        raise ValueError(f"PPO seed mismatch: {seeds}")
    cube = values[:, methods.index("PPO_POINTER_V10"), :]
    if cube.shape != (250, N_SHOW):
        raise ValueError(f"PPO cube shape: {cube.shape}")
    return cube


def render(score: pd.DataFrame) -> None:
    font_path = Path.home() / ".fonts/NanumGothic-Regular.ttf"
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "NanumGothic",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#30343A",
            "axes.labelcolor": "#30343A",
            "xtick.color": "#606870",
            "ytick.color": "#30343A",
        }
    )

    palette = {
        "PPO": "#245B86",
        "LB-T3 공정선정": "#E47A26",
        "LB-T3 기타": "#E8A45A",
        "기준 휴리스틱": "#A6A8AB",
        "Shin": "#9A8034",
    }
    score = score.sort_values(["pdr_wog_mean", "label"], ignore_index=True)
    y = np.arange(len(score))
    means = score.pdr_wog_mean.to_numpy(float)
    cis = score.pdr_wog_ci95_region.to_numpy(float)
    colors = [palette[x] for x in score.family]

    fig_h = 0.62 * len(score) + 3.2
    fig, ax = plt.subplots(figsize=(16.5, fig_h), dpi=180)
    ax.barh(y, means, color=colors, height=0.72, edgecolor="white", linewidth=0.7, zorder=2)
    ax.errorbar(
        means,
        y,
        xerr=cis,
        fmt="none",
        ecolor="#4C5258",
        elinewidth=1.25,
        capsize=3.2,
        capthick=1.25,
        zorder=3,
    )
    for yi, (mean, ci) in enumerate(zip(means, cis, strict=True)):
        ax.text(mean + ci + 0.003, yi, f"{mean:.4f}", va="center", ha="left", fontsize=11, color="#43484D")

    ax.set_yticks(y, score.label, fontsize=12.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.29)
    ax.set_xticks(np.arange(0, 0.30, 0.05))
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)", fontsize=13.5, labelpad=12)
    ax.set_title(
        "동일 평가조건 종합 Scoreboard: PPO · 휴리스틱 · LB-T3 · Shin",
        fontsize=21,
        pad=42,
    )
    ax.text(
        0.5,
        1.015,
        "대표점 250개 × 공통 seed 0–29 · 정책당 7,500 paired episodes · 오차막대=지역평균 95% CI",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#68717A",
    )
    ax.text(
        0.5,
        -0.115,
        "† 대표점 결과를 이용한 좌표별 사후선택 정책(공정한 고정정책이 아닌 참고 기준) · 지역별 LB3는 각 시군구 random4 학습좌표에서 선택 후 대표점에 동결 적용",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color="#68717A",
    )
    ax.xaxis.grid(True, color="#E2E6E9", linewidth=0.9, zorder=0)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#757B81")
    ax.spines["bottom"].set_color("#757B81")

    legend_order = ["PPO", "LB-T3 공정선정", "LB-T3 기타", "기준 휴리스틱", "Shin"]
    handles = [Patch(facecolor=palette[k], edgecolor="none", label=k) for k in legend_order]
    ax.legend(
        handles=handles,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.175),
        fontsize=10.5,
        columnspacing=1.8,
    )
    plt.subplots_adjust(left=0.29, right=0.95, top=0.84, bottom=0.27)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    heur_files = npz_map(HEUR_ROOT / "work/heur/eval250")
    t4_files = npz_map(HEUR_ROOT / "work/t4/eval250")
    train_lb, eval_lb, lb_meta = load_lb_assets()
    lb_cubes, selection_audit = lb_fixed_and_regional_cubes(train_lb, eval_lb, lb_meta)

    shin_align_files = npz_map(LB_ROOT / "work/shin_align/eval250")
    shin_original_files = npz_map(SHIN_ORIGINAL)

    rows = [
        summarize("PPO Pointer v10", "PPO", ppo_cube(), "학습 random4 1,000 → 대표점 고정평가"),
        summarize("Full64-LB3 지역별 선택", "LB-T3 공정선정", lb_cubes["regional_all"], "시군구별 random4 4점 선택"),
        summarize("START-LB3 지역별 선택", "LB-T3 공정선정", lb_cubes["regional_start"], "시군구별 random4 4점 선택"),
        summarize("Full64-LB3 전국 단일", "LB-T3 공정선정", lb_cubes["fixed_all"], "학습 1,000좌표 전국 단일 선택"),
        summarize("LB3-AGN 기본형", "LB-T3 기타", lb_cubes["agnostic"], "사전 고정규칙"),
        summarize("START-LB3 전국 단일", "LB-T3 공정선정", lb_cubes["fixed_start"], "학습 1,000좌표 전국 단일 선택"),
        summarize("LB3 HEUR-best→cap3† (지난주 정정)", "LB-T3 기타", lb_heurbest_cap_cube(heur_files, eval_lb), "대표점별 HEUR64 best 사후선택"),
        summarize("LB-T4 HEUR-best→cap4†", "기준 휴리스틱", t4_cube(t4_files), "대표점별 HEUR64 best 사후선택"),
        summarize("Shin 원문형 Best-of-16†", "Shin", posthoc_best_cube(shin_original_files), "대표점별 사후선택"),
        summarize("HEUR64 Best-of-64†", "기준 휴리스틱", posthoc_best_cube(heur_files), "대표점별 사후선택"),
        summarize("Shin 정합형 Best-of-16†", "Shin", posthoc_best_cube(shin_align_files), "대표점별 사후선택"),
    ]
    score = pd.DataFrame(rows).sort_values(["pdr_wog_mean", "label"], ignore_index=True)
    score.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    pd.Series(selection_audit).to_json(
        OUT.with_name(OUT.name + "_selection_audit.json"),
        force_ascii=False,
        indent=2,
    )
    render(score)
    print(score[["label", "pdr_wog_mean", "pdr_wog_ci95_region", "selection"]].to_string(index=False))
    print(f"\nPNG: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
