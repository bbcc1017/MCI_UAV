# -*- coding: utf-8 -*-
"""v12 LB-T 1,000회 전수 스윕 논문·발표용 그림 2종.

입력
  results/scoreboard/v12/lbT_sweep/lbT_eval250_1000ep_pe.npz

출력
  results/scoreboard/v12/lbT_sweep/plots/
    lbT_national_sweep_eval250_1000ep.{png,pdf,svg}
    lbT_sido_sweep_eval250_1000ep.{png,pdf,svg}
    lbT_sido_summary_eval250_1000ep.csv

두 그림 모두 대표점 250개, seed 0..999, PDR_woG(낮을수록 우수)를 사용한다.
광역시도 집계는 시군구 동일가중 평균이며 지역별 최적 T는 설명용 사후 최적값이다.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "results/scoreboard/v12/lbT_sweep"
DEFAULT_PE = DEFAULT_DIR / "lbT_eval250_1000ep_pe.npz"
DEFAULT_OUT = DEFAULT_DIR / "plots"

SIDO_ORDER = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
SIDO_BY_PREFIX = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천",
    "29": "광주", "30": "대전", "31": "울산", "36": "세종",
    "41": "경기", "51": "강원", "43": "충북", "44": "충남",
    "45": "전북", "46": "전남", "47": "경북", "48": "경남",
    "50": "제주",
}

BLUE = "#236A9A"
BLUE_LIGHT = "#D8EAF3"
RED = "#C83B32"
INK = "#20262D"
MUTED = "#65717C"
GRID = "#D9E0E6"
GREEN = "#36836A"
WHITE = "#FFFFFF"


def _set_style() -> None:
    """서버에 설치된 한글 폰트를 사용하고 두 그림의 시각 규격을 통일한다."""
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((x for x in candidates if x in installed), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.edgecolor": "#8C99A4",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.titleweight": "regular",
        "savefig.facecolor": WHITE,
        "savefig.bbox": "tight",
    })


def _load(pe_path: Path):
    z = np.load(pe_path, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    regions = [str(x) for x in z["regions"]]
    pdr = np.asarray(z["pdr"], dtype=np.float64)
    seeds = np.asarray(z["seeds"], dtype=np.int64) if "seeds" in z.files else np.arange(pdr.shape[2])

    pairs = []
    for i, name in enumerate(names):
        m = re.fullmatch(r"lb_T(\d+)", name)
        if m:
            pairs.append((int(m.group(1)), i))
    pairs.sort()
    Ts = np.asarray([x[0] for x in pairs], dtype=int)
    idx = np.asarray([x[1] for x in pairs], dtype=int)
    pdr = pdr[:, idx, :]

    if pdr.shape != (250, 39, 1000):
        raise ValueError(f"최신 정본 형상 불일치: {pdr.shape}, 기대=(250,39,1000)")
    if not np.array_equal(Ts, np.arange(2, 41)):
        raise ValueError(f"T 격자 불일치: {Ts.tolist()}")
    if not np.array_equal(seeds, np.arange(1000)):
        raise ValueError(f"seed 불일치: {seeds[:3]}..{seeds[-3:]}")
    return Ts, regions, pdr


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))


def plot_national(Ts: np.ndarray, pdr: np.ndarray, out_dir: Path) -> Path:
    """전국 T-PDR 곡선과 최적구간 확대."""
    region_mean = pdr.mean(axis=2)
    mean = region_mean.mean(axis=0)
    ci = 1.96 * region_mean.std(axis=0, ddof=1) / math.sqrt(region_mean.shape[0])
    best_i = int(np.argmin(mean))
    best_t = int(Ts[best_i])
    t4_i = int(np.flatnonzero(Ts == 4)[0])

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[3.25, 1.15],
        left=0.07, right=0.97, bottom=0.12, top=0.76, wspace=0.12,
    )
    ax = fig.add_subplot(gs[0, 0])
    zoom = fig.add_subplot(gs[0, 1])

    fig.suptitle("LB-T 발송상한 임계값 전수 스윕", fontsize=28, y=0.95)
    fig.text(
        0.5, 0.89,
        "T=2–40 · 대표점 250개 × 1,000회 · 공통 seed 0–999 · PDR_woG",
        ha="center", fontsize=15, color=MUTED,
    )

    ax.fill_between(Ts, mean - ci, mean + ci, color=BLUE_LIGHT, alpha=0.92,
                    label="지역평균 95% CI", linewidth=0)
    ax.plot(
        Ts, mean, color=BLUE, lw=3.1, marker="o", ms=5.0,
        markerfacecolor=WHITE, markeredgewidth=1.5, label="LB-T 전국 평균",
    )
    ax.scatter([best_t], [mean[best_i]], marker="*", s=260, color=RED,
               edgecolor=WHITE, linewidth=1.1, zorder=5, label=f"전국 최적 T={best_t}")

    # 1,000회 자료에서 T34~40이 에피소드 단위로 정확히 동일하다.
    sat_t = 34
    ax.axvspan(sat_t - 0.45, 40.5, color="#EEF1F4", alpha=0.9, zorder=-2)
    ax.axvline(sat_t, color="#7A8792", lw=2.0, ls=(0, (6, 5)))
    ax.annotate(
        f"T≥{sat_t}: T=40과 동일\nPDR {mean[-1]:.6f}",
        xy=(36.0, mean[-1]), xytext=(29.8, 0.218),
        arrowprops=dict(arrowstyle="-|>", color="#7A8792", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.45", fc=WHITE, ec="#CBD3DA"),
        fontsize=12, color=MUTED,
    )
    ax.annotate(
        f"전국 최적  T={best_t}\nPDR {mean[best_i]:.6f}",
        xy=(best_t, mean[best_i]), xytext=(7.0, 0.153),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.45", fc=WHITE, ec="#E0B2AE"),
        fontsize=13, color=INK,
    )
    ax.set_xlabel("병원당 발송상한 T", fontsize=15, labelpad=12)
    ax.set_ylabel("전국 평균 PDR_woG  (낮을수록 우수)", fontsize=15, labelpad=14)
    ax.set_xlim(1.5, 40.5)
    ax.set_ylim(0.145, 0.263)
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 34, 40])
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0, 2]
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper center", bbox_to_anchor=(0.5, 0.84),
        frameon=False, ncol=3, fontsize=12.5, handlelength=2.6,
    )

    # 최적구간은 좁은 PDR 범위를 별도 패널에서 확대한다.
    zmask = (Ts >= 2) & (Ts <= 7)
    zoom.fill_between(Ts[zmask], (mean - ci)[zmask], (mean + ci)[zmask],
                      color=BLUE_LIGHT, alpha=0.92, linewidth=0)
    zoom.plot(Ts[zmask], mean[zmask], color=BLUE, lw=3.1, marker="o", ms=6.3,
              markerfacecolor=WHITE, markeredgewidth=1.5)
    zoom.scatter([best_t], [mean[best_i]], marker="*", s=260, color=RED,
                 edgecolor=WHITE, linewidth=1.1, zorder=5)
    zoom.axvline(4, color="#7A8792", lw=1.2, ls="-")
    zoom.text(4, mean[t4_i] + 0.024, f"T=4  {mean[t4_i]:.6f}",
              ha="center", fontsize=11.5, color=MUTED)
    zoom.annotate(
        f"T={best_t}  {mean[best_i]:.6f}\n"
        f"T=4보다 {mean[t4_i] - mean[best_i]:.6f} 낮음",
        xy=(best_t, mean[best_i]), xytext=(4.35, 0.158),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4),
        fontsize=11.8, color=RED,
    )
    zoom.set_title("최적 구간 확대", fontsize=16, pad=14)
    zoom.set_xlabel("T", fontsize=14, labelpad=10)
    zoom.set_xlim(1.7, 7.3)
    zoom.set_ylim(0.154, 0.206)
    zoom.set_xticks(np.arange(2, 8))
    zoom.grid(color=GRID, lw=1.0)
    zoom.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.07, 0.045,
        "주: 음영은 250개 시군구 대표점 평균의 95% CI. T는 누적 발송 환자 수가 임계값에 "
        "도달하면 다음 적격 병원으로 분산하는 발송상한이다.",
        fontsize=10.5, color=MUTED,
    )
    stem = out_dir / "lbT_national_sweep_eval250_1000ep"
    _save(fig, stem)
    plt.close(fig)
    return stem.with_suffix(".png")


def _sido_rows(regions: list[str]) -> dict[str, list[int]]:
    out = {s: [] for s in SIDO_ORDER}
    for i, key in enumerate(regions):
        m = re.search(r"_(\d{5})$", key)
        if m is None or m.group(1)[:2] not in SIDO_BY_PREFIX:
            raise ValueError(f"시도 매핑 실패: {key}")
        out[SIDO_BY_PREFIX[m.group(1)[:2]]].append(i)
    if any(not rows for rows in out.values()):
        raise ValueError({s: len(rows) for s, rows in out.items()})
    return out


def plot_sido(Ts: np.ndarray, regions: list[str], pdr: np.ndarray, out_dir: Path) -> Path:
    """광역시도별 T 민감도 heatmap과 최적 T 요약표."""
    groups = _sido_rows(regions)
    sido_mean = np.stack([pdr[groups[s]].mean(axis=(0, 2)) for s in SIDO_ORDER])
    best_i = np.argmin(sido_mean, axis=1)
    best_t = Ts[best_i]
    best_pdr = sido_mean[np.arange(len(SIDO_ORDER)), best_i]
    t3_i = int(np.flatnonzero(Ts == 3)[0])
    pdr_t3 = sido_mean[:, t3_i]
    gain = pdr_t3 - best_pdr
    gain_pct = np.divide(gain, pdr_t3, out=np.zeros_like(gain), where=pdr_t3 > 0) * 100.0

    # 색은 각 시도의 최적 T 대비 상대 PDR 증가율. 긴 꼬리는 log1p로 압축한다.
    rel = (sido_mean - best_pdr[:, None]) / best_pdr[:, None] * 100.0
    cap = 200.0
    heat = np.log1p(np.clip(rel, 0.0, cap))

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_axes([0.085, 0.20, 0.70, 0.59])
    tab = fig.add_axes([0.80, 0.20, 0.18, 0.59])
    cax = fig.add_axes([0.14, 0.095, 0.77, 0.032])

    fig.suptitle("광역시도별 LB-T 발송상한 스윕", fontsize=28, y=0.96)
    fig.text(
        0.5, 0.905,
        "대표점 250개 · 시군구 동일가중 평균 · 좌표당 1,000회 · 공통 seed 0–999",
        ha="center", fontsize=15, color=MUTED,
    )
    fig.text(
        0.435, 0.855,
        "흰색 테두리 = 해당 광역시도의 최적 T  |  점선 = 전국 고정 최적 T=3",
        ha="center", fontsize=12.5, color=MUTED,
    )

    im = ax.imshow(heat, aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=np.log1p(cap),
                   interpolation="nearest")
    ax.axvline(t3_i, color="#263746", lw=2.0, ls=(0, (6, 5)), zorder=4)
    ax.text(t3_i, -0.87, "전국 기준 T=3", ha="center", va="bottom",
            fontsize=11.5, color="#263746")

    for r, j in enumerate(best_i):
        ax.add_patch(Rectangle(
            (j - 0.48, r - 0.48), 0.96, 0.96,
            fill=False, edgecolor=WHITE, lw=2.3, zorder=5,
        ))

    tick_T = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 34, 40]
    tick_pos = [int(np.flatnonzero(Ts == t)[0]) for t in tick_T]
    ax.set_xticks(tick_pos, tick_T, fontsize=11.5)
    ax.set_yticks(
        np.arange(len(SIDO_ORDER)),
        [f"{s}  ({len(groups[s])})" for s in SIDO_ORDER],
        fontsize=12.5,
    )
    ax.set_xlabel("병원당 발송상한 T", fontsize=15, labelpad=12)
    ax.set_ylabel("광역시도  (시군구 수)", fontsize=15, labelpad=14)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    # 우측의 정확값 표.
    tab.set_xlim(0, 1)
    tab.set_ylim(-0.9, len(SIDO_ORDER) - 0.3)
    tab.invert_yaxis()
    tab.axis("off")
    tab.text(0.10, -0.65, "최적 T", fontsize=12.5, ha="center")
    tab.text(0.48, -0.65, "평균 PDR", fontsize=12.5, ha="center")
    tab.text(0.86, -0.65, "T3 대비", fontsize=12.5, ha="center")
    for r in range(len(SIDO_ORDER)):
        tab.axhline(r + 0.50, color="#E1E6EA", lw=0.8, xmin=0.0, xmax=1.0)
        tab.text(0.10, r, f"{int(best_t[r])}", fontsize=11.5, ha="center", va="center")
        tab.text(0.48, r, f"{best_pdr[r]:.4f}", fontsize=11.5, ha="center", va="center",
                 color=MUTED)
        val = "기준" if int(best_t[r]) == 3 else f"{gain_pct[r]:.2f}%↓"
        tab.text(0.86, r, val, fontsize=11.5, ha="center", va="center",
                 color=MUTED if val == "기준" else GREEN)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb_ticks = np.asarray([0, 5, 10, 25, 50, 100, 200], dtype=float)
    cb.set_ticks(np.log1p(cb_ticks))
    cb.set_ticklabels(["0", "5", "10", "25", "50", "100", "200+"])
    cb.ax.tick_params(labelsize=10.5)
    cb.outline.set_edgecolor("#AAB4BD")
    cb.set_label(
        "해당 광역시도의 최적 T 대비 PDR 증가율 (%)  —  낮을수록 우수",
        fontsize=11.5, labelpad=9,
    )

    fig.text(
        0.085, 0.035,
        "주: 광역시도별 최적 T는 대표점 평가자료에서 사후 산출한 설명용 결과이며, "
        "독립적인 배포 정책 성능으로 해석하지 않는다.",
        fontsize=10.5, color=MUTED,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lbT_sido_summary_eval250_1000ep.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["sido", "n_sigungu", "best_T", "best_pdr_wog", "pdr_wog_T3",
                     "gain_vs_T3", "relative_reduction_vs_T3_pct"])
        for r, sido in enumerate(SIDO_ORDER):
            wr.writerow([
                sido, len(groups[sido]), int(best_t[r]),
                f"{best_pdr[r]:.9f}", f"{pdr_t3[r]:.9f}",
                f"{gain[r]:.9f}", f"{gain_pct[r]:.6f}",
            ])

    stem = out_dir / "lbT_sido_sweep_eval250_1000ep"
    _save(fig, stem)
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default=str(DEFAULT_PE))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    _set_style()
    Ts, regions, pdr = _load(Path(args.pe))
    out_dir = Path(args.out_dir)
    national = plot_national(Ts, pdr, out_dir)
    sido = plot_sido(Ts, regions, pdr, out_dir)
    print(f"저장: {national}")
    print(f"저장: {sido}")


if __name__ == "__main__":
    main()
