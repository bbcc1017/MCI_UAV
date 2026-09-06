# -*- coding: utf-8 -*-
"""v20 임계값 일반화 — 그림 4장.

1) 응답곡선: 네 규칙 형태의 lambda 축 U 자 (base)
2) 스케일링: 축별 최적 lambda 의 이동 (용량축은 이론선 1/c 함께)
3) 전이 후회: base 최적 상수를 27개 다른 물리조건에 그대로 가져갔을 때의 손해
4) 형태별 성능: 28개 조건 전부에서 현행 CARD 대비 paired 개선

실행: python tools/v20_threshold_plots.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import cube, paired, AXIS_BASE, SEED_NOISE  # noqa: E402

OUT = REPO / "results/scoreboard/v20/plots"
SWEEP = REPO / "results/scoreboard/v20/sweep"
INK, SUB = "#1F2933", "#6B7683"
COL = {"K": "#8C96A0", "T": "#5B8DEF", "L": "#F2994A", "H": "#27AE60", "Q": "#C0392B"}
NAME = {"K": "K  거리(km) + lam·부하        [현행 CARD]",
        "T": "T  시간(분) + lam·부하",
        "L": "L  거리(km) + lam·대기초과분",
        "H": "H  시간(분) + lam·대기초과분",
        "Q": "Q  시간(분) + lam·대기초과분/서버수  [완전 유도형]"}


def setup():
    fp = Path.home() / ".fonts/NanumGothic-Regular.ttf"
    if fp.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": INK,
                         "axes.edgecolor": "#D5DAE0", "savefig.bbox": "tight", "savefig.dpi": 170})


def fig1(d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, tag, ttl in zip(axes, ["base", "capa05", "capa20"],
                            ["기준 물리 (환자100·AMB30·UAV26·50/200km/h)",
                             "병원 용량 ×0.5 (수술실 c_bar 2.16→1.30)",
                             "병원 용량 ×2.0 (수술실 c_bar 2.16→4.32)"]):
        v = d.get(tag, {})
        for fam in ("K", "T", "L", "H", "Q"):
            if fam not in v:
                continue
            g = np.array(v[fam]["grid"])
            ax.plot(g[:, 0], g[:, 1], "o-", ms=3.5, lw=1.7, color=COL[fam],
                    label=NAME[fam] if tag == "base" else None)
            k = int(np.argmin(g[:, 1]))
            ax.plot(g[k, 0], g[k, 1], "*", ms=13, color=COL[fam], zorder=5)
        ax.set_xscale("log"); ax.set_xlabel("부하 교환율 lam  (km/명 또는 분/명)")
        ax.set_title(ttl, fontsize=10.5, color=INK)
        ax.grid(alpha=.25, lw=.6)
    axes[0].set_ylabel("PDR_woG  (낮을수록 좋음)")
    axes[0].legend(fontsize=8.5, framealpha=.95, loc="upper center")
    fig.suptitle("(1) 임계값 응답곡선 — 규칙 형태 다섯 가족, 250 시군구 × 10 시드 전수",
                 fontsize=12.5, y=1.02)
    fig.savefig(OUT / "01_응답곡선.png"); plt.close(fig)


def fig2(d, ceff):
    axes_list = ["vamb", "vuav", "capa", "amb", "uav", "n"]
    lbl = {"vamb": "AMB 속도 (km/h)", "vuav": "UAV 속도 (km/h)", "capa": "병원 용량 배율",
           "amb": "AMB 대수", "uav": "UAV 대수", "n": "환자 수"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, axis in zip(axes.ravel(), axes_list):
        for fam in ("K", "H", "Q"):
            pts = [(AXIS_BASE[axis], d["base"][fam]["lam"], d["base"][fam]["plateau"])] if fam in d["base"] else []
            for t, v in d.items():
                if v.get("axis") == axis and fam in v:
                    pts.append((v["axis_value"], v[fam]["lam"], v[fam]["plateau"]))
            if len(pts) < 2:
                continue
            pts.sort()
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            lo = [p[1] - p[2][0] for p in pts]; hi = [p[2][1] - p[1] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", ms=4, lw=1.6, capsize=3,
                        color=COL[fam], label=fam, alpha=.9)
        if axis == "capa":       # 이론선 lam ∝ 1/c_bar
            s = np.array(sorted(ceff)); cb = np.array([ceff[k].mean() for k in s])
            ax.plot(s, d["base"]["H"]["lam"] * ceff[1.0].mean() / cb, "k--", lw=1.3,
                    label="이론 lam ∝ 1/c_bar")
        ax.axvline(AXIS_BASE[axis], color="#D5DAE0", lw=1, zorder=0)
        ax.set_xlabel(lbl[axis]); ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("최적 lam"); axes[1, 0].set_ylabel("최적 lam")
    fig.suptitle("(2) 최적 임계값의 스케일링 — 세로막대 = 시드 잡음바닥 안쪽 평탄대", fontsize=12.5, y=.995)
    fig.tight_layout(); fig.savefig(OUT / "02_스케일링.png"); plt.close(fig)


def fig3(t):
    fams = [f for f in ("K", "T", "L", "H", "Q") if f in set(t.family)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    order = sorted(t.setting.unique())
    w = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        sub = t[t.family == fam].set_index("setting").reindex(order)
        ax.bar(np.arange(len(order)) + i * w, sub.regret * 1000, w,
               color=COL[fam], label=NAME[fam].split("[")[0].strip(), alpha=.92)
    ax.axhline(SEED_NOISE * 1000, color="#C0392B", ls="--", lw=1.2)
    ax.text(len(order) - .5, SEED_NOISE * 1000 * 1.06, "학습 시드 잡음바닥", color="#C0392B",
            fontsize=8.5, ha="right")
    ax.set_xticks(np.arange(len(order)) + .4); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("전이 후회 (PDR x1e-3)"); ax.grid(alpha=.25, lw=.6, axis="y")
    ax.legend(fontsize=8.5, ncol=len(fams))
    ax.set_title("(3) 기준 물리에서 고른 상수를 그대로 다른 조건에 가져갔을 때의 손해 (낮을수록 일반화)",
                 fontsize=12)
    fig.savefig(OUT / "03_전이후회.png"); plt.close(fig)


def fig4(rows):
    df = pd.DataFrame(rows).set_index("setting").sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(df))
    for i, fam in enumerate([f for f in ("Q", "H", "L", "T") if f + "_d" in df]):
        ax.errorbar(df[fam + "_d"] * 1000, y + (i - 1.5) * .18, xerr=df[fam + "_ci"] * 1000,
                    fmt="o", ms=4, lw=1.3, color=COL[fam], label=NAME[fam].split("[")[0].strip())
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("현행 CARD 대비 PDR 개선 (x1e-3, 오른쪽이 좋음) — 95% CI")
    ax.grid(alpha=.25, lw=.6, axis="x"); ax.legend(fontsize=8.5)
    ax.set_title("(4) 28개 물리조건 전부에서의 형태별 개선 (같은 좌표·같은 시드 paired)", fontsize=12)
    fig.savefig(OUT / "04_형태별개선.png"); plt.close(fig)


def main():
    setup(); OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads((REPO / "results/scoreboard/v20/optima.json").read_text(encoding="utf-8"))
    t = pd.read_csv(REPO / "results/scoreboard/v20/transfer.csv")

    import random
    man = json.loads((REPO / "scenarios/manifests/v19/tradeoff250_manifest.json").read_text())
    random.seed(0)
    c0 = np.concatenate([pd.read_csv(
        (man[k] if isinstance(man[k], str) else man[k]["path"]).rsplit("/", 1)[0] + "/hospital_info.csv"
    )["수술실수"].values for k in random.sample(list(man), 60)]).astype(float)
    ceff = {s: np.maximum(1, np.round(c0 * s)) for s in (0.5, 0.75, 1.0, 1.5, 2.0)}

    BASE = {"K": ("lam_", "K12"), "T": ("lam_", "T9"), "H": ("lam_", "H12"), "L": ("lamL_", "L14")}
    qb = d["base"].get("Q")
    if qb:
        BASE["Q"] = ("lamQ_", f"Q{qb['lam']:g}")
    rows = []
    for tag in sorted({os.path.basename(f).split("_", 1)[1][:-4]
                       for f in glob.glob(str(SWEEP / "lam_*.csv"))}):
        cs = {}
        for fam, (pre, pol) in BASE.items():
            f = SWEEP / f"{pre}{tag}.csv"
            if not (f.parent / (f.name + ".meta.json")).exists():
                continue
            try:
                cs[fam] = cube(pd.read_csv(f), pol)
            except ValueError:
                pass
        if "K" not in cs:
            continue
        r = {"setting": tag}
        for fam in cs:
            if fam == "K":
                continue
            p = paired(cs[fam], cs["K"])
            r[fam + "_d"], r[fam + "_ci"] = p["delta"], p["ci95"]
        rows.append(r)

    fig1(d); fig2(d, ceff); fig3(t); fig4(rows)
    print(f"[plots] 4장 → {OUT}")


if __name__ == "__main__":
    main()
