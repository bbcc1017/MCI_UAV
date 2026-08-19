# -*- coding: utf-8 -*-
"""v17 현장 규칙집 집계·그림 — 물리단위 규칙, ablation, 정보 진단."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
F = REPO / "results/scoreboard/v17/fieldrules"
D = REPO / "results/scoreboard/v17/distill"
INK, SUB = "#30343A", "#606870"


def setup():
    fp = Path.home() / ".fonts/NanumGothic-Regular.ttf"
    if fp.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": INK,
                         "axes.edgecolor": "#D5DAE0", "savefig.bbox": "tight",
                         "savefig.dpi": 170})


def ci(x):
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


def cube(path, pol):
    d = pd.read_csv(path)
    d = d[d.policy == pol]
    if d.empty:
        raise ValueError(f"{path}: {pol} 없음")
    p = d.pivot_table(index="region", columns="seed", values="pdr_woG").sort_index()
    if p.isna().any().any():
        raise ValueError(f"{path}/{pol}: 결측")
    return p.to_numpy(float)


def wtl(a, b):
    w = t = l = 0
    for i in range(a.shape[0]):
        dd = b[i] - a[i]
        m, c = dd.mean(), ci(dd)
        w += m > c
        l += m < -c
        t += (-c <= m <= c)
    return w, t, l


ARMS = [
    ("CARD 현장 규칙집", "card", "CARD", "규칙집"),
    ("CARD - UAV규칙 없음", "card", "CARD_NOUAV", "ablation"),
    ("CARD - Red우선으로 교체", "card", "CARD_REDFIRST", "ablation"),
    ("CARD 행동추정 파라미터", "card", "CARD_BEHAV", "규칙집"),
    ("PPO Pointer v10 (교사)", "ppo", "PPO_POINTER_V10", "RL"),
    ("AUG68 GBDT31 증류트리", "tree", "AUG68_G31", "증류"),
    ("Full64-LB3 전국단일", "card", "FULL64_LB3", "휴리스틱"),
    ("LB3-AGN 기본형", "card", "LB3_AGN", "휴리스틱"),
    ("START-LB3 전국단일", "card", "START_LB3", "휴리스틱"),
    ("CARD (PPO와 같은 정보)", "cardnorm", "CARD_PPOINFO", "정보진단"),
    ("CARD - 부하가중 없음(최근접만)", "card", "CARD_NOLOAD", "ablation"),
]

PATHS = {
    "eval250": {"card": F / "card_eval250_seed0_29.csv",
                "ppo": D / "ppo_eval250_seed0_29.csv",
                "tree": D / "tree_eval250_seed0_29.csv",
                "cardnorm": F / "cardnorm_eval250.csv"},
    "ext250": {"card": F / "card_ext250_seed10000.csv",
               "ppo": F / "ppo_ext250_seed10000.csv",
               "tree": F / "tree_ext250_seed10000.csv",
               "cardnorm": F / "cardnorm_ext250.csv"},
}


def build(setname):
    src = PATHS[setname]
    out, cubes = [], {}
    for label, kind, pol, fam in ARMS:
        p = src[kind]
        if kind == "card" and pol in ("START_LB3", "FULL64_LB3", "LB3_AGN") and setname == "eval250":
            p = D / "rule_eval250_seed0_29.csv"
        cubes[label] = cube(p, pol)
        out.append((label, fam))
    ref = "START-LB3 전국단일"
    base = cubes[ref]
    ppo = cubes["PPO Pointer v10 (교사)"]
    gap = base.mean(1).mean() - ppo.mean(1).mean()
    rows = []
    for label, fam in out:
        v = cubes[label]
        dd = base.mean(1) - v.mean(1)
        w, t, l = wtl(v, base)
        rows.append({"set": setname, "label": label, "family": fam,
                     "pdr_wog": v.mean(1).mean(), "ci95": ci(v.mean(1)),
                     "delta_vs_lb3": dd.mean(), "delta_ci95": ci(dd),
                     "win": w, "tie": t, "loss": l,
                     "gap_recovery_pct": 100 * dd.mean() / gap})
    return pd.DataFrame(rows), cubes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(F / "fieldcard_scoreboard.csv"))
    a = ap.parse_args()
    setup()
    frames, allc = [], {}
    for s in ("eval250", "ext250"):
        df, c = build(s)
        frames.append(df)
        allc[s] = c
    sb = pd.concat(frames)
    sb.to_csv(a.out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    for s in ("eval250", "ext250"):
        print(f"\n=== {s} ===")
        print(sb[sb.set == s].sort_values("pdr_wog").to_string(
            index=False, float_format=lambda x: f"{x:.5f}"))

    # --- 그림 1: 두 좌표집합 나란히 ---
    col = {"규칙집": "#1F6FB2", "ablation": "#9DC3E6", "RL": "#1F4E79",
           "증류": "#5B9BD5", "휴리스틱": "#C0846B", "정보진단": "#8A6FB0"}
    order = sb[sb.set == "eval250"].sort_values("pdr_wog").label.tolist()[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 0.5 * len(order) + 2.0), sharey=True)
    for ax, s, title in zip(axes, ("eval250", "ext250"),
                            ("대표점 250좌표 · seed 0–29", "외부 250좌표(무중복) · seed 10000–10029")):
        g = sb[sb.set == s].set_index("label").loc[order]
        y = np.arange(len(order))
        ax.barh(y, g.pdr_wog, color=[col[f] for f in g.family], height=0.68, zorder=3)
        ax.errorbar(g.pdr_wog, y, xerr=g.ci95, fmt="none", ecolor="#5A6570",
                    elinewidth=1.0, capsize=3, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=10.5)
        ax.set_xlabel("PDR_woG (낮을수록 우수)", fontsize=11)
        ax.set_title(title, fontsize=12.5, loc="left", pad=10)
        ax.axvline(float(g.loc["START-LB3 전국단일", "pdr_wog"]), color="#C0846B",
                   ls="--", lw=1.1, zorder=2)
        ax.axvline(float(g.loc["PPO Pointer v10 (교사)", "pdr_wog"]), color="#1F4E79",
                   ls=":", lw=1.3, zorder=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#EDEFF2", lw=0.9)
        ax.set_axisbelow(True)
        for i, v in enumerate(g.pdr_wog):
            ax.text(v + float(g.ci95.max()) * 1.25, i, f"{v:.4f}", va="center",
                    fontsize=9, color=SUB)
        ax.set_xlim(0, float(g.pdr_wog.max()) * 1.18)
    fig.suptitle("현장 규칙집 vs RL 교사 vs 공정 휴리스틱 (점선=휴리스틱 기준선, 점선=PPO)",
                 fontsize=14, x=0.02, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(F / f"fieldcard_scoreboard.{ext}")
    plt.close(fig)
    print(f"\n[report] 저장 {F/'fieldcard_scoreboard.png'}")
    evidence()


def evidence():
    """규칙 근거 3패널: UAV 임계 · λ 곡선과 정보진단 · 1,000좌표 안정성."""
    import json

    d = pd.read_csv(F / "decisions_train1000.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.9))

    # (1) Red 자유선택 구간의 UAV 선택률 vs 최근접 tier3 도로거리
    g = d[(d.free_mode == 1) & (d.cls == 0)]
    edges = [0, 4, 8, 12, 16, 20, 30, 45, 70, 120]
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        gg = g[(g.near_R_amb_km >= edges[i]) & (g.near_R_amb_km < edges[i + 1])]
        if len(gg) >= 25:
            xs.append(0.5 * (edges[i] + edges[i + 1]))
            ys.append(100 * gg["mode"].mean())
            ns.append(len(gg))
    a0 = ax[0]
    a0.plot(xs, ys, "o-", color="#1F6FB2", lw=2, ms=6, zorder=3)
    a0.axvline(12.0, color="#C0392B", ls="--", lw=1.4, zorder=2)
    a0.axhline(50, color="#C8CED6", lw=1, zorder=1)
    a0.text(12.4, 8, "임계 12 km", color="#C0392B", fontsize=10.5, weight="bold")
    a0.set_xlabel("가장 가까운 Tier-3 병원까지 도로거리 (km)", fontsize=11)
    a0.set_ylabel("UAV 선택 비율 (%)", fontsize=11)
    a0.set_title("(1) Red · 두 수단이 모두 현장에 있을 때", fontsize=12, loc="left", pad=10)
    a0.set_xscale("log")
    a0.set_xticks([4, 8, 16, 32, 64])
    a0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # (2) λ 곡선 + 정보 제약
    a1 = ax[1]
    for f, lab, c, mk in ((F / "card_dev40_grid.csv", "거리 = 실제 km", "#1F6FB2", "o"),):
        dd = pd.read_csv(f)
        pv = dd.pivot_table(index="region", columns="policy", values="pdr_woG")
        pts = sorted((int(k.split("_l")[1].split("_")[0]), pv[k].mean())
                     for k in pv.columns if k.endswith("_r12_y0"))
        a1.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=c, lw=2, label=lab)
    dd = pd.read_csv(F / "cardnorm_dev40.csv")
    pv = dd.pivot_table(index="region", columns="policy", values="pdr_woG")
    best_norm = min(pv[k].mean() for k in pv.columns)
    a1.axhline(best_norm, color="#8A6FB0", ls="--", lw=1.6,
               label=f"거리 = PPO 관측(정규화) 최적 {best_norm:.4f}")
    a1.axhline(0.14853, color="#2C7355", ls=":", lw=1.6, label="실제 km 최적 0.14853")
    a1.set_xlabel("부하 가중치  (km 상당 / 환자 1명)", fontsize=11)
    a1.set_ylabel("PDR_woG (dev40)", fontsize=11)
    a1.set_title("(2) 부하를 거리로 환산하는 계수", fontsize=12, loc="left", pad=10)
    a1.legend(frameon=False, fontsize=9.5, loc="upper center")

    # (3) 1,000좌표 안정성
    a2 = ax[2]
    st = json.loads((F / "stability.json").read_text(encoding="utf-8"))
    f2 = d[(d.free_mode == 1) & (d.cls == 0)]
    cuts = []
    for sig, gg in f2.groupby("sigcd"):
        if len(gg) >= 15 and gg["mode"].nunique() == 2:
            x = np.asarray(gg.near_R_amb_km, float)
            y = np.asarray(gg["mode"], int)
            P, N = y.sum(), len(y) - y.sum()
            best = (None, -1)
            for c in np.unique(x):
                pr = x >= c
                ba = 0.5 * ((pr & (y == 1)).sum() / P + ((~pr) & (y == 0)).sum() / N)
                if ba > best[1]:
                    best = (float(c), ba)
            cuts.append(best[0])
    a2.hist(cuts, bins=np.arange(0, 40, 2.5), color="#1F6FB2", alpha=.85, zorder=3)
    a2.axvline(np.median(cuts), color="#C0392B", ls="--", lw=1.5, zorder=4)
    a2.text(np.median(cuts) + 0.8, a2.get_ylim()[1] * .88,
            f"중위 {np.median(cuts):.1f} km", color="#C0392B", fontsize=10.5, weight="bold")
    a2.set_xlabel("시군구별로 따로 추정한 UAV 임계 (km)", fontsize=11)
    a2.set_ylabel("시군구 수", fontsize=11)
    lam = st["lambda"]
    a2.set_title(f"(3) 임계값 안정성  ·  부하가중 {lam['all']:.2f} km/명 "
                 f"[{lam['cluster_boot95'][0]:.2f}, {lam['cluster_boot95'][1]:.2f}]",
                 fontsize=12, loc="left", pad=10)
    for A in ax:
        for sp in ("top", "right"):
            A.spines[sp].set_visible(False)
        A.grid(color="#EDEFF2", lw=0.9)
        A.set_axisbelow(True)
    fig.suptitle("규칙 임계값의 근거 — 학습 1,000좌표 · 교사 결정 37,000건",
                 fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(F / f"fieldcard_evidence.{ext}")
    plt.close(fig)
    print(f"[report] 저장 {F/'fieldcard_evidence.png'}  (시군구 {len(cuts)}곳)")


if __name__ == "__main__":
    main()
