# -*- coding: utf-8 -*-
"""260908 랩미팅 브리핑용 그림 6장 (SVG).

기술적 세부보다 "무엇을 알았나" 를 한눈에 보이게 만든다.
  F1 규칙 형태 사다리와 응답곡선
  F2 ★임계값의 정체 — 치료시간 축에서 lambda 가 비례한다
  F3 ★용량축 — 구조 보정마다 의존이 한 단위씩 사라진다
  F4 전이 후회 — 상수 하나로 27개 다른 조건을 덮는가
  F5 최종 판정 (미개봉 판정셋 750좌표)
  F6 기전 — 부하항은 대기를 줄여 치료개시를 앞당긴다 · UAV 도입 효과

실행: python tools/labmeeting_20260908/plot_brief.py
산출: docs/260908랩미팅/*.svg
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from v20_threshold_report import cube, paired, _interp_opt  # noqa: E402

OUT = REPO / "docs/260908랩미팅"
V20 = REPO / "results/scoreboard/v20"
INK, SUB, GRID = "#1F2933", "#6B7683", "#D5DAE0"
C = {"K": "#8C96A0", "T": "#5B8DEF", "L": "#F2994A", "H": "#27AE60",
     "Q": "#C0392B", "S": "#8E44AD", "PPO": "#2D3B8F", "LB": "#B0B7BF"}
NAME = {"K": "현행 규칙  거리(km) + λ·부하",
        "T": "시간(분) + λ·부하",
        "L": "거리(km) + λ·대기초과",
        "H": "시간(분) + λ·대기초과",
        "Q": "시간(분) + λ·대기초과/수술실수",
        "S": "생존확률 직접 최대화 (무튜닝)"}


def setup():
    fp = Path.home() / ".fonts/NanumGothic-Regular.ttf"
    if fp.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": INK,
        "axes.edgecolor": GRID, "savefig.bbox": "tight", "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    })


def _nolabel(ax):
    """로그축 minor tick 라벨을 끈다 — NanumGothic 에 마이너스 글리프가 없어 대체문자가 찍힌다."""
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    for ax in fig.get_axes():
        _nolabel(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print(f"  {name}.svg")


# ------------------------------------------------------------------ F1
def f1(opt):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for fam in ("K", "T", "L", "H", "Q"):
        if fam not in opt["base"]:
            continue
        g = np.array(opt["base"][fam]["grid"])
        ax.plot(g[:, 0], g[:, 1], "o-", ms=4, lw=2, color=C[fam], label=NAME[fam], alpha=.95)
        k = int(np.argmin(g[:, 1]))
        ax.plot(g[k, 0], g[k, 1], "*", ms=17, color=C[fam], zorder=5,
                markeredgecolor="white", markeredgewidth=.8)
    ax.set_xscale("log")
    ax.set_xlabel("부하 교환율 λ   (환자 1명을 거리 km 또는 시간 분으로 환산)")
    ax.set_ylabel("예방가능 사망률 (낮을수록 좋음)")
    ax.grid(alpha=.25, lw=.6)
    ax.legend(fontsize=9.5, framealpha=.95, loc="upper center", ncol=1)
    ax.set_title("규칙의 식을 고치면 곡선 전체가 내려간다\n"
                 "별표 = 각 형태의 최적 임계값 · 250 시군구 × 10 시드 전수",
                 fontsize=12.5, pad=12)
    save(fig, "F1_응답곡선")


# ------------------------------------------------------------------ F2
def f2():
    """치료시간 축에서 최적 lambda 가 비례하는가 (이론의 직접 검증)."""
    T = V20 / "theory"
    grid = {"K": [4, 6, 8, 10, 12, 14, 17, 21, 26, 32], "Q": [6, 10, 14, 18, 22, 26, 32, 40, 50],
            "S": [0.25, 0.5, 0.62, 0.75, 1.0, 1.5]}
    scales = [("ts0.5", 0.5), ("ts0.75", 0.75), ("base", 1.0), ("ts1.5", 1.5),
              ("ts2.0", 2.0), ("ts3.0", 3.0)]
    pts = {f: [] for f in grid}
    for tag, s in scales:
        f = T / f"treat_{tag}.csv"
        if not (f.parent / (f.name + ".meta.json")).exists():
            continue
        d = pd.read_csv(f)
        for fam, L in grid.items():
            LL = [l for l in L if (d.policy == f"{fam}{l:g}").any()]
            if len(LL) < 3:
                continue
            m = [d[d.policy == f"{fam}{l:g}"].pdr_woG.mean() for l in LL]
            pts[fam].append((s, _interp_opt(LL, m)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    for fam, lab in (("Q", "유도형 규칙 (λ 자리가 서비스시간)"), ("K", "현행 규칙 (거리 km)")):
        p = sorted(pts[fam])
        if len(p) < 3:
            continue
        x = np.array([a for a, _ in p]); y = np.array([b for _, b in p])
        sl = np.polyfit(np.log(x), np.log(y), 1)[0]
        ax.plot(x, y, "o-", ms=6, lw=2.2, color=C[fam], label=f"{lab}\n기울기 {sl:+.2f}")
    xx = np.array([0.5, 3.0])
    ax.plot(xx, 18.6 * xx, "k--", lw=1.3, label="이론 예측: 정비례 (기울기 +1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.5×", "0.75×", "1×", "1.5×", "2×", "3×"])
    ax.set_xlabel("병원 치료시간 배율"); ax.set_ylabel("최적 임계값 λ")
    ax.grid(alpha=.25, lw=.6, which="both"); ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("치료시간을 6배 범위로 바꾸면\n최적 임계값이 정확히 그만큼 따라 움직인다", fontsize=12)

    ax = axes[1]
    p = sorted(pts["S"])
    if p:
        x = np.array([a for a, _ in p]); y = np.array([b for _, b in p])
        sl = np.polyfit(np.log(x), np.log(y), 1)[0]
        ax.plot(x, y / y[list(x).index(1.0)] if 1.0 in list(x) else y, "o-", ms=6, lw=2.2,
                color=C["S"], label=f"명부에서 직접 읽는 형태\n기울기 {sl:+.2f} (거의 평평)")
    ax.axhline(1.0, color=GRID, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.5×", "0.75×", "1×", "1.5×", "2×", "3×"])
    ax.set_ylim(0.5, 1.8)
    ax.set_xlabel("병원 치료시간 배율"); ax.set_ylabel("최적 상수 (기준=1 로 정규화)")
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("치료시간을 병원 명부에서 읽어 쓰면\n조건이 바뀌어도 다시 고를 것이 없다", fontsize=12)
    fig.suptitle("★ 임계값의 정체 — 그것은 '수술실 하나가 환자 하나를 처리하는 시간' 이다",
                 fontsize=13.5, y=1.04)
    save(fig, "F2_임계값의정체")


# ------------------------------------------------------------------ F3
def f3(opt, ceff):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    SL = {"K": (-1.297, -1.333, -1.241), "H": (-1.032, -1.073, -0.821), "Q": (-0.198, -0.250, -0.141)}
    LAB = {"K": "현행 규칙\n거리 + λ·부하", "H": "시간축 + 대기초과", "Q": "＋ 수술실수로 나눔"}
    for fam in ("K", "H", "Q"):
        pts = [(1.0, opt["base"][fam]["lam"])]
        for t, v in opt.items():
            if isinstance(v, dict) and v.get("axis") == "capa" and fam in v:
                pts.append((v["axis_value"], v[fam]["lam"]))
        pts.sort()
        x = [p[0] for p in pts]; y = [p[1] / pts[[q[0] for q in pts].index(1.0)][1] for p in pts]
        s, lo, hi = SL[fam]
        ax.plot(x, y, "o-", ms=6, lw=2.2, color=C[fam],
                label=f"{LAB[fam]}\n기울기 {s:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    xx = np.array([0.5, 2.0])
    ax.plot(xx, 1 / xx, "k--", lw=1.3, label="이론: 수술실 수에 반비례")
    ax.axhline(1.0, color=GRID, lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2]); ax.set_xticklabels(["×0.5", "×0.75", "×1", "×1.5", "×2"])
    ax.set_xlabel("병원 수술실 수 배율"); ax.set_ylabel("최적 임계값 (기준=1 로 정규화)")
    ax.grid(alpha=.25, lw=.6, which="both"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("★ 식을 한 단계씩 고칠 때마다 용량 의존이 한 단위씩 사라진다\n"
                 "세로막대 없는 점선 = 이론값 · 괄호는 좌표 재표집 95% 구간", fontsize=12, pad=12)
    save(fig, "F3_용량축")


# ------------------------------------------------------------------ F4
def f4():
    t = pd.read_csv(V20 / "transfer.csv")
    fams = [f for f in ("K", "T", "H", "Q") if f in set(t.family)]
    LAB = {"K": "현행 규칙", "T": "시간축만", "H": "시간축+대기초과", "Q": "＋수술실수로 나눔"}
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    order = sorted(t.setting.unique())
    KOR = {"amb": "AMB", "uav": "UAV", "n": "환자", "vamb": "AMB속도", "vuav": "UAV속도",
           "capa": "용량", "hamb": "AMB인계", "huav": "UAV인계"}
    lbl = []
    for s in order:
        import re
        m = re.match(r"^([a-z]+)([0-9]+)$", s)
        lbl.append(f"{KOR.get(m.group(1), m.group(1))}{m.group(2)}" if m else s)
    w = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        sub = t[t.family == fam].set_index("setting").reindex(order)
        ax.bar(np.arange(len(order)) + i * w, sub.regret * 1000, w, color=C[fam],
               label=LAB[fam], alpha=.93)
    ax.axhline(0.53, color="#C0392B", ls="--", lw=1.3)
    ax.text(len(order) - .4, 0.60, "구분 불가 수준", color="#C0392B", fontsize=9, ha="right")
    ax.set_xticks(np.arange(len(order)) + .4)
    ax.set_xticklabels(lbl, rotation=55, ha="right", fontsize=8.5)
    ax.set_ylabel("사망률 손해 (1000분율)")
    ax.grid(alpha=.25, lw=.6, axis="y"); ax.legend(fontsize=9.5, ncol=4)
    ax.set_title("★ 기준 조건에서 고른 상수 하나를 27개 다른 물리 조건에 그대로 가져갔을 때의 손해\n"
                 "낮을수록 일반화 — 붙은 막대(맨 오른쪽 색)가 거의 보이지 않는다", fontsize=12.5, pad=12)
    save(fig, "F4_전이후회")


# ------------------------------------------------------------------ F5
def f5():
    T = V20 / "theory"
    a = pd.read_csv(T / "test750_final.csv")
    b = pd.read_csv(T / "test750_ppo.csv")
    cb = {p: cube(a, p) for p in a.policy.unique()}
    cb["PPO"] = cube(b, "PPO_NATIONAL")
    LAB = {"START_LB3": "현실적 휴리스틱 (START-LB3)", "CARD_K12": "현행 현장 규칙집",
           "CARD_H12": "시간축 + 대기초과", "CARD_MODE": "수단별 교환율",
           "PPO": "강화학습 교사 (PPO)", "CARD_P18": "유도형 · 병원 통신 불필요",
           "CARD_S062": "유도형 · 무튜닝", "CARD_Q18": "유도형 (채택)"}
    COL = {"START_LB3": C["LB"], "CARD_K12": C["K"], "CARD_H12": C["H"], "CARD_MODE": C["H"],
           "PPO": C["PPO"], "CARD_P18": C["S"], "CARD_S062": C["S"], "CARD_Q18": C["Q"]}
    items = sorted(((p, c.mean()) for p, c in cb.items() if p in LAB), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y = np.arange(len(items))
    for i, (p, v) in enumerate(items):
        ax.barh(i, v, .68, color=COL[p], alpha=.93)
        ax.text(v + 0.0012, i, f"{v:.4f}", va="center", fontsize=10,
                fontweight="bold" if p in ("CARD_Q18", "PPO") else "normal")
    ax.set_yticks(y); ax.set_yticklabels([LAB[p] for p, _ in items], fontsize=10)
    ax.set_xlim(0, 0.19); ax.set_xlabel("예방가능 사망률 (낮을수록 좋음)")
    ax.grid(alpha=.25, lw=.6, axis="x")
    ax.axvline(cb["PPO"].mean(), color=C["PPO"], ls=":", lw=1.4)
    ax.set_title("최종 판정 — 임계값 조정에 한 번도 쓰지 않은 750개 좌표\n"
                 "점선 = 강화학습 교사 수준 · 전부 같은 좌표·같은 난수", fontsize=12.5, pad=12)
    save(fig, "F5_최종판정")


# ------------------------------------------------------------------ F6
def f6():
    M = V20 / "mechanism"
    d = pd.read_csv(M / "base.csv")
    LAB = {"NOLOAD": "부하를 안 보면\n(최근접만)", "START_LB3": "현실적\n휴리스틱",
           "K12": "현행\n규칙집", "Q18": "유도형\n(채택)", "Q50": "부하를 과하게\n보면"}
    keys = ["NOLOAD", "START_LB3", "K12", "Q18", "Q50"]
    g = d[d.policy.isin(keys)].groupby("policy").agg(
        q=("frac_queued", "mean"), w=("wait_med_queued", "mean"),
        ty=("t_admit_med_Y", "mean"), pdr=("pdr_woG", "mean")).reindex(keys)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    col = [C["LB"], C["LB"], C["K"], C["Q"], "#E8A9A0"]
    for ax, (m, ttl, unit) in zip(axes, [
            ("q", "병상에서 기다린 환자 비율", "%"),
            ("w", "기다린 시간 (중위)", "분"),
            ("ty", "Yellow 환자 치료개시 시각 (중위)", "분")]):
        v = g[m].values * (100 if m == "q" else 1)
        ax.bar(range(len(g)), v, .66, color=col, alpha=.93)
        for i, x in enumerate(v):
            ax.text(i, x * 1.02, f"{x:.1f}{unit}", ha="center", fontsize=9.5,
                    fontweight="bold" if g.index[i] == "Q18" else "normal")
        ax.set_xticks(range(len(g))); ax.set_xticklabels([LAB[k] for k in g.index], fontsize=9)
        ax.set_title(ttl, fontsize=11.5)
        ax.grid(alpha=.25, lw=.6, axis="y")
        ax.set_ylim(0, max(v) * 1.18)
    fig.suptitle("기전 — 부하를 보는 항은 병원 쏠림을 막아 수술실 대기를 줄이고 치료개시를 앞당긴다\n"
                 "다만 과하게 보면 더 멀리 가야 해서 치료개시가 다시 늦어진다",
                 fontsize=12.5, y=1.06)
    fig.tight_layout()
    save(fig, "F6_기전")


# ------------------------------------------------------------------ F7
def f7():
    T = V20 / "theory"
    fam, L = "Q", [6, 10, 14, 18, 22, 26, 32, 40, 50]
    out = {}
    for tag, lab in (("extreme_uav0", "UAV 0대\n(도입 전)"), ("treat_base", "UAV 26대\n(현재 설정)")):
        f = T / f"{tag}.csv"
        if not (f.parent / (f.name + ".meta.json")).exists():
            continue
        d = pd.read_csv(f)
        LL = [l for l in L if (d.policy == f"{fam}{l:g}").any()]
        m = [d[d.policy == f"{fam}{l:g}"].pdr_woG.mean() for l in LL]
        out[lab] = min(m)
    if len(out) < 2:
        print("  F7 skip"); return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ks = list(out)
    ax.bar(range(2), [out[k] for k in ks], .5, color=[C["LB"], C["Q"]], alpha=.93)
    for i, k in enumerate(ks):
        ax.text(i, out[k] + .004, f"{out[k]:.3f}", ha="center", fontsize=12, fontweight="bold")
    drop = 100 * (1 - out[ks[1]] / out[ks[0]])
    ax.annotate("", xy=(1, out[ks[1]] + .002), xytext=(0, out[ks[0]] - .002),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2))
    ax.text(.5, (out[ks[0]] + out[ks[1]]) / 2 + .012, f"-{drop:.0f}%", color="#C0392B",
            fontsize=15, fontweight="bold", ha="center")
    ax.set_xticks(range(2)); ax.set_xticklabels(ks, fontsize=11)
    ax.set_ylabel("예방가능 사망률"); ax.set_ylim(0, .24)
    ax.grid(alpha=.25, lw=.6, axis="y")
    ax.set_title("UAV 도입 효과를 처음 측정했다\n같은 규칙·같은 좌표에서 UAV 출발지만 없앤 비교", fontsize=12, pad=10)
    save(fig, "F7_UAV도입효과")


def main():
    setup()
    opt = json.loads((V20 / "optima.json").read_text(encoding="utf-8"))
    import random
    man = json.loads((REPO / "scenarios/manifests/v19/tradeoff250_manifest.json").read_text())
    random.seed(0)
    c0 = np.concatenate([pd.read_csv(
        (man[k] if isinstance(man[k], str) else man[k]["path"]).rsplit("/", 1)[0] + "/hospital_info.csv"
    )["수술실수"].values for k in random.sample(list(man), 40)]).astype(float)
    ceff = {s: np.maximum(1, np.round(c0 * s)).mean() for s in (0.5, 0.75, 1.0, 1.5, 2.0)}
    print("[brief] SVG 생성")
    f1(opt); f2(); f3(opt, ceff); f4(); f5(); f6(); f7()
    print(f"[brief] 완료 → {OUT}")


if __name__ == "__main__":
    main()
