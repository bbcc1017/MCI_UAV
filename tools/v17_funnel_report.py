"""시나리오 1개 기준 후보 깔때기 — 법적 → 실효(총소요) → 실제 사용.

이전 해부(`v17_decision_anatomy.py`)는 "법적 후보"만 보고했다. 그 수치가
"마스크가 자른 뒤 남은 병원 수"라서 실제 결정에 관여할 수 있는 규모를 과대
표시한다. 여기서는 두 가지를 보탠다.

  * 실효 후보 = 그 수단으로 **총소요 30분 이내**. 이동시간뿐 아니라 **인계시간**
    을 포함한다(AMB 5분 · UAV 10분). 이전 계산은 인계를 빼먹었다.
  * 실제 사용 = 그 좌표의 한 에피소드(결정 37건) 안에서 실제로 목적지가 된 병원 수.

도농 분리도 같이 낸다(특·광역시+세종 vs 도). "Tier3 25km 내 0곳"이 전국 중위인
것은 비수도권 사정이고 서울에는 해당하지 않는다 — 그 오독을 막기 위한 분해다.

입력: static_train1000.npz · decisions_train1000.csv (재수집 0)
출력: results/scoreboard/v17/anatomy/funnel3.{png,svg} + funnel3.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "results/scoreboard/v17/fieldrules/static_train1000.npz"
DEC = REPO / "results/scoreboard/v17/fieldrules/decisions_train1000.csv"
OUT = REPO / "results/scoreboard/v17/anatomy"

V_AMB, V_UAV = 50.0, 200.0          # km/h
H_AMB, H_UAV = 5.0, 10.0            # 인계 분
CUT = 30.0                          # 실효 컷 (총소요 분)
METRO = {"11", "26", "27", "28", "29", "30", "31", "36"}   # 특·광역시 + 세종

CELLS = [("Yellow+AMB", "Y", 0), ("Yellow+UAV", "Y", 1),
         ("Red+AMB", "R", 0), ("Red+UAV", "R", 1)]
COL = {"Yellow+AMB": "#2c6fbb", "Yellow+UAV": "#1f9d76",
       "Red+AMB": "#c0392b", "Red+UAV": "#7b1f6e"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(STATIC, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    ki = {k: i for i, k in enumerate(keys)}
    dr, de, hl, ti = z["d_road"], z["d_euc"], z["heli"], z["tier"]

    per = defaultdict(list)
    with open(DEC, encoding="utf-8-sig") as f:
        for d in csv.DictReader(f):
            per[d["key"]].append((int(float(d["hosp"])), int(float(d["mode"])),
                                 int(float(d["cls"])), float(d["site_near_tier3_km"])))

    rec = {(nm, kind): [] for nm, _, _ in CELLS for kind in ("legal", "eff", "used")}
    metro_flag = []
    for k, rows in per.items():
        i = ki[k]
        metro_flag.append(k.rsplit("_", 2)[-2][:2] in METRO)
        heli = hl[i] > 0.5
        t3 = ti[i] == 3
        ta = dr[i] * 60.0 / V_AMB + H_AMB      # 총소요 = 이동 + 인계
        tu = de[i] * 60.0 / V_UAV + H_UAV
        elig = {("Yellow+AMB"): (np.ones(len(heli), bool), ta, 1, 0),
                ("Yellow+UAV"): (heli, tu, 1, 1),
                ("Red+AMB"): (t3, ta, 0, 0),
                ("Red+UAV"): (t3 & heli, tu, 0, 1)}
        for nm, (mask, t, cls, mode) in elig.items():
            rec[(nm, "legal")].append(int(mask.sum()))
            rec[(nm, "eff")].append(int((mask & (t <= CUT)).sum()))
            rec[(nm, "used")].append(len({h for h, m, c, _ in rows
                                          if m == mode and c == cls}))
    metro_flag = np.asarray(metro_flag)

    def q(nm, kind, sel=None):
        a = np.asarray(rec[(nm, kind)], float)
        if sel is not None:
            a = a[sel]
        return np.median(a), np.percentile(a, 10), np.percentile(a, 90)

    res = {"cut_min": CUT, "handover": [H_AMB, H_UAV], "n_coord": len(per),
           "cells": {}, "metro_n": int(metro_flag.sum())}
    for nm, _, _ in CELLS:
        res["cells"][nm] = {
            kind: dict(zip(("median", "p10", "p90"), map(float, q(nm, kind))))
            for kind in ("legal", "eff", "used")}
        res["cells"][nm]["eff_metro"] = float(q(nm, "eff", metro_flag)[0])
        res["cells"][nm]["eff_rural"] = float(q(nm, "eff", ~metro_flag)[0])

    # Red 수단 선택 곡선
    red = [(m, nt) for rows in per.values() for h, m, c, nt in rows if c == 0]
    nt = np.asarray([x[1] for x in red])
    md = np.asarray([x[0] for x in red], float)
    ed = [0, 2, 5, 10, 15, 20, 30, 50, 1e9]
    cx, cy, cn = [], [], []
    for a, b in zip(ed[:-1], ed[1:]):
        m = (nt >= a) & (nt < b)
        if m.sum() >= 20:
            cx.append(min(b, 70) if b < 1e9 else 70)
            cy.append(float(md[m].mean()))
            cn.append(int(m.sum()))
    res["red_curve"] = {"x_upper_km": cx, "uav_rate": cy, "n": cn}

    # ---------------------------------------------------------------- 그림
    fig = plt.figure(figsize=(15.6, 4.9))

    ax = fig.add_subplot(1, 3, 1)
    w = 0.26
    xs = np.arange(4)
    for j, (kind, lab, al) in enumerate((("legal", "법적 후보 (마스크 통과)", 0.32),
                                         ("eff", f"{CUT:.0f}분 내 도달 (인계 포함)", 0.66),
                                         ("used", "실제 사용", 1.0))):
        med = [q(nm, kind)[0] for nm, _, _ in CELLS]
        lo = [q(nm, kind)[0] - q(nm, kind)[1] for nm, _, _ in CELLS]
        hi = [q(nm, kind)[2] - q(nm, kind)[0] for nm, _, _ in CELLS]
        cols = [COL[nm] for nm, _, _ in CELLS]
        ax.bar(xs + (j - 1) * w, med, w, color=cols, alpha=al,
               yerr=[lo, hi], capsize=3, error_kw=dict(lw=1, alpha=0.6),
               label=lab, edgecolor="k", linewidth=0.4)
        for x, v in zip(xs + (j - 1) * w, med):
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([nm.replace("+", "\n+") for nm, _, _ in CELLS], fontsize=9)
    ax.set_ylabel("시나리오 1개당 병원 수 (중위)")
    ax.set_title("(1) 법적 후보와 도달 가능 후보는 다르다", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 54)

    ax = fig.add_subplot(1, 3, 2)
    m_ = [res["cells"][nm]["eff_metro"] for nm, _, _ in CELLS]
    r_ = [res["cells"][nm]["eff_rural"] for nm, _, _ in CELLS]
    ax.bar(xs - 0.19, m_, 0.36, color="#2b6cb0", label=f"특·광역시 (n={int(metro_flag.sum())})",
           edgecolor="k", linewidth=0.4)
    ax.bar(xs + 0.19, r_, 0.36, color="#c98a2b", label=f"도 (n={int((~metro_flag).sum())})",
           edgecolor="k", linewidth=0.4)
    for x, v in zip(xs - 0.19, m_):
        ax.text(x, v + 0.25, f"{v:.0f}", ha="center", fontsize=9)
    for x, v in zip(xs + 0.19, r_):
        ax.text(x, v + 0.25, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([nm.replace("+", "\n+") for nm, _, _ in CELLS], fontsize=9)
    ax.set_ylabel(f"{CUT:.0f}분 내 도달 병원 수 (중위)")
    ax.set_title("(2) 도달 가능 후보는 도농이 갈린다", fontsize=11, weight="bold")
    ax.legend(fontsize=8.5)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(cx, np.asarray(cy) * 100, "o-", color="#7b1f6e", lw=2.4, ms=7)
    for x, y, n in zip(cx, cy, cn):
        ax.annotate(f"{y*100:.0f}%", (x, y * 100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    ax.axvline(12, color="#c0392b", ls=":", lw=2)
    ax.text(12.6, 8, "12 km", color="#c0392b", fontsize=10, weight="bold")
    ax.set_xlabel("현장에서 가장 가까운 Tier3까지 도로거리 (km, 구간 상한)")
    ax.set_ylabel("Red 결정에서 UAV 선택률 (%)")
    ax.set_title("(3) Tier3가 가까우면 UAV를 안 쓴다", fontsize=11, weight="bold")
    ax.set_ylim(-5, 100)
    ax.text(0.97, 0.06, "0–2 km 구간 140건 전부 앰뷸런스",
            transform=ax.transAxes, ha="right", fontsize=9, color="#333")

    fig.suptitle("시나리오 1개 기준 — 법적 후보 · 도달 가능 후보 · 실제 사용",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"funnel3.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (OUT / "funnel3.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[funnel3] 좌표 {len(per)} · 특·광역시 {int(metro_flag.sum())}", flush=True)
    hdr = f"{'셀':12s}{'법적':>8s}{'실효':>8s}{'사용':>8s}{'실효(광역시)':>13s}{'실효(도)':>10s}"
    print(hdr, flush=True)
    for nm, _, _ in CELLS:
        c = res["cells"][nm]
        print(f"{nm:12s}{c['legal']['median']:8.0f}{c['eff']['median']:8.0f}"
              f"{c['used']['median']:8.0f}{c['eff_metro']:13.0f}{c['eff_rural']:10.0f}", flush=True)
    print("  Red UAV 곡선:", [f"{u*100:.0f}%" for u in cy], flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
