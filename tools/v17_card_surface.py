"""CARD 규칙집의 3차원 응답면 — z축은 항상 PDR_woG.

두 종류를 만든다.

  fig1 파라미터 응답면 (개발 40좌표 × 20 seed = 팔당 800 에피소드)
       x = 부하 가중 lam (km/명) · y = Red UAV 임계 (km) · z = PDR
       숫자 두 개를 왜 그 값으로 골랐는지, 그리고 평탄한지를 곡면으로 본다.

  fig2 지리 응답면 (대표점 250좌표 × 30 seed)
       x = 현장에서 최근접 Tier3 도로거리 · y = 30분 내 앰뷸런스로 갈 수 있는 병원 수
       z = CARD PDR / (PPO − CARD) 개선폭
       "어떤 현장에서 규칙집이 이기나"를 지형 축으로 본다.

입력(재수집 0): card_dev40_grid{,2,3}.csv · card_eval250_seed0_29.csv ·
  ppo_eval250_seed0_29.csv · tree_eval250_seed0_29.csv · eval250 매니페스트의
  hospital_info.csv(좌표별 병원 물리량)
출력: results/scoreboard/v17/anatomy/card_surface_{param,geo}.{png,svg} + card_surface.json
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
FR = REPO / "results/scoreboard/v17/fieldrules"
DS = REPO / "results/scoreboard/v17/distill"
OUT = REPO / "results/scoreboard/v17/anatomy"
MANI = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"

V_AMB, H_AMB = 50.0, 5.0
CUT = 30.0
LAM_STAR, RED_STAR = 12.0, 12.0


def _rows(fn):
    with open(FR / fn if (FR / fn).exists() else DS / fn, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- 파라미터 격자
def param_grid():
    rows = []
    for f in ("card_dev40_grid.csv", "card_dev40_grid2.csv", "card_dev40_grid3.csv"):
        rows += _rows(f)
    pat = re.compile(r"^[A-Z]_l([\d.]+)(?:_r([\d.]+))?(?:_y(\d+))?$")
    acc = defaultdict(list)
    for r in rows:
        m = pat.match(r["policy"])
        if not m:
            continue
        lam = float(m.group(1))
        red = float(m.group(2)) if m.group(2) else None
        yh = float(m.group(3)) if m.group(3) else None
        acc[(lam, red, yh)].append(float(r["pdr_woG"]))
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


# ------------------------------------------------------------ 좌표 정적 특성
def geo_features():
    mani = json.load(open(MANI, encoding="utf-8"))
    out = {}
    for key, v in mani.items():
        p = v if isinstance(v, str) else v["path"]
        f = os.path.join(os.path.dirname(p), "hospital_info.csv")
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        rd = np.asarray([float(x["road_dist"]) for x in rows])
        t3 = np.asarray([x["종별코드"].strip() == "1" for x in rows])
        ta = rd * 60.0 / V_AMB + H_AMB
        out[key] = dict(
            near_t3=float(rd[t3].min()) if t3.any() else float("nan"),
            near_hosp=float(rd.min()),
            n_reach30=int((ta <= CUT).sum()),
            n_t3=int(t3.sum()))
    return out


def region_mean(fn, pol):
    acc = defaultdict(list)
    for r in _rows(fn):
        if r["policy"] == pol:
            acc[r["region"]].append(float(r["pdr_woG"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _surf(ax, X, Y, Z, xlab, ylab, zlab, title, cmap="viridis", elev=24, azim=-125):
    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="#333", linewidth=0.3,
                    rstride=1, cstride=1, alpha=0.96, antialiased=True)
    ax.set_xlabel(xlab, labelpad=7)
    ax.set_ylabel(ylab, labelpad=7)
    ax.set_zlabel(zlab, labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, weight="bold")


def fig_param(G, res):
    lams = sorted({k[0] for k in G if k[1] is not None and k[0] <= 64})
    reds = sorted({k[1] for k in G if k[1] is not None and k[1] <= 24})
    fig = plt.figure(figsize=(15.8, 5.0))

    # (1) 3D: lam × red_km (yhold = 14)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    Z = np.full((len(reds), len(lams)), np.nan)
    for i, r in enumerate(reds):
        for j, l in enumerate(lams):
            if (l, r, 14.0) in G:
                Z[i, j] = G[(l, r, 14.0)][0]
    X, Y = np.meshgrid(np.arange(len(lams)), np.arange(len(reds)))
    _surf(ax, X, Y, Z, "부하 가중 lam (km/명)", "Red UAV 임계 (km)", "PDR_woG",
          "(1) 숫자 두 개의 응답면", cmap="viridis_r")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams], fontsize=7.5)
    ax.set_yticks(range(len(reds))); ax.set_yticklabels([f"{r:g}" for r in reds], fontsize=8)
    bi = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.text(bi[1], bi[0], np.nanmin(Z), f"  최적 {np.nanmin(Z):.4f}", color="#c0392b",
            fontsize=9, weight="bold")

    # (2) lam 단면 — yhold 3값
    ax = fig.add_subplot(1, 3, 2)
    for yh, c, lab in ((0.0, "#2c6fbb", "항상 Yellow 우선"),
                       (14.0, "#1f9d76", "Yellow 14명 규칙"),
                       (99999.0, "#c0392b", "항상 Red 우선")):
        xs = [l for l in lams if (l, RED_STAR, yh) in G]
        ys = [G[(l, RED_STAR, yh)][0] for l in xs]
        if xs:
            ax.plot(xs, ys, "o-", color=c, lw=2, ms=5, label=lab)
    ax.axvline(LAM_STAR, color="#555", ls=":", lw=1.6)
    ax.text(LAM_STAR + 0.6, ax.get_ylim()[1] * 0.995, "채택 12", fontsize=9, va="top")
    ax.set_xlabel("부하 가중 lam (km/명)"); ax.set_ylabel("PDR_woG")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_title("(2) 부하 가중이 전부, 등급 규칙은 곁가지", fontsize=11, weight="bold")
    ax.legend(fontsize=8.5)

    # (3) lam 미세 스윕
    ax = fig.add_subplot(1, 3, 3)
    fine = sorted([k[0] for k in G if k[1] is None and k[2] is None])
    if fine:
        ys = [G[(l, None, None)][0] for l in fine]
        ax.plot(fine, ys, "o-", color="#7b1f6e", lw=2.2, ms=7)
        lo, hi = min(ys), max(ys)
        ax.axhspan(lo, lo + (hi - lo) * 0.15, color="#ffe08a", alpha=0.45, zorder=0)
        for x, y in zip(fine, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        res["fine_lam"] = {"lam": fine, "pdr": ys}
    ax.set_xlabel("부하 가중 lam (km/명)"); ax.set_ylabel("PDR_woG")
    ax.set_title("(3) 10–13 구간은 평탄하다", fontsize=11, weight="bold")

    fig.suptitle("CARD 파라미터 응답면 — 개발 40좌표 × 20 seed (팔당 800 에피소드)",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"card_surface_param.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    res["param_best"] = {"lam": float(lams[bi[1]]), "red_km": float(reds[bi[0]]),
                         "pdr": float(np.nanmin(Z))}


def fig_geo(res):
    geo = geo_features()
    card = region_mean("card_eval250_seed0_29.csv", "CARD")
    ppo = region_mean("ppo_eval250_seed0_29.csv", "PPO_POINTER_V10")
    keys = sorted(set(geo) & set(card) & set(ppo))
    x = np.asarray([geo[k]["near_t3"] for k in keys])
    y = np.asarray([geo[k]["n_reach30"] for k in keys])
    zc = np.asarray([card[k] for k in keys])
    zd = np.asarray([ppo[k] - card[k] for k in keys])
    res["n_regions"] = len(keys)

    xb = np.unique(np.percentile(x, [0, 25, 50, 75, 100]))
    yb = np.unique(np.percentile(y, [0, 25, 50, 75, 100]))
    xi = np.clip(np.digitize(x, xb[1:-1]), 0, len(xb) - 2)
    yi = np.clip(np.digitize(y, yb[1:-1]), 0, len(yb) - 2)
    nx, ny = len(xb) - 1, len(yb) - 1
    Zc = np.full((ny, nx), np.nan); Zd = np.full((ny, nx), np.nan)
    N = np.zeros((ny, nx), int)
    for a in range(ny):
        for b in range(nx):
            m = (yi == a) & (xi == b)
            N[a, b] = int(m.sum())
            if m.sum() >= 3:
                Zc[a, b] = zc[m].mean(); Zd[a, b] = zd[m].mean()
    xc = [(xb[i] + xb[i + 1]) / 2 for i in range(nx)]
    yc = [(yb[i] + yb[i + 1]) / 2 for i in range(ny)]
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    fig = plt.figure(figsize=(15.8, 5.0))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _surf(ax, X, Y, Zc, "최근접 Tier3 (km)", "30분 내 병원 수", "PDR_woG",
          "(1) 규칙집 성능의 지형", cmap="magma_r")
    ax.set_xticks(range(nx)); ax.set_xticklabels([f"{v:.0f}" for v in xc], fontsize=8)
    ax.set_yticks(range(ny)); ax.set_yticklabels([f"{v:.0f}" for v in yc], fontsize=8)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    _surf(ax, X, Y, Zd * 1000, "최근접 Tier3 (km)", "30분 내 병원 수",
          "PPO - CARD (x0.001)", "(2) 어디서 이기나", cmap="coolwarm")
    ax.set_xticks(range(nx)); ax.set_xticklabels([f"{v:.0f}" for v in xc], fontsize=8)
    ax.set_yticks(range(ny)); ax.set_yticklabels([f"{v:.0f}" for v in yc], fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    sc = ax.scatter(x, zd * 1000, c=y, cmap="viridis", s=34, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="30분 내 병원 수")
    ax.axhline(0, color="k", lw=1, ls=":")
    ed = [0, 5, 10, 20, 40, 400]
    cx, cy = [], []
    for a, b in zip(ed[:-1], ed[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 5:
            cx.append(min(b, 60)); cy.append(zd[m].mean() * 1000)
    ax.plot(cx, cy, "s-", color="#c0392b", lw=2.2, ms=7, label="구간 평균")
    ax.set_xlabel("최근접 Tier3 도로거리 (km)"); ax.set_ylabel("PPO - CARD (x0.001)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_title("(3) 이득은 원거리 좌표에 몰린다", fontsize=11, weight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("CARD 지형 응답면 — 대표점 250좌표 × 30 seed",
                 fontsize=13.5, weight="bold")
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"card_surface_geo.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    res["geo"] = {
        "x_bins_km": [float(v) for v in xb], "y_bins": [float(v) for v in yb],
        "cell_n": N.tolist(),
        "card_pdr": np.where(np.isfinite(Zc), Zc, None).tolist(),
        "delta_ppo_card": np.where(np.isfinite(Zd), Zd, None).tolist(),
        "corr_x_delta": float(np.corrcoef(np.log1p(x), zd)[0, 1]),
        "corr_y_delta": float(np.corrcoef(y, zd)[0, 1]),
        "delta_by_near_t3": {f"{a}-{b}km": float(zd[(x >= a) & (x < b)].mean())
                             for a, b in zip(ed[:-1], ed[1:])
                             if ((x >= a) & (x < b)).sum() >= 5},
        "win_rate_by_bin": {f"{a}-{b}km": float((zd[(x >= a) & (x < b)] > 0).mean())
                            for a, b in zip(ed[:-1], ed[1:])
                            if ((x >= a) & (x < b)).sum() >= 5},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print("[1/2] 파라미터 응답면", flush=True)
    fig_param(param_grid(), res)
    print("[2/2] 지형 응답면", flush=True)
    fig_geo(res)
    (OUT / "card_surface.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("  최적 파라미터:", res["param_best"], flush=True)
    print("  미세 스윕:", {f"{l:g}": round(p, 5)
                        for l, p in zip(*res["fine_lam"].values())}, flush=True)
    g = res["geo"]
    print(f"  상관 log(최근접Tier3) vs 개선폭 {g['corr_x_delta']:+.3f} · "
          f"30분내 병원수 vs 개선폭 {g['corr_y_delta']:+.3f}", flush=True)
    print("  구간별 개선폭:", {k: round(v, 5) for k, v in g["delta_by_near_t3"].items()}, flush=True)
    print("  구간별 승률 :", {k: round(v, 3) for k, v in g["win_rate_by_bin"].items()}, flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
