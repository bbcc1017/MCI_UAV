# -*- coding: utf-8 -*-
"""v17 논문형 특징증류 집계·그림.

subcommands
  ladder     : 충실도 사다리 표(markdown/CSV)
  importance : 중요도 막대그래프(논문 Fig.8 대응) + 유형별 합계
  logstats   : 교사 결정의 순위·구간별 선택비율(논문 Fig.9-10 대응)
  closedloop : 폐루프 cube 병합 + paired 검정 + 충실도-성능 산점도
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
V17 = REPO / "results/scoreboard/v17"
DISTILL = V17 / "distill"

INK = "#30343A"
SUB = "#606870"
PALETTE = {
    "BASE": "#9AA3AD",
    "RANK": "#2F6FB2",
    "CAT": "#3E9C6E",
    "REL": "#C98A2B",
    "GLOBAL": "#8A5FB0",
}


def setup_font() -> None:
    path = Path.home() / ".fonts/NanumGothic-Regular.ttf"
    if path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": SUB,
        "ytick.color": INK,
        "axes.edgecolor": "#D5DAE0",
        "savefig.bbox": "tight",
        "savefig.dpi": 170,
    })


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


# ----------------------------------------------------------------- ladder
def ladder_main(args) -> None:
    df = pd.read_csv(Path(args.fit_dir) / "fit_summary.csv")
    base = df[df.feature_set == "BASE43"].set_index("model")
    df["delta_vs_base"] = df.apply(
        lambda r: r.fidelity_full - float(base.loc[r.model, "fidelity_full"])
        if r.model in base.index else np.nan, axis=1)
    df["x_chance"] = df.fidelity_full / df.chance_full
    cols = ["policy", "feature_set", "model", "n_features", "n_features_used", "n_aug_used",
            "leaves", "fidelity_full", "delta_vs_base", "x_chance", "fidelity_class",
            "fidelity_dest", "fidelity_mode", "teacher_top3_hit", "prob_retention",
            "eta_rank_match", "chance_full"]
    out = df[cols].sort_values(["model", "fidelity_full"], ascending=[True, False])
    out.to_csv(Path(args.fit_dir) / "ladder.csv", index=False, encoding="utf-8-sig")

    lines = ["| 특징집합 | 모델 | 특징수 | 사용 | 증강사용 | exact | Δ vs BASE | ×random | dest | top3 | 확률보존 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out.itertuples(index=False):
        d = "—" if not np.isfinite(r.delta_vs_base) else f"{r.delta_vs_base:+.4f}"
        lines.append(
            f"| {r.feature_set} | {r.model} | {r.n_features} | {r.n_features_used} | "
            f"{r.n_aug_used} | {r.fidelity_full:.4f} | {d} | {r.x_chance:.1f}× | "
            f"{r.fidelity_dest:.4f} | {r.teacher_top3_hit:.4f} | {r.prob_retention:.4f} |")
    (Path(args.fit_dir) / "ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 모델별 최고 팔
    print("\n[모델별 최고 exact]")
    for model, g in out.groupby("model"):
        b = g.iloc[0]
        print(f"  {model:<5} {b.feature_set:<8} {b.fidelity_full:.4f} "
              f"(BASE43 {float(base.loc[model,'fidelity_full']):.4f})")


# ------------------------------------------------------------- importance
def importance_main(args) -> None:
    setup_font()
    imp = pd.read_csv(Path(args.fit_dir) / "feature_importance.csv")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    fig, axes = plt.subplots(1, len(policies), figsize=(7.6 * len(policies), 6.6))
    fig.subplots_adjust(wspace=0.42)
    axes = np.atleast_1d(axes)
    for ax, pol in zip(axes, policies):
        g = imp[imp.policy == pol].nlargest(args.top, "importance").iloc[::-1]
        colors = [PALETTE.get(f, "#9AA3AD") for f in g.family]
        ax.barh(range(len(g)), g.importance, color=colors, height=0.72)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.feature, fontsize=10.5)
        ax.set_xlabel("불순도 감소 기여도 (정규화)", fontsize=11)
        ax.set_title(f"{pol}  상위 {args.top}", fontsize=13, pad=12, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#EDEFF2", lw=0.9)
        ax.set_axisbelow(True)
        hi = float(max(g.importance))
        ax.set_xlim(0, hi * 1.22)
        for i, v in enumerate(g.importance):
            ax.text(v + hi * 0.015, i, f"{v:.3f}", va="center", fontsize=9, color=SUB)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE.values()]
    labels = ["기존 43특징", "Rank(순위)", "Categorical(구간)", "Relative(상대량)", "Global(상태요약)"]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("증류 트리가 실제로 쓰는 특징 — 교사는 순수 PPO (GBDT 31잎)",
                 fontsize=15, x=0.02, ha="left", y=0.99)
    for ext in ("png", "svg"):
        fig.savefig(Path(args.fit_dir) / f"importance.{ext}")
    plt.close(fig)

    rows = []
    for pol in policies:
        g = imp[imp.policy == pol]
        tot = g.importance.sum()
        for fam, gg in g.groupby("family"):
            rows.append({"policy": pol, "family": fam,
                         "n_features": len(gg), "importance_sum": gg.importance.sum(),
                         "share": gg.importance.sum() / max(tot, 1e-12)})
    fam = pd.DataFrame(rows).sort_values(["policy", "share"], ascending=[True, False])
    fam.to_csv(Path(args.fit_dir) / "importance_by_family.csv", index=False,
               encoding="utf-8-sig")
    print(fam.to_string(index=False))


# --------------------------------------------------------------- logstats
def logstats_main(args) -> None:
    setup_font()
    df = pd.read_csv(args.stats)
    axes_show = [("rank_eta_all", "선택한 병원의 ETA 순위 (0 = 가장 가까움)"),
                 ("eta_bin", "선택한 병원까지 이송시간 구간"),
                 ("p_sent_bin", "선택 시점에 그 병원으로 이미 보낸 환자 수"),
                 ("occ_bin", "선택한 병원의 점유율 구간")]
    tick_labels = {
        "eta_bin": ["<5분", "5–10", "10–20", "20–40", "40분+"],
        "p_sent_bin": ["0명", "1명", "2명", "3명", "4명+"],
        "occ_bin": ["<25%", "25–50", "50–75", "75–100", "100%+"],
    }
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.2))
    for ax, (key, label) in zip(axs.ravel(), axes_show):
        g = df[(df.axis == key) & (df.value >= 0)].sort_values("value")
        g = g[g.value <= args.max_rank]
        ax.bar(g.value, g.share_of_decisions * 100, color="#2F6FB2", width=0.78)
        if key in tick_labels:
            lab = tick_labels[key][: len(g)]
            ax.set_xticks(list(g.value)[: len(lab)])
            ax.set_xticklabels(lab, fontsize=10)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("교사 선택 비율 (%)", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#EDEFF2", lw=0.9)
        ax.set_axisbelow(True)
        note = {
            "rank_eta_all": f"최근접 선택 {float(g[g.value == 0].share_of_decisions.iloc[0])*100:.1f}%"
            if (g.value == 0).any() else "",
            "eta_bin": f"20분 이상 {float(g[g.value >= 3].share_of_decisions.sum())*100:.1f}%",
            "p_sent_bin": f"3명 미만 {float(g[g.value <= 2].share_of_decisions.sum())*100:.1f}%",
            "occ_bin": f"25% 미만 {float(g[g.value == 0].share_of_decisions.iloc[0])*100:.1f}%"
            if (g.value == 0).any() else "",
        }.get(key, "")
        if note:
            ax.text(0.97, 0.93, note, transform=ax.transAxes, ha="right",
                    fontsize=10.5, color="#2F6FB2", weight="bold")
    fig.suptitle("순수 PPO 교사의 실제 선택 분포 (학습 1,000좌표 · 37,000 결정)",
                 fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(Path(args.stats).with_suffix("").as_posix() + f"_ranks.{ext}")
    plt.close(fig)
    print("[logstats] 그림 저장", Path(args.stats).with_suffix("").as_posix() + "_ranks.png")
    for key, _ in axes_show:
        g = df[(df.axis == key) & (df.value >= 0)].sort_values("value").head(4)
        print(f"  {key}: " + "  ".join(
            f"r{int(r.value)}={r.share_of_decisions*100:.1f}%" for r in g.itertuples()))


# -------------------------------------------------------------- closedloop
def _cube_from_long(path: Path, policy: str, n_seeds: int) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df = df[df.policy == policy]
    if df.empty:
        raise ValueError(f"{path}: policy {policy} 없음")
    df = df[df.seed < n_seeds]
    piv = df.pivot_table(index="region", columns="seed", values="pdr_woG")
    if piv.isna().any().any():
        raise ValueError(f"{path}/{policy}: 결측 cell")
    return piv.to_numpy(dtype=float), list(piv.index)


def _cube_min_from_long(path: Path, prefix: str, n_seeds: int) -> tuple[np.ndarray, list[str]]:
    """좌표별 사후 최소(oracle형) cube. 표시 seed 구간에서 argmin 하므로 라벨에 명시한다."""
    df = pd.read_csv(path)
    df = df[df.policy.str.startswith(prefix) & (df.seed < n_seeds)]
    if df.empty:
        raise ValueError(f"{path}: prefix {prefix} 없음")
    means = df.groupby(["region", "policy"]).pdr_woG.mean().reset_index()
    best = means.loc[means.groupby("region").pdr_woG.idxmin(), ["region", "policy"]]
    sel = df.merge(best, on=["region", "policy"])
    piv = sel.pivot_table(index="region", columns="seed", values="pdr_woG")
    if piv.isna().any().any():
        raise ValueError(f"{path}/{prefix}: 결측 cell")
    return piv.to_numpy(dtype=float), list(piv.index)


def _paired_wtl(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """지역별 에피소드 배열 95%CI 기준 승/무/패 (b 대비 a 개선)."""
    w = t = l = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]          # 양수 = a 가 더 낮은 PDR = 개선
        m, c = float(d.mean()), ci95(d)
        if m > c:
            w += 1
        elif m < -c:
            l += 1
        else:
            t += 1
    return w, t, l


def closedloop_main(args) -> None:
    setup_font()
    spec = json.load(open(args.spec, encoding="utf-8"))
    n = int(spec.get("n_seeds", 30))
    cubes, regions_ref = {}, None
    for item in spec["arms"]:
        label = item["label"]
        if item["kind"] == "long_csv":
            cube, regions = _cube_from_long(REPO / item["path"], item["policy"], n)
        elif item["kind"] == "long_csv_min":
            cube, regions = _cube_min_from_long(REPO / item["path"], item["prefix"], n)
        elif item["kind"] == "npz_cube":
            with np.load(REPO / item["path"], allow_pickle=True) as z:
                cube = np.asarray(z[item["value_key"]], dtype=float)[:, :n]
                regions = [str(x) for x in z["regions"]]
        else:
            raise ValueError(item["kind"])
        if regions is not None:
            if regions_ref is None:
                regions_ref = regions
            elif regions != regions_ref:
                raise ValueError(f"{label}: 지역 순서 불일치")
        if cube.shape[0] != 250 or cube.shape[1] != n:
            raise ValueError(f"{label}: cube {cube.shape}")
        cubes[label] = cube

    ref = spec["reference"]
    rows = []
    for label, cube in cubes.items():
        rm = cube.mean(axis=1)
        w, t, l = _paired_wtl(cube, cubes[ref])
        d = cubes[ref].mean(axis=1) - rm
        rows.append({
            "label": label,
            "family": next(x.get("family", "") for x in spec["arms"] if x["label"] == label),
            "pdr_wog_mean": float(rm.mean()),
            "ci95_region": ci95(rm),
            f"delta_vs_{ref}": float(d.mean()),
            "delta_ci95": ci95(d),
            "win": w, "tie": t, "loss": l,
            "n_regions": cube.shape[0], "n_seeds": n,
        })
    out = pd.DataFrame(rows).sort_values("pdr_wog_mean")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    # --- scoreboard 그림 ---
    fam_color = {"PPO": "#1F4E79", "증류(특징증강)": "#2F6FB2", "증류(순위만)": "#5B9BD5",
                 "증류(기존특징)": "#9DC3E6", "휴리스틱": "#C0846B"}
    g = out.sort_values("pdr_wog_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10.4, 0.52 * len(g) + 1.8))
    y = np.arange(len(g))
    ax.barh(y, g.pdr_wog_mean, color=[fam_color.get(f, "#9AA3AD") for f in g.family],
            height=0.68, zorder=3)
    ax.errorbar(g.pdr_wog_mean, y, xerr=g.ci95_region, fmt="none", ecolor="#5A6570",
                elinewidth=1.1, capsize=3, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(g.label, fontsize=10.5)
    ax.set_xlabel("PDR_woG — 예방가능 사망률 (낮을수록 우수)", fontsize=11)
    ax.set_xlim(0, float(g.pdr_wog_mean.max()) * 1.16)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color="#EDEFF2", lw=0.9)
    ax.set_axisbelow(True)
    for i, r_ in enumerate(g.itertuples(index=False)):
        ax.text(r_.pdr_wog_mean + float(g.ci95_region.max()) * 1.3, i,
                f"{r_.pdr_wog_mean:.4f}", va="center", fontsize=9.5, color=SUB)
    ref_v = float(out[out.label == ref].pdr_wog_mean.iloc[0])
    ax.axvline(ref_v, color="#C0846B", ls="--", lw=1.2, zorder=2)
    ax.set_title("대표점 250좌표 × seed 0–29 폐루프 재시뮬 — 점선 = 공정 휴리스틱 기준선",
                 fontsize=13, loc="left", pad=12)
    for ext in ("png", "svg"):
        fig.savefig(dest.with_name(dest.stem + f"_scoreboard.{ext}"))
    plt.close(fig)

    if args.fit_dir:
        fit = pd.read_csv(Path(args.fit_dir) / "fit_summary.csv")
        link = []
        for item in spec["arms"]:
            if item.get("fit_policy"):
                m = fit[fit.policy == item["fit_policy"]]
                if len(m):
                    link.append({
                        "label": item["label"],
                        "fidelity_full": float(m.fidelity_full.iloc[0]),
                        "prob_retention": float(m.prob_retention.iloc[0]),
                        "pdr": float(out[out.label == item["label"]].pdr_wog_mean.iloc[0]),
                    })
        if len(link) >= 3:
            ld = pd.DataFrame(link)
            r = float(np.corrcoef(ld.fidelity_full, ld.pdr)[0, 1])
            ld.to_csv(dest.with_name(dest.stem + "_fidelity_link.csv"), index=False,
                      encoding="utf-8-sig")
            fig, ax = plt.subplots(figsize=(6.6, 5.2))
            ax.scatter(ld.fidelity_full * 100, ld.pdr, s=70, color="#2F6FB2", zorder=3)
            ld = ld.sort_values("fidelity_full")
            for i, r_ in enumerate(ld.itertuples(index=False)):
                dx, dy = (8, 5) if i % 2 == 0 else (-8, -14)
                ax.annotate(r_.label, (r_.fidelity_full * 100, r_.pdr),
                            textcoords="offset points", xytext=(dx, dy), fontsize=9,
                            color=SUB, ha="left" if dx > 0 else "right")
            ax.set_xlabel("교사 행동 재현율 exact fidelity (%)", fontsize=11)
            ax.set_ylabel("폐루프 PDR_woG (낮을수록 우수)", fontsize=11)
            ax.set_title(f"충실도와 실제 성능의 관계  (상관 r = {r:+.2f}, n={len(ld)})",
                         fontsize=13, loc="left", pad=12)
            ax.text(0.02, 0.03, "팔마다 모델 용량이 달라 순위 검증이며 인과 주장은 아니다",
                    transform=ax.transAxes, fontsize=8.5, color="#8A929B")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(color="#EDEFF2", lw=0.9)
            ax.set_axisbelow(True)
            for ext in ("png", "svg"):
                fig.savefig(dest.with_name(dest.stem + f"_fidelity_link.{ext}"))
            plt.close(fig)
            print(f"\n[link] 충실도-PDR 상관 r={r:+.3f}  n={len(ld)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ladder")
    a.add_argument("--fit_dir", default=str(DISTILL / "trees"))
    b = sub.add_parser("importance")
    b.add_argument("--fit_dir", default=str(DISTILL / "trees"))
    b.add_argument("--policies", default="BASE43_C4,AUG68_C4")
    b.add_argument("--top", type=int, default=15)
    c = sub.add_parser("logstats")
    c.add_argument("--stats", required=True)
    c.add_argument("--max_rank", type=int, default=11)
    d = sub.add_parser("closedloop")
    d.add_argument("--spec", required=True)
    d.add_argument("--fit_dir", default="")
    d.add_argument("--out", required=True)
    args = p.parse_args()
    {"ladder": ladder_main, "importance": importance_main,
     "logstats": logstats_main, "closedloop": closedloop_main}[args.cmd](args)


if __name__ == "__main__":
    main()
