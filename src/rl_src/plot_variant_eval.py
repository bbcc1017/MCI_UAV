"""평가 CSV 가 표준 PPO/DQN/REINFORCE 3종 스키마와 다를 때 쓰는 PNG 헬퍼.

cross_location_eval.plot_results 의 Figure Convention 을 그대로 준수:
  - figsize=(20, 10), dpi=150 (= 3000x1500 px)
  - 2x2 subplot 레이아웃
    · top-left:    Mean reward (R, Green 포함)
    · top-right:   Mean reward (R_woG, Green 제외)
    · bottom-left: RL 우위 폭 (R)        — Δ vs heuristic
    · bottom-right:RL 우위 폭 (R_woG)    — Δ vs heuristic
  - Colors: heur "#888"  PPO "#1f77b4"  DQN "#ff7f0e"  REINFORCE "#2ca02c"

용도:
  1) plot_variant_eval — BC_PPO / EnrichedPPO 같이 단일 변형 vs heuristic vs
     기준 PPO(f3) 3종 비교
  2) plot_extra_eval   — A2C/TRPO/QRDQN/RecurrentPPO 4종 + heuristic + PPO(f3)

woG 컬럼이 variant CSV 에 없는 경우 f3_csv 에서 끌어와 합친다.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PALETTE = {
    "heur":          "#888888",
    "PPO":           "#1f77b4",
    "DQN":           "#ff7f0e",
    "REINFORCE":     "#2ca02c",
    "BC_PPO":        "#d62728",
    "EnrichedPPO":   "#9467bd",
    "a2c":           "#8c564b",
    "trpo":          "#e377c2",
    "qrdqn":         "#bcbd22",
    "recurrentppo":  "#17becf",
}


def _set_korean_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for kf in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic",
               "Noto Sans CJK JP", "AppleGothic"):
        if kf in installed:
            plt.rcParams["font.family"] = kf
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_variant_eval(variant_csv: str, f3_csv: str, tag: str, out_path: str):
    """단일 PPO 변형 vs heuristic vs 기준 PPO(f3) 의 2x2 컨벤션 PNG.

    Parameters
    ----------
    variant_csv : str
        eval_ppo_variant.py 가 만든 CSV. 컬럼: region, heuristic_R,
        PPO_R_baseline, <tag>_R, <tag>_R_woG, <tag>_vs_heur, <tag>_vs_baseline_PPO.
    f3_csv : str
        plan1nat_f3_eval.csv — heuristic_R_woG, PPO_R_woG 등 woG 컬럼 출처.
    tag : str
        "BC_PPO" / "EnrichedPPO" 등 variant CSV 의 컬럼 prefix.
    out_path : str
        PNG 저장 경로.
    """
    _set_korean_font()
    df = pd.read_csv(variant_csv, encoding="utf-8-sig")
    f3 = pd.read_csv(f3_csv, encoding="utf-8-sig").set_index("region")
    df["heuristic_R_woG"] = df["region"].map(f3["heuristic_R_woG"])
    df["PPO_R_woG_baseline"] = df["region"].map(f3["PPO_R_woG"])

    regions = df["region"].tolist()
    x = np.arange(len(regions))
    width = 0.27
    variant_color = PALETTE.get(tag, "#d62728")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    def _bar(ax, col_h, col_p, col_v, ylabel, title):
        ax.bar(x - width, df[col_h], width, label="Heuristic best",   color=PALETTE["heur"])
        ax.bar(x,         df[col_p], width, label="기준 PPO (f3)",     color=PALETTE["PPO"])
        ax.bar(x + width, df[col_v], width, label=tag,                 color=variant_color)
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="lower right", ncol=3); ax.grid(axis="y", alpha=0.3)

    def _delta(ax, dh, dp, ylabel, title):
        ax.axhline(0, color="#888", linewidth=1)
        ax.plot(x, dh, "o-", label=f"{tag} - Heur",   color=variant_color)
        ax.plot(x, dp, "s-", label=f"{tag} - 기준PPO", color="#444")
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="best"); ax.grid(axis="y", alpha=0.3)

    _bar(axes[0, 0], "heuristic_R",     "PPO_R_baseline",     f"{tag}_R",
         "mean reward",     "Mean reward (R, Green 포함)")
    _bar(axes[0, 1], "heuristic_R_woG", "PPO_R_woG_baseline", f"{tag}_R_woG",
         "mean reward woG", "Mean reward (R_woG, Green 제외)")
    _delta(axes[1, 0],
           df[f"{tag}_vs_heur"],
           df[f"{tag}_vs_baseline_PPO"],
           "Δ reward", "RL 우위 폭 (R)")
    _delta(axes[1, 1],
           df[f"{tag}_R_woG"] - df["heuristic_R_woG"],
           df[f"{tag}_R_woG"] - df["PPO_R_woG_baseline"],
           "Δ reward woG", "RL 우위 폭 (R_woG)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}")


def plot_extra_eval(extra_csv: str, f3_csv: str, out_path: str,
                    extras=("a2c", "trpo", "qrdqn", "recurrentppo")):
    """추가 SB3 알고리즘 4종 + heuristic + PPO(f3 기준) 2x2 컨벤션 PNG.

    Parameters
    ----------
    extra_csv : str
        eval_extra_algos.py 가 만든 CSV. 컬럼: region, heuristic_R, PPO_R,
        DQN_R, REINFORCE_R, <algo>_R, <algo>_R_woG, <algo>_vs_heur.
    f3_csv : str
        woG 베이스 컬럼 (heuristic_R_woG, PPO_R_woG) 출처.
    """
    _set_korean_font()
    df = pd.read_csv(extra_csv, encoding="utf-8-sig")
    f3 = pd.read_csv(f3_csv, encoding="utf-8-sig").set_index("region")
    df["heuristic_R_woG"] = df["region"].map(f3["heuristic_R_woG"])
    df["PPO_R_woG"] = df["region"].map(f3["PPO_R_woG"])

    regions = df["region"].tolist()
    x = np.arange(len(regions))
    # heur + PPO(ref) + 4 extras = 6 bars
    n_bars = 2 + len(extras)
    width = 0.78 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    def _bar(ax, suffix, ylabel, title):
        ax.bar(x + offsets[0], df[f"heuristic_R{suffix}"], width,
               label="Heuristic best", color=PALETTE["heur"])
        ax.bar(x + offsets[1], df[f"PPO_R{suffix}"], width,
               label="PPO (f3 기준)", color=PALETTE["PPO"])
        for i, algo in enumerate(extras):
            col = f"{algo}_R{suffix}"
            if col in df.columns:
                ax.bar(x + offsets[2 + i], df[col], width,
                       label=algo, color=PALETTE.get(algo))
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="lower right", ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    def _delta(ax, suffix, ylabel, title):
        ax.axhline(0, color="#888", linewidth=1)
        # 기준 PPO 차이도 참조선으로 그려둠
        ax.plot(x, df[f"PPO_R{suffix}"] - df[f"heuristic_R{suffix}"], "-",
                label="PPO (f3) - Heur", color=PALETTE["PPO"], linewidth=2, alpha=0.6)
        markers = ["o", "s", "^", "v"]
        for i, algo in enumerate(extras):
            col = f"{algo}_R{suffix}"
            if col in df.columns:
                ax.plot(x, df[col] - df[f"heuristic_R{suffix}"],
                        markers[i % len(markers)] + "-",
                        label=f"{algo} - Heur", color=PALETTE.get(algo))
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="best", fontsize=9); ax.grid(axis="y", alpha=0.3)

    _bar(axes[0, 0], "",     "mean reward",     "Mean reward (R, Green 포함)")
    _bar(axes[0, 1], "_woG", "mean reward woG", "Mean reward (R_woG, Green 제외)")
    _delta(axes[1, 0], "",     "Δ reward",     "RL 우위 폭 (R)")
    _delta(axes[1, 1], "_woG", "Δ reward woG", "RL 우위 폭 (R_woG)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {os.path.abspath(out_path)}")


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("variant", help="BC/Enriched 같은 단일 변형 PNG")
    pv.add_argument("--csv", required=True, help="variant eval CSV")
    pv.add_argument("--f3_csv", default="results/plan1nat_f3_eval.csv")
    pv.add_argument("--tag", required=True, help="컬럼 prefix (예: BC_PPO)")
    pv.add_argument("--out", required=True)

    pe = sub.add_parser("extra", help="A2C/TRPO/QRDQN/RecurrentPPO PNG")
    pe.add_argument("--csv", required=True, help="eval_extra_algos CSV")
    pe.add_argument("--f3_csv", default="results/plan1nat_f3_eval.csv")
    pe.add_argument("--out", required=True)

    a = p.parse_args()
    if a.mode == "variant":
        plot_variant_eval(a.csv, a.f3_csv, a.tag, a.out)
    elif a.mode == "extra":
        plot_extra_eval(a.csv, a.f3_csv, a.out)


if __name__ == "__main__":
    _cli()
