"""멀티시드 재현 결과 집계 — repro_seed*.json 을 읽어 발견 재현 여부를 표·그림으로.

발견별 재현 기준:
  · 이송수단 규칙(AMB우선): mode_rule_acc ≥ 0.95
  · 적응형 우선순위: priority_start_agree 가 START(=1.0)와 뚜렷이 다름(<0.7)
  · 부하분산 라우팅: route_occ_gap < 0 (RL이 덜 혼잡한 병원)
  · 증류 정책: distill_vs_heur > 0, 17지역 다수 추월

사용:  CUDA_VISIBLE_DEVICES="" python src/rl_src/aggregate_seeds.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_variant_eval import _set_korean_font

ADIR = "results/analysis"


def main():
    _set_korean_font()
    files = sorted(glob.glob(os.path.join(ADIR, "repro_seed*.json")))
    rows = [json.load(open(f, encoding="utf-8")) for f in files]
    if not rows:
        print("repro_seed*.json 없음", file=sys.stderr); return
    df = pd.DataFrame([{
        "seed": r["seed"],
        "mode_rule_acc": r["mode_rule_acc"],
        "priority_start_agree": r["priority_start_agree"],
        "priority_red_ratio": r["priority_red_ratio"],
        "route_occ_gap": r["route_occ_gap"],
        "route_tier3_rl": r["route_tier3_rl"],
        "route_tier3_heur": r["route_tier3_heur"],
        "ppo_vs_heur": r["ppo_vs_heur"],
        "distill_vs_heur": r["distill_vs_heur"],
        "distill_regions_won": r["distill_regions_won"],
        "distill_retention_pct": r["distill_retention_pct"],
    } for r in rows]).sort_values("seed").reset_index(drop=True)

    metric_cols = ["mode_rule_acc", "priority_start_agree", "route_occ_gap",
                   "route_tier3_rl", "ppo_vs_heur", "distill_vs_heur",
                   "distill_regions_won", "distill_retention_pct"]
    summ = df[metric_cols].agg(["mean", "std"]).T
    print(f"=== 멀티시드 재현 ({len(df)} 시드: {list(df['seed'])}) ===\n")
    print(df.to_string(index=False))
    print("\n--- 평균 ± 표준편차 ---")
    for m in metric_cols:
        print(f"  {m:24s} {summ.loc[m,'mean']:+.3f} ± {summ.loc[m,'std']:.3f}")

    # 재현 판정
    print("\n--- 발견 재현 판정 ---")
    print(f"  이송수단 규칙(AMB우선)  : mode_rule_acc 모두≥0.95? {(df['mode_rule_acc']>=0.95).all()}")
    print(f"  적응형 우선순위(≠START): priority_start_agree 모두<0.7? {(df['priority_start_agree']<0.7).all()}")
    print(f"  부하분산 라우팅        : route_occ_gap 모두<0? {(df['route_occ_gap']<0).all()}")
    print(f"  증류>휴리스틱          : distill_vs_heur 모두>0? {(df['distill_vs_heur']>0).all()} "
          f"(평균 추월지역 {df['distill_regions_won'].mean():.1f}/17)")

    df.to_csv(os.path.join(ADIR, "multiseed_summary.csv"), index=False)

    # 그림 2x2
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    s = df["seed"].astype(str)
    ax[0, 0].bar(s, df["mode_rule_acc"], color="#54a24b")
    ax[0, 0].axhline(0.95, ls="--", color="#888"); ax[0, 0].set_ylim(0.8, 1.01)
    ax[0, 0].set_title("(a) 이송수단 규칙 'AMB우선' 일치율 (≥0.95)")
    ax[0, 0].set_xlabel("seed")

    ax[0, 1].bar(s, df["route_occ_gap"], color="#1f77b4")
    ax[0, 1].axhline(0, color="#888")
    ax[0, 1].set_title("(b) 라우팅 점유율 격차 RL−휴리스틱 (<0 = 부하분산)")
    ax[0, 1].set_xlabel("seed")

    ax[1, 0].bar(s, df["priority_start_agree"], color="#e45756")
    ax[1, 0].axhline(1.0, ls="--", color="#888", label="START(=1.0)")
    ax[1, 0].set_ylim(0, 1.05); ax[1, 0].legend()
    ax[1, 0].set_title("(c) 우선순위 START 일치율 (<1 = 적응형)")
    ax[1, 0].set_xlabel("seed")

    w = 0.38
    x = np.arange(len(df))
    ax[1, 1].bar(x - w/2, df["ppo_vs_heur"], w, label="풀 PPO", color="#1f77b4")
    ax[1, 1].bar(x + w/2, df["distill_vs_heur"], w, label="증류 규칙", color="#d62728")
    ax[1, 1].axhline(0, color="#888")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(s)
    ax[1, 1].set_title("(d) 휴리스틱 대비 우위 (Δ보상)"); ax[1, 1].legend()
    ax[1, 1].set_xlabel("seed")

    plt.suptitle(f"멀티시드 재현성 — MaskablePPO 정책 해석+증류 ({len(df)} 시드)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ADIR, "fig_multiseed_repro.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n[저장] {out}, multiseed_summary.csv")


if __name__ == "__main__":
    main()
