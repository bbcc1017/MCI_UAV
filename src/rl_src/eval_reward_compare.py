"""woG 보상 학습 효과 검증 — 휴리스틱-best / f3-champion(R학습) / woG-model 을
17지역에서 R(Green포함)·R_woG(Green제외)·PDR_woG 로 동시 평가·비교.

가설: Green(50%·생존1.0 고정)이 총보상 R 을 띄워 개선을 가린다. woG 로 학습하면
시간-위중(Red/Yellow) 집중 → woG 지표에서 휴리스틱 대비 마진이 더 커지는가?

평가 env 는 표준(make_eval_env, 보상 wrapper 없음)이라 eval_policy 가 R 과 R_woG 를
모두 정확히 계산한다 (모델 학습 보상과 무관).

사용:
  MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/eval_reward_compare.py \
    --manifest scenarios/plan1nat_manifest.json --heur_csv results/plan1nat_f3_eval.csv \
    --f3_model results/rl/plan1nat_f3/national/ppo/final_model.zip \
    --wog_model results/rl/plan1nat_f3_woG/national/ppo/final_model.zip --n_episodes 100
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import eval_policy, make_eval_env, ppo_policy
from distill_policy import make_heuristic_policy
from plot_variant_eval import _set_korean_font


@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as dn:
        old = sys.stdout
        sys.stdout = dn
        try:
            yield
        finally:
            sys.stdout = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--heur_csv", required=True)
    ap.add_argument("--f3_model", required=True)
    ap.add_argument("--wog_model", required=True)
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--out_csv", default="results/plan1nat_woG_compare.csv")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    f3 = MaskablePPO.load(args.f3_model)
    wog = MaskablePPO.load(args.wog_model)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    best_rule = dict(zip(*[pd.read_csv(args.heur_csv)[c] for c in ("region", "heuristic_rule")]))

    rows = []
    for ri, (region, cfg) in enumerate(manifest.items()):
        if region not in best_rule:
            continue
        sys.stderr.write(f"[{ri+1}/{len(manifest)}] {region} ...\n"); sys.stderr.flush()
        ef = make_eval_env(cfg)
        with _silence():
            mh = eval_policy(ef, make_heuristic_policy(best_rule[region]), args.n_episodes, args.seed_base)
            m3 = eval_policy(ef, ppo_policy(f3), args.n_episodes, args.seed_base)
            mw = eval_policy(ef, ppo_policy(wog), args.n_episodes, args.seed_base)
        rows.append({
            "region": region,
            # R (Green 포함)
            "heur_R": mh["mean_R"], "f3_R": m3["mean_R"], "woG_R": mw["mean_R"],
            # R_woG (Green 제외 — 임상적 핵심)
            "heur_RwoG": mh["mean_R_woG"], "f3_RwoG": m3["mean_R_woG"], "woG_RwoG": mw["mean_R_woG"],
            # PDR_woG (예방가능 사망률, 낮을수록 좋음)
            "heur_PDRwoG": mh["mean_PDR_woG"], "f3_PDRwoG": m3["mean_PDR_woG"], "woG_PDRwoG": mw["mean_PDR_woG"],
        })
        r = rows[-1]
        r["f3_vs_heur_R"] = r["f3_R"] - r["heur_R"]
        r["woG_vs_heur_R"] = r["woG_R"] - r["heur_R"]
        r["f3_vs_heur_RwoG"] = r["f3_RwoG"] - r["heur_RwoG"]
        r["woG_vs_heur_RwoG"] = r["woG_RwoG"] - r["heur_RwoG"]
        r["woG_vs_f3_RwoG"] = r["woG_RwoG"] - r["f3_RwoG"]
        sys.stderr.write(f"    R_woG: heur={r['heur_RwoG']:.2f} f3={r['f3_RwoG']:.2f} woG={r['woG_RwoG']:.2f} "
                         f"(woG vs heur {r['woG_vs_heur_RwoG']:+.2f}, vs f3 {r['woG_vs_f3_RwoG']:+.2f})\n")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n=== woG 학습 효과 (17지역, n_episodes={}) ===".format(args.n_episodes))
    print("\n[R_woG = Green 제외, 임상적 핵심 지표]")
    print(f"  휴리스틱 평균 R_woG = {df['heur_RwoG'].mean():.2f}")
    print(f"  f3(R학습)  vs 휴리스틱: {df['f3_vs_heur_RwoG'].mean():+.3f}  ({(df['f3_vs_heur_RwoG']>0).sum()}/{len(df)} 추월)")
    print(f"  woG학습     vs 휴리스틱: {df['woG_vs_heur_RwoG'].mean():+.3f}  ({(df['woG_vs_heur_RwoG']>0).sum()}/{len(df)} 추월)")
    print(f"  woG학습     vs f3      : {df['woG_vs_f3_RwoG'].mean():+.3f}  ({(df['woG_vs_f3_RwoG']>0).sum()}/{len(df)} 우세)")
    print("\n[R = Green 포함, 기존 지표]")
    print(f"  f3  vs 휴리스틱: {df['f3_vs_heur_R'].mean():+.3f}  ({(df['f3_vs_heur_R']>0).sum()}/{len(df)})")
    print(f"  woG vs 휴리스틱: {df['woG_vs_heur_R'].mean():+.3f}  ({(df['woG_vs_heur_R']>0).sum()}/{len(df)})")
    print("\n[PDR_woG = 예방가능 사망률(낮을수록 좋음)]")
    print(f"  휴리스틱={df['heur_PDRwoG'].mean():.4f}  f3={df['f3_PDRwoG'].mean():.4f}  woG={df['woG_PDRwoG'].mean():.4f}")

    # 그림: woG 지표에서 마진 비교
    _set_korean_font()
    dd = df.sort_values("woG_vs_heur_RwoG", ascending=False)
    x = np.arange(len(dd)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].bar(x - w/2, dd["f3_vs_heur_RwoG"], w, label="f3 (R 학습)", color="#1f77b4")
    ax[0].bar(x + w/2, dd["woG_vs_heur_RwoG"], w, label="woG 학습", color="#d62728")
    ax[0].axhline(0, color="#888"); ax[0].set_xticks(x); ax[0].set_xticklabels(dd["region"], rotation=45)
    ax[0].set_ylabel("Δ R_woG vs 휴리스틱"); ax[0].legend()
    ax[0].set_title(f"(a) Green제외 지표 마진 | f3 {df['f3_vs_heur_RwoG'].mean():+.2f} vs woG {df['woG_vs_heur_RwoG'].mean():+.2f}")
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(x - w/2, dd["f3_vs_heur_R"], w, label="f3 (R 학습)", color="#1f77b4")
    ax[1].bar(x + w/2, dd["woG_vs_heur_R"], w, label="woG 학습", color="#d62728")
    ax[1].axhline(0, color="#888"); ax[1].set_xticks(x); ax[1].set_xticklabels(dd["region"], rotation=45)
    ax[1].set_ylabel("Δ R vs 휴리스틱"); ax[1].legend()
    ax[1].set_title(f"(b) Green포함 지표 마진 | f3 {df['f3_vs_heur_R'].mean():+.2f} vs woG {df['woG_vs_heur_R'].mean():+.2f}")
    ax[1].grid(axis="y", alpha=0.3)

    plt.suptitle("woG 보상 학습 효과 — Green 제외 지표에서 마진이 더 커지는가?", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = "results/analysis/fig_woG_compare.png"
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"\n[저장] {args.out_csv}, {out_png}")


if __name__ == "__main__":
    main()
