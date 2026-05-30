"""woG 보상 학습의 멀티시드 확정 — seed0..N woG 모델을 17지역에서 평가하고
f3(R학습 챔피언)·휴리스틱 대비 R_woG·PDR_woG 마진을 시드별 + mean±std 로 집계.

배경: woG 단일시드(seed0)에서 woG vs f3 R_woG +0.56·13/17 우세였으나 폭이 modest.
멀티시드(seed0~3)로 이 우세와 충남 교정(f3 −0.32 → woG +3.27)이 강건한지 검증.

평가 env 는 표준(make_eval_env, 보상 wrapper 없음) → eval_policy 가 R·R_woG 를
참 보상으로 계산(모델 학습 보상과 무관). 휴리스틱·f3 는 1회만 평가해 공유.

사용:
  MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/eval_woG_multiseed.py \
    --manifest scenarios/plan1nat_manifest.json --heur_csv results/plan1nat_f3_eval.csv \
    --f3_model results/rl/plan1nat_f3/national/ppo/final_model.zip \
    --wog_models results/rl/plan1nat_f3_woG/national/ppo/final_model.zip \
                 results/rl/plan1nat_f3_woG_seed1/national/ppo/final_model.zip \
                 results/rl/plan1nat_f3_woG_seed2/national/ppo/final_model.zip \
                 results/rl/plan1nat_f3_woG_seed3/national/ppo/final_model.zip \
    --n_episodes 100
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from evaluate import eval_policy, make_eval_env, ppo_policy
from distill_policy import make_heuristic_policy


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
    ap.add_argument("--wog_models", nargs="+", required=True, help="woG 모델 경로 (seed0..N 순)")
    ap.add_argument("--wog_tags", nargs="+", default=None, help="각 woG 모델 라벨 (기본 s0,s1,...)")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--out_csv", default="results/analysis/plan1nat_woG_multiseed.csv")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    f3 = MaskablePPO.load(args.f3_model)
    wogs = [MaskablePPO.load(p) for p in args.wog_models]
    tags = args.wog_tags or [f"s{i}" for i in range(len(wogs))]
    assert len(tags) == len(wogs)

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
            mws = [eval_policy(ef, ppo_policy(w), args.n_episodes, args.seed_base) for w in wogs]
        row = {
            "region": region,
            "heur_RwoG": mh["mean_R_woG"], "f3_RwoG": m3["mean_R_woG"],
            "heur_PDRwoG": mh["mean_PDR_woG"], "f3_PDRwoG": m3["mean_PDR_woG"],
            "f3_vs_heur_RwoG": m3["mean_R_woG"] - mh["mean_R_woG"],
        }
        for tag, mw in zip(tags, mws):
            row[f"woG{tag}_RwoG"] = mw["mean_R_woG"]
            row[f"woG{tag}_PDRwoG"] = mw["mean_PDR_woG"]
            row[f"woG{tag}_vs_heur_RwoG"] = mw["mean_R_woG"] - mh["mean_R_woG"]
            row[f"woG{tag}_vs_f3_RwoG"] = mw["mean_R_woG"] - m3["mean_R_woG"]
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    n = len(df)
    print(f"\n=== woG 멀티시드 확정 ({n}지역, n_episodes={args.n_episodes}, woG seeds={tags}) ===")
    print(f"\n[기준] 휴리스틱 R_woG={df['heur_RwoG'].mean():.2f}  f3 R_woG={df['f3_RwoG'].mean():.2f} "
          f"(f3 vs heur {df['f3_vs_heur_RwoG'].mean():+.3f}, {(df['f3_vs_heur_RwoG']>0).sum()}/{n})")

    # 시드별 마진
    vs_heur = []
    vs_f3 = []
    print("\n[woG vs 휴리스틱 / woG vs f3 — R_woG 기준, 시드별]")
    for tag in tags:
        vh = df[f"woG{tag}_vs_heur_RwoG"]; vf = df[f"woG{tag}_vs_f3_RwoG"]
        vs_heur.append(vh.mean()); vs_f3.append(vf.mean())
        print(f"  woG[{tag}]: vs heur {vh.mean():+.3f} ({(vh>0).sum()}/{n}) | "
              f"vs f3 {vf.mean():+.3f} ({(vf>0).sum()}/{n})")
    vs_heur = np.array(vs_heur); vs_f3 = np.array(vs_f3)
    sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
    print(f"\n[시드 전체 mean±std]")
    print(f"  woG vs heur (R_woG): {vs_heur.mean():+.3f} ± {sd(vs_heur):.3f}")
    print(f"  woG vs f3   (R_woG): {vs_f3.mean():+.3f} ± {sd(vs_f3):.3f}")

    # 충남 교정 추적 (f3 가 유일하게 졌던 지역)
    if "충남" in df["region"].values:
        r = df[df["region"] == "충남"].iloc[0]
        print(f"\n[충남 — f3 가 휴리스틱에 졌던 지역] f3 vs heur {r['f3_vs_heur_RwoG']:+.2f}")
        for tag in tags:
            print(f"  woG[{tag}] vs heur {r[f'woG{tag}_vs_heur_RwoG']:+.2f}  vs f3 {r[f'woG{tag}_vs_f3_RwoG']:+.2f}")

    # PDR_woG (낮을수록 좋음)
    pdr_w = np.array([df[f"woG{tag}_PDRwoG"].mean() for tag in tags])
    print(f"\n[PDR_woG] 휴리스틱={df['heur_PDRwoG'].mean():.4f}  f3={df['f3_PDRwoG'].mean():.4f}  "
          f"woG={pdr_w.mean():.4f}±{sd(pdr_w):.4f}")
    print(f"\n[저장] {args.out_csv}")


if __name__ == "__main__":
    main()
