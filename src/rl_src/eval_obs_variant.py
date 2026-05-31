"""obs_reduced v2 평가 — 한 obs variant 모델을 17지역에서 휴리스틱 대비 R·R_woG 로 평가.

MCI_OBS_VARIANT 를 import 전에 환경변수로 설정해 평가 env 의 obs 차원을 학습 모델과
일치시킨다(불일치 시 MaskablePPO.predict 가 shape 에러). base(f3) 평가는 --variant ""
로 같은 스크립트를 돌리면 동일 코드·조건 결과가 나와 공정 비교 가능.

평가 env 는 표준(보상 wrapper 없음) → eval_policy 가 R·R_woG·PDR_woG 를 참 보상으로 계산.
휴리스틱은 obs 와 무관(rule 이 raw dict obs 사용)하므로 variant 와 상관없이 동일.

사용:
  MCI_REDUCED_OBS=1 CUDA_VISIBLE_DEVICES="" python src/rl_src/eval_obs_variant.py \
    --variant "idle+eta" --model results/rl/plan1nat_obsv2_idle_eta/national/ppo/final_model.zip \
    --manifest scenarios/plan1nat_manifest.json --heur_csv results/plan1nat_f3_eval.csv \
    --tag idle_eta --n_episodes 100
"""
import argparse
import os
import sys

# --- variant 를 import 전에 환경변수로 (obs 차원 결정) ---
_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", default="", help="MCI_OBS_VARIANT 값 (base 면 빈 문자열)")
_ap.add_argument("--model", required=True)
_ap.add_argument("--manifest", required=True)
_ap.add_argument("--heur_csv", required=True)
_ap.add_argument("--tag", required=True)
_ap.add_argument("--n_episodes", type=int, default=100)
_ap.add_argument("--seed_base", type=int, default=2000)
_ap.add_argument("--out_csv", default=None)
_ap.add_argument("--vecnorm_path", default=None, help="VecNormalize pkl (obs 표준화 통계). 있으면 평가 obs 에 적용.")
args = _ap.parse_args()

os.environ["MCI_REDUCED_OBS"] = "1"
if args.variant and args.variant.lower() != "base":
    os.environ["MCI_OBS_VARIANT"] = args.variant
else:
    os.environ.pop("MCI_OBS_VARIANT", None)

sys.path.insert(0, os.path.dirname(__file__))

import contextlib
import json
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
    out_csv = args.out_csv or f"results/analysis/obsv2_eval_{args.tag}.csv"

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)

    # vecnorm 통계 적용 정책 (학습 시 obs 표준화했으면 평가 obs 도 동일 변환)
    if args.vecnorm_path:
        import pickle
        with open(args.vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        mean = vn.obs_rms.mean.astype(np.float32)
        std = np.sqrt(vn.obs_rms.var + vn.epsilon).astype(np.float32)
        clip = float(vn.clip_obs)
        def policy_fn(obs, mask, env_unwrapped):
            o = np.clip((np.asarray(obs, np.float32) - mean) / std, -clip, clip)
            a, _ = model.predict(o, action_masks=mask, deterministic=True)
            return int(a)
        sys.stderr.write(f"[vecnorm] {args.vecnorm_path} 통계로 평가 obs 표준화\n")
    else:
        policy_fn = ppo_policy(model)

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
            mm = eval_policy(ef, policy_fn, args.n_episodes, args.seed_base)
        rows.append({
            "region": region,
            "heur_R": mh["mean_R"], "model_R": mm["mean_R"],
            "heur_RwoG": mh["mean_R_woG"], "model_RwoG": mm["mean_R_woG"],
            "heur_PDRwoG": mh["mean_PDR_woG"], "model_PDRwoG": mm["mean_PDR_woG"],
            "vs_heur_R": mm["mean_R"] - mh["mean_R"],
            "vs_heur_RwoG": mm["mean_R_woG"] - mh["mean_R_woG"],
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    n = len(df)
    print(f"\n=== obs variant '{args.variant or 'base'}' ({n}지역, n_ep={args.n_episodes}) ===")
    print(f"  vs 휴리스틱 R    : {df['vs_heur_R'].mean():+.3f}  ({(df['vs_heur_R']>0).sum()}/{n})")
    print(f"  vs 휴리스틱 R_woG: {df['vs_heur_RwoG'].mean():+.3f}  ({(df['vs_heur_RwoG']>0).sum()}/{n})")
    print(f"  PDR_woG: heur={df['heur_PDRwoG'].mean():.4f}  model={df['model_PDRwoG'].mean():.4f}")
    print(f"[저장] {out_csv}")


if __name__ == "__main__":
    main()
