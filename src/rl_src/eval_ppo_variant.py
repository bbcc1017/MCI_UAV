"""단일 MaskablePPO 모델을 17지역에서 평가, 휴리스틱·기준 PPO 와 비교.

피드백5 학습 변형(BC pretrain, enriched env) 평가용. 기존 eval_region.py 는
ppo/dqn/reinforce 3종을 고정 로드라 단일 모델 + 옵션 wrapper 변형을 다루지
못해 별도로 둔다.

--enriched 면 EnrichedObsMaskWrapper(+AggregateObsWrapper if MCI_REDUCED_OBS=1)
를 직접 끼워 env 를 만든다. baseline plan1nat_f3_eval.csv 의 heuristic_R 과
PPO_R 을 동일 표에 합쳐 비교.

예:
  MCI_REDUCED_OBS=1 python src/rl_src/eval_ppo_variant.py \
    --model_path results/rl/plan1nat_f5/national/ppo_bc/final_model.zip \
    --tag BC_PPO --out_csv results/plan1nat_f5_bc_eval.csv

  MCI_REDUCED_OBS=1 python src/rl_src/eval_ppo_variant.py \
    --model_path results/rl/plan1nat_f5/national/ppo_enriched/final_model.zip \
    --enriched --topk 10 --tag EnrichedPPO \
    --out_csv results/plan1nat_f5_enriched_eval.csv
"""
import argparse
import json
import os
import sys

_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sim_src"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper
from sb3_contrib import MaskablePPO


def make_env_factory(config_path: str, *, enriched: bool, topk):
    """평가용 env factory — enriched 면 EnrichedObsMaskWrapper, 아니면 표준."""
    if enriched:
        from enriched_env_wrapper import EnrichedObsMaskWrapper
    def _f(seed=0):
        base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=True)
        if enriched:
            return EnrichedObsMaskWrapper(base, topk=topk)
        return FlattenAndDiscreteWrapper(base)
    return _f


def eval_one_region(model, env_factory, n_episodes, seed_base):
    rs, rs_woG, pdrs, pdrs_woG, times = [], [], [], [], []
    for ep in range(n_episodes):
        env = env_factory(seed=seed_base + ep)
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_r = ep_r_woG = 0.0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r
            ep_r_woG += info.get("r_woG", 0.0)
            done = term or trunc
        prev = env.unwrapped.preventable
        prev_woG = env.unwrapped.preventable_woG
        rs.append(ep_r); rs_woG.append(ep_r_woG)
        pdrs.append(1 - ep_r / prev if prev > 0 else 0.0)
        pdrs_woG.append(1 - ep_r_woG / prev_woG if prev_woG > 0 else 0.0)
        times.append(info["time"])
    a = np.array
    return {
        "mean_R": float(a(rs).mean()), "mean_R_woG": float(a(rs_woG).mean()),
        "mean_PDR": float(a(pdrs).mean()), "mean_PDR_woG": float(a(pdrs_woG).mean()),
        "mean_time": float(a(times).mean()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--manifest", default="scenarios/manifests/plan1nat_manifest.json")
    p.add_argument("--f3_csv", default="results/plan1nat_f3_eval.csv",
                   help="휴리스틱·기준 PPO 비교를 위한 기존 평가 CSV")
    p.add_argument("--enriched", action="store_true")
    p.add_argument("--topk", type=int, default=10,
                   help="enriched 평가 시 학습과 동일하게 (기본 10)")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--tag", default="variant",
                   help="결과 CSV 컬럼명 prefix (예: BC_PPO → BC_PPO_R)")
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if os.environ.get("MCI_REDUCED_OBS") != "1":
        log("⚠️ MCI_REDUCED_OBS=1 미설정 — plan1nat_f3 학습 모델 obs 와 차원 불일치 가능. 계속.")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    f3 = pd.read_csv(args.f3_csv).set_index("region")

    model = MaskablePPO.load(args.model_path)
    log(f"[load] {args.tag} ← {args.model_path}  obs={model.observation_space.shape}  "
        f"actions={model.action_space.n}  enriched={args.enriched}")

    rows = []
    for region, cfg in manifest.items():
        m = eval_one_region(model, make_env_factory(cfg, enriched=args.enriched,
                                                    topk=args.topk),
                            args.n_episodes, args.seed)
        heur_R = float(f3.loc[region, "heuristic_R"])
        base_ppo_R = float(f3.loc[region, "PPO_R"])
        rows.append({
            "region": region,
            "heuristic_R": heur_R,
            "PPO_R_baseline": base_ppo_R,
            f"{args.tag}_R": m["mean_R"],
            f"{args.tag}_R_woG": m["mean_R_woG"],
            f"{args.tag}_PDR": m["mean_PDR"],
            f"{args.tag}_PDR_woG": m["mean_PDR_woG"],
            f"{args.tag}_time": m["mean_time"],
            f"{args.tag}_vs_heur": m["mean_R"] - heur_R,
            f"{args.tag}_vs_baseline_PPO": m["mean_R"] - base_ppo_R,
        })
        log(f"[{region}] R={m['mean_R']:.3f}  vs_heur={m['mean_R']-heur_R:+.3f}  "
            f"vs_basePPO={m['mean_R']-base_ppo_R:+.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    log("\n" + "=" * 70)
    log(f"{args.tag} 평균 R = {df[f'{args.tag}_R'].mean():.3f}")
    log(f"  vs 휴리스틱 평균 Δ = {df[f'{args.tag}_vs_heur'].mean():+.3f}  "
        f"({int((df[f'{args.tag}_vs_heur']>0).sum())}/{len(df)} 지역 추월)")
    log(f"  vs 기준 PPO   평균 Δ = {df[f'{args.tag}_vs_baseline_PPO'].mean():+.3f}  "
        f"({int((df[f'{args.tag}_vs_baseline_PPO']>0).sum())}/{len(df)} 지역 우세)")
    log(f"\n  기준 PPO    평균 R = {df['PPO_R_baseline'].mean():.3f} (vs 휴리스틱 +1.605)")
    log(f"  휴리스틱    평균 R = {df['heuristic_R'].mean():.3f}")
    log(f"CSV: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
