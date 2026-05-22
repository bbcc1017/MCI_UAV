"""REINFORCE 알고리즘 분리 실험 평가 (피드백 3 후속).

목적: plan1nat_f3 에서 REINFORCE 가 악화(-3.93 → -9.41)한 원인이 알고리즘
변경(v1 순수 REINFORCE → v2 actor-critic+엔트로피) 때문인지, obs 축소 때문인지
가른다. obs 축소는 v2 와 동일하게 유지하고 알고리즘만 v1 으로 재학습한 모델을
v2(f3) 와 같은 조건(17지역 plan1nat 시나리오, reduced obs)에서 평가한다.

비교 대상:
  - 휴리스틱 best, v2 REINFORCE  : results/plan1nat_f3_eval.csv 에서 읽음
  - v1 REINFORCE                 : 이 스크립트가 새로 평가

판정:
  v1 평균 R > v2(f3) 평균 R  → 알고리즘 변경이 악화 원인 → reinforce_agent.py 원복
  그렇지 않으면                → 알고리즘 무죄(obs 축소 영향) → v2 유지 검토

reduced obs 모델이므로 MCI_REDUCED_OBS=1 로 실행해야 한다.

예:
  MCI_REDUCED_OBS=1 python src/rl_src/eval_reinforce_v1.py \
    --model results/rl/plan1nat_f3_reinV1/national/reinforce/final_model.pt
"""
import argparse
import json
import os
import sys

# sim_src 디버그 print 억제
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sim_src"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

import pandas as pd

from evaluate import eval_policy, reinforce_policy, make_eval_env
from reinforce_agent_v1 import ReinforceAgent as ReinforceAgentV1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="v1 REINFORCE final_model.pt")
    p.add_argument("--manifest", default="scenarios/plan1nat_manifest.json")
    p.add_argument("--f3_csv", default="results/plan1nat_f3_eval.csv",
                   help="휴리스틱·v2 REINFORCE 기준값 출처")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out_csv", default="results/reinforce_v1v2_eval.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if os.environ.get("MCI_REDUCED_OBS") != "1":
        log("⚠️ MCI_REDUCED_OBS=1 미설정 — v1 모델은 reduced obs 로 학습됨. 중단.")
        raise SystemExit(1)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    f3 = pd.read_csv(args.f3_csv).set_index("region")

    agent = ReinforceAgentV1.load(args.model)
    log(f"[v1] 모델 로드: {args.model}  obs_dim={agent.obs_dim}")

    rows = []
    for region, cfg in manifest.items():
        m = eval_policy(make_eval_env(cfg), reinforce_policy(agent),
                        args.n_episodes, args.seed)
        heur_R = float(f3.loc[region, "heuristic_R"])
        v2_R = float(f3.loc[region, "REINFORCE_R"])
        rows.append({
            "region": region,
            "heuristic_R": heur_R,
            "REINFORCE_v2_R": v2_R,          # f3 (actor-critic 개정판)
            "REINFORCE_v1_R": m["mean_R"],   # 순수 REINFORCE
            "v1_vs_heur": m["mean_R"] - heur_R,
            "v2_vs_heur": v2_R - heur_R,
            "v1_minus_v2": m["mean_R"] - v2_R,
        })
        log(f"[{region}] v1_R={m['mean_R']:.3f}  v2_R={v2_R:.3f}  "
            f"heur={heur_R:.3f}  v1-v2={m['mean_R']-v2_R:+.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    v1_mean = df["REINFORCE_v1_R"].mean()
    v2_mean = df["REINFORCE_v2_R"].mean()
    log("\n" + "=" * 60)
    log(f"v1(순수 REINFORCE) 평균 R = {v1_mean:.3f}  (vs heur {df['v1_vs_heur'].mean():+.3f})")
    log(f"v2(f3 actor-critic) 평균 R = {v2_mean:.3f}  (vs heur {df['v2_vs_heur'].mean():+.3f})")
    log(f"v1 - v2 = {v1_mean - v2_mean:+.3f}  (지역별 v1 우세 {int((df['v1_minus_v2']>0).sum())}/{len(df)})")
    if v1_mean > v2_mean:
        log("판정 → v1 이 우세. 알고리즘 변경이 악화 원인 → reinforce_agent.py 원복 권장.")
    else:
        log("판정 → v2 가 우세하거나 동등. 알고리즘 변경은 무죄 → v2 유지 검토.")
    log(f"CSV: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
