"""추가 SB3 알고리즘(A2C/TRPO/QR-DQN/RecurrentPPO) 17지역 평가 (피드백 3 후속).

train_sb3_extra.py 로 학습한 4종 모델을 plan1nat 17지역 시나리오에서 평가하고,
plan1nat_f3_eval.csv 의 휴리스틱·PPO·DQN·REINFORCE 와 같은 표에 합쳐 비교한다.

4종 모두 action masking 비사용. 평가 루프는 model.predict 의 (state, episode_start)
인자를 써서 RecurrentPPO 의 LSTM 상태를 에피소드 경계에서 초기화한다 — 비순환
모델은 이 인자를 무시하므로 동일 루프로 처리된다.

reduced obs 모델이므로 MCI_REDUCED_OBS=1 로 실행해야 한다.

예:
  MCI_REDUCED_OBS=1 python src/rl_src/eval_extra_algos.py \
    --model_root results/rl/plan1nat_f3_extra/national
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

from evaluate import make_eval_env

_ALGO_LOADERS = {
    "a2c":          ("stable_baselines3", "A2C"),
    "trpo":         ("sb3_contrib", "TRPO"),
    "qrdqn":        ("sb3_contrib", "QRDQN"),
    "recurrentppo": ("sb3_contrib", "RecurrentPPO"),
}


def load_model(algo, path):
    mod, cls = _ALGO_LOADERS[algo]
    Model = getattr(__import__(mod, fromlist=[cls]), cls)
    return Model.load(path)


def eval_model(model, env_factory, n_episodes, seed_base):
    """model.predict 기반 평가 — RecurrentPPO LSTM 상태를 에피소드마다 초기화."""
    rewards, rewards_woG, pdrs, pdrs_woG, times = [], [], [], [], []
    for ep in range(n_episodes):
        env = env_factory(seed=seed_base + ep)
        obs, _ = env.reset(seed=seed_base + ep)
        state = None
        ep_start = np.ones((1,), dtype=bool)
        done = False
        ep_r = ep_r_woG = 0.0
        while not done:
            action, state = model.predict(obs, state=state,
                                          episode_start=ep_start,
                                          deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r
            ep_r_woG += info.get("r_woG", 0.0)
            done = term or trunc
        prev = env.unwrapped.preventable
        prev_woG = env.unwrapped.preventable_woG
        rewards.append(ep_r)
        rewards_woG.append(ep_r_woG)
        pdrs.append(1 - ep_r / prev if prev > 0 else 0.0)
        pdrs_woG.append(1 - ep_r_woG / prev_woG if prev_woG > 0 else 0.0)
        times.append(info["time"])
    a = lambda x: np.array(x)
    return {
        "mean_R": float(a(rewards).mean()),
        "mean_R_woG": float(a(rewards_woG).mean()),
        "mean_PDR": float(a(pdrs).mean()),
        "mean_PDR_woG": float(a(pdrs_woG).mean()),
        "mean_time": float(a(times).mean()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_root", default="results/rl/plan1nat_f3_extra/national")
    p.add_argument("--manifest", default="scenarios/manifests/plan1nat_manifest.json")
    p.add_argument("--f3_csv", default="results/plan1nat_f3_eval.csv")
    p.add_argument("--algos", nargs="+", default=list(_ALGO_LOADERS),
                   choices=list(_ALGO_LOADERS))
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out_csv", default="results/plan1nat_f3_extra_eval.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if os.environ.get("MCI_REDUCED_OBS") != "1":
        log("⚠️ MCI_REDUCED_OBS=1 미설정 — 학습과 obs 차원 불일치. 중단.")
        raise SystemExit(1)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    f3 = pd.read_csv(args.f3_csv).set_index("region")

    models = {}
    for algo in args.algos:
        mp = os.path.join(args.model_root, algo, "final_model.zip")
        if not os.path.exists(mp):
            log(f"⚠️ {algo}: 모델 미발견 {mp} — 건너뜀")
            continue
        models[algo] = load_model(algo, mp)
        log(f"[load] {algo} ← {mp}")

    rows = []
    for region, cfg in manifest.items():
        row = {"region": region,
               "heuristic_R": float(f3.loc[region, "heuristic_R"]),
               "PPO_R": float(f3.loc[region, "PPO_R"]),
               "DQN_R": float(f3.loc[region, "DQN_R"]),
               "REINFORCE_R": float(f3.loc[region, "REINFORCE_R"])}
        for algo, model in models.items():
            m = eval_model(model, make_eval_env(cfg), args.n_episodes, args.seed)
            row[f"{algo}_R"] = m["mean_R"]
            row[f"{algo}_R_woG"] = m["mean_R_woG"]
            row[f"{algo}_vs_heur"] = m["mean_R"] - row["heuristic_R"]
            log(f"[{region}] {algo:13s} R={m['mean_R']:.3f}  "
                f"vs_heur={m['mean_R']-row['heuristic_R']:+.3f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    log("\n" + "=" * 64)
    log(f"{'algo':14s} {'평균R':>9} {'vs휴리스틱':>11} {'휴리스틱추월':>12}")
    base = {"heuristic": "heuristic_R", "PPO(f3)": "PPO_R",
            "DQN(f3)": "DQN_R", "REINFORCE(f3)": "REINFORCE_R"}
    for name, col in base.items():
        d = df[col] - df["heuristic_R"]
        log(f"{name:14s} {df[col].mean():9.3f} {d.mean():+11.3f} "
            f"{int((d>0).sum()):>10d}/{len(df)}")
    for algo in models:
        d = df[f"{algo}_vs_heur"]
        log(f"{algo:14s} {df[f'{algo}_R'].mean():9.3f} {d.mean():+11.3f} "
            f"{int((d>0).sum()):>10d}/{len(df)}")
    log(f"\nCSV: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
