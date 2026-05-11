"""휴리스틱 룰 평균 reward를 TensorBoard scalar로 기록.

목적: RL 학습 곡선과 휴리스틱 best baseline을 같은 TB 화면에서 비교.

각 룰을 N ep 평가 → mean_R 계산 → best 룰 식별 → TB run 으로
`rollout/ep_rew_mean` 태그(SB3 학습 곡선과 동일 태그)로 평탄선 기록.

예:
    python src/rl_src/log_heuristic_baseline.py \
        --config_path scenarios/.../config.yaml \
        --n_episodes 1000 \
        --tb_dir results/rl \
        --max_step 200000
"""
import argparse
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from env_factory import make_base_env, load_config

SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from RuleManager import RuleManager  # noqa: E402


def evaluate_rule(rule, config_path: str, n_episodes: int, seed_base: int) -> float:
    rewards = []
    for ep in range(n_episodes):
        env = make_base_env(config_path, seed=seed_base + ep, rule_test=True, eval_mode=True)
        rule.init_with_scenario({"EntityManager": env.en_manager})
        rule.set_seed(np.random.default_rng(seed_base + ep))
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_r = 0.0
        while not done:
            a = rule.select(obs)
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    return float(np.mean(rewards))


def write_flat_line(tb_dir: str, run_name: str, value: float, max_step: int, tag: str):
    """TB run 폴더에 (0, max_step) 두 점으로 평탄선 기록."""
    run_path = os.path.join(tb_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    writer = SummaryWriter(log_dir=run_path)
    for step in (0, max_step):
        writer.add_scalar(tag, value, step)
    writer.close()
    return run_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--n_episodes", type=int, default=1000,
                   help="휴리스틱 룰 평가 episode 수 (기본 1000)")
    p.add_argument("--tb_dir", default="results/rl",
                   help="TB run 들이 기록될 부모 디렉터리 (학습용 logdir 와 동일하게)")
    p.add_argument("--max_step", type=int, default=200000,
                   help="평탄선 끝 x 좌표 (학습 total_timesteps 와 맞추면 시각적으로 깔끔)")
    p.add_argument("--seed_base", type=int, default=1000)
    p.add_argument("--tag", default="rollout/ep_rew_mean",
                   help="SB3 학습 곡선과 동일 태그면 같은 차트에 겹쳐 보임")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config_path)
    base = make_base_env(args.config_path, seed=args.seed_base, rule_test=True, eval_mode=True)
    rule_mgr = RuleManager(cfg["rule_info"], scenario={"EntityManager": base.en_manager},
                           rng=np.random.default_rng(args.seed_base))
    rules = rule_mgr.rules
    print(f"룰 {len(rules)}개 평가 — 각 {args.n_episodes} ep ...")

    results: List[Tuple[str, float]] = []
    for rule in rules:
        mean_r = evaluate_rule(rule, args.config_path, args.n_episodes, args.seed_base)
        print(f"  {rule.rule_name:<60s}  mean_R = {mean_r:.4f}")
        results.append((rule.rule_name, mean_r))

    # Best 식별
    best_name, best_r = max(results, key=lambda x: x[1])
    print()
    print(f"BEST : {best_name}  =>  {best_r:.4f}")

    os.makedirs(args.tb_dir, exist_ok=True)
    best_path = write_flat_line(args.tb_dir, "heuristic_best/tb",
                                best_r, args.max_step, args.tag)
    print(f"\nTB run 작성: {best_path}  (value={best_r:.4f})")
    print(f"TensorBoard --logdir {args.tb_dir} 로 띄우면 학습곡선과 함께 비교 가능.")


if __name__ == "__main__":
    main()
