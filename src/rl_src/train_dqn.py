"""DQN 학습 스크립트.

표준 SB3 DQN 은 action mask 를 직접 지원하지 않는다.
대안:
  - sb3-contrib 의 MaskableDQN 은 아직 없음 (2.x 기준).
  - 따라서 평가시 mask 적용으로 illegal 액션 막고, 학습은 wrapper 가 illegal 액션을 자연 페널티(NO AMB/NO UAV → 자원 미사용 + 보상 없음)로 처리.

이 구현은 단순화를 위해 vanilla DQN 을 사용하되, predict 시 mask 를 적용하는 helper 를 제공.

예:
    python src/rl_src/train_dqn.py --config_path scenarios/.../config.yaml --total_timesteps 200000 --seed 0 --log_dir results/rl/dqn_seoul
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper, HybridAMBHeurWrapper


def make_env(config_path: str, seed: int = 0, hybrid_amb_rule=None):
    base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
    if hybrid_amb_rule:
        from RuleManager import Universal_Rule
        env = HybridAMBHeurWrapper(base, Universal_Rule(*hybrid_amb_rule))
    else:
        env = FlattenAndDiscreteWrapper(base)
    env = Monitor(env)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/dqn")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--learning_starts", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--exploration_fraction", type=float, default=0.3)
    p.add_argument("--exploration_final_eps", type=float, default=0.05)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 학습: AMB 결정을 룰에 위임 (UAV 만 RL)")
    return p.parse_args()


def predict_with_mask(model: DQN, obs: np.ndarray, mask: np.ndarray) -> int:
    """Q-values 에 -inf mask 를 씌우고 argmax."""
    import torch
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q_values = model.q_net(obs_t).cpu().numpy().reshape(-1)
    q_masked = np.where(mask, q_values, -np.inf)
    if not np.isfinite(q_masked).any():
        # fallback: stay (보통 action_idx 0 = [0,0,0])
        return 0
    return int(np.argmax(q_masked))


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.hybrid_amb_rule:
        print(f"[hybrid] AMB 결정 룰: {' / '.join(args.hybrid_amb_rule)}")
    env = make_env(args.config_path, seed=args.seed, hybrid_amb_rule=args.hybrid_amb_rule)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.log_dir, "tb"),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="dqn",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="dqn", progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")

    # 짧은 평가 (mask 적용)
    eval_env = make_env(args.config_path, seed=args.seed + 999, hybrid_amb_rule=args.hybrid_amb_rule)
    rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=args.seed + 999 + ep)
        done = False
        ep_r = 0.0
        while not done:
            mask = eval_env.action_masks()
            a = predict_with_mask(model, obs, mask)
            obs, r, term, trunc, info = eval_env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    print(f"Eval mean reward (masked): {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")


if __name__ == "__main__":
    main()
