"""REINFORCE 학습 스크립트.

예:
    python src/rl_src/train_reinforce.py \
        --config_path scenarios/.../config.yaml \
        --total_timesteps 200000 --seed 0 \
        --log_dir results/rl/reinforce_helipad
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper, HybridAMBHeurWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/reinforce")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--print_every_episodes", type=int, default=20)
    p.add_argument("--agent_version", choices=["v1", "v2"], default="v2",
                   help="v2=현행(actor-critic baseline+엔트로피, reinforce_agent.py), "
                        "v1=피드백3 이전 순수 REINFORCE(reinforce_agent_v1.py). "
                        "분리 실험: 알고리즘 변경 효과 격리용")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 학습: AMB 결정을 룰에 위임 (UAV 만 RL)")
    return p.parse_args()


def make_env(config_path: str, seed: int = 0, hybrid_amb_rule=None):
    # config_path 가 .json 매니페스트면 전국 단일 정책용 멀티-지역 env
    if config_path.endswith(".json"):
        from multi_region_env import MultiRegionEnv
        return MultiRegionEnv(config_path, seed=seed)
    base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
    if hybrid_amb_rule:
        from RuleManager import Universal_Rule
        return HybridAMBHeurWrapper(base, Universal_Rule(*hybrid_amb_rule))
    return FlattenAndDiscreteWrapper(base)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_dir = os.path.join(args.log_dir, "tb", "reinforce_1")
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"[REINFORCE] log_dir = {args.log_dir}")
    print(f"            tb     = {tb_dir}")

    if args.hybrid_amb_rule:
        print(f"[hybrid] AMB 결정 룰: {' / '.join(args.hybrid_amb_rule)}")
    env = make_env(args.config_path, seed=args.seed, hybrid_amb_rule=args.hybrid_amb_rule)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    print(f"            obs_dim={obs_dim}  n_actions={n_actions}")

    if args.agent_version == "v1":
        from reinforce_agent_v1 import ReinforceAgent
    else:
        from reinforce_agent import ReinforceAgent
    agent = ReinforceAgent(obs_dim=obs_dim, n_actions=n_actions,
                           lr=args.learning_rate, gamma=args.gamma)
    print(f"            device={agent.device}  lr={args.learning_rate}  "
          f"gamma={args.gamma}  agent_version={args.agent_version}")

    rng_seed = args.seed
    total_steps = 0
    episode_idx = 0
    last_ckpt_at = 0
    rew_window = []
    len_window = []
    t0 = time.time()

    while total_steps < args.total_timesteps:
        obs, _ = env.reset(seed=rng_seed)
        rng_seed += 1
        done = False
        ep_steps = 0
        ep_total_r = 0.0
        while not done:
            mask = env.action_masks()
            action, _ = agent.act(obs, mask=mask, deterministic=False, store=True)
            obs, r, term, trunc, _ = env.step(action)
            agent.store_reward(r)
            ep_steps += 1
            ep_total_r += float(r)
            total_steps += 1
            done = term or trunc

        loss = agent.train_step()
        episode_idx += 1
        rew_window.append(ep_total_r)
        len_window.append(ep_steps)
        if len(rew_window) > 100:
            rew_window.pop(0)
            len_window.pop(0)

        elapsed = time.time() - t0
        fps = int(total_steps / max(elapsed, 1e-3))

        ep_rew_mean = float(np.mean(rew_window))
        ep_len_mean = float(np.mean(len_window))
        writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, total_steps)
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, total_steps)
        writer.add_scalar("rollout/ep_reward", ep_total_r, total_steps)
        writer.add_scalar("train/loss", loss, total_steps)
        writer.add_scalar("time/fps", fps, total_steps)
        writer.add_scalar("time/total_timesteps", total_steps, total_steps)
        writer.add_scalar("time/episodes", episode_idx, total_steps)
        writer.add_scalar("time/time_elapsed", elapsed, total_steps)

        if episode_idx % args.print_every_episodes == 0:
            print(f"  step={total_steps:7d}  ep={episode_idx:5d}  "
                  f"ep_rew_mean={ep_rew_mean:7.3f}  ep_len={ep_len_mean:5.1f}  "
                  f"loss={loss:9.2f}  fps={fps}  elapsed={int(elapsed)}s")

        if total_steps - last_ckpt_at >= args.checkpoint_freq:
            ckpt_path = os.path.join(ckpt_dir, f"reinforce_{total_steps}_steps.pt")
            agent.save(ckpt_path)
            last_ckpt_at = total_steps

    final_path = os.path.join(args.log_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nSaved: {final_path}")
    writer.close()

    eval_env = make_env(args.config_path, seed=args.seed + 99999, hybrid_amb_rule=args.hybrid_amb_rule)
    rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=args.seed + 99999 + ep)
        ep_r = 0.0
        done = False
        while not done:
            mask = eval_env.action_masks()
            a, _ = agent.act(obs, mask=mask, deterministic=True, store=False)
            obs, r, term, trunc, _ = eval_env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    print(f"Eval mean reward (deterministic, 10 ep): {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")


if __name__ == "__main__":
    main()
