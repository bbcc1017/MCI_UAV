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
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper, HybridAMBHeurWrapper
from learning_curve_plot import try_plot_learning_curve


class DoubleDQN(DQN):
    """Double DQN — 피드백 3/5 의 DQN 알고리즘 개선.

    표준 DQN 은 타깃 네트워크로 행동 선택·평가를 둘 다 해서 Q 과대추정이 누적
    → 본 환경에서 정책 붕괴. Double DQN 은 행동 선택은 online net, 평가는 target
    net 으로 분리해 과대추정을 완화한다. train() 의 타깃 계산만 교체.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = (replay_data.discounts
                         if getattr(replay_data, "discounts", None) is not None
                         else self.gamma)
            with th.no_grad():
                # 행동 선택은 online q_net, 가치 평가는 target net (Double DQN 핵심)
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q = self.q_net_target(replay_data.next_observations)
                next_q = th.gather(next_q, dim=1, index=next_actions)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q = self.q_net(replay_data.observations)
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def make_env(config_path: str, seed: int = 0, hybrid_amb_rule=None):
    # config_path 가 .json 매니페스트면 전국 단일 정책용 멀티-지역 env
    if config_path.endswith(".json"):
        from multi_region_env import MultiRegionEnv
        return Monitor(MultiRegionEnv(config_path, seed=seed))
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
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--learning_starts", type=int, default=5_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--exploration_fraction", type=float, default=0.4)
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

    # Double DQN + 은닉 256×2 (개선 알고리즘)
    model = DoubleDQN(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=dict(net_arch=[256, 256]),
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
    try_plot_learning_curve(args.log_dir)

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
