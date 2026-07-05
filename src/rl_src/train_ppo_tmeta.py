"""T-메타 RL 학습 (플랜 v2 Phase 3-C) — train_ppo_feature 파생, TMetaWrapper 삽입.

RL action = 발송상한 T 선택(Discrete). obs=essential+load feature obs 통과, 실행=프로그램 정책.
MlpPolicy(작은 net)로 충분(action 수 개). pdrwog·occ·VecNorm·PPO위생 승계.

예: MCI_OBS_VARIANT=essential+load MCI_CAP_GATE=occ python src/rl_src/train_ppo_tmeta.py \
      --config_path scenarios/manifests/sigungu_osrm_manifest.json --total_timesteps 5000000 \
      --n_envs 8 --vec subproc --reward_mode pdrwog --norm_reward --lr_anneal --target_kl 0.03 \
      --batch_size 512 --n_epochs 5 --log_dir results/rl/redesign/tmeta_s0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env_factory import make_base_env
from hospital_feature_wrapper import HospitalFeatureWrapper
from reward_redesign_wrapper import RewardRedesignWrapper
from t_meta_wrapper import TMetaWrapper, T_SET_DEFAULT
from train_ppo_feature import FeatureMultiRegionEnv
from learning_curve_plot import try_plot_learning_curve


def _wrap(env):
    return TMetaWrapper(env)


def make_env_fn(config_path, seed=0):
    def _f():
        if config_path.endswith(".json"):
            base = FeatureMultiRegionEnv(config_path, seed=seed)   # 이미 HospitalFeatureWrapper 포함
            env = TMetaWrapper(base)
        else:
            b = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            env = TMetaWrapper(HospitalFeatureWrapper(RewardRedesignWrapper(b)))
        env = ActionMasker(env, lambda e: e.action_masks())
        return Monitor(env)
    return _f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--log_dir", default="results/rl/redesign/tmeta_s0")
    p.add_argument("--total_timesteps", type=int, default=5_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--checkpoint_freq", type=int, default=250_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--reward_mode", choices=["raw", "woG", "pdrwog", "rywt"], default="pdrwog")
    p.add_argument("--norm_reward", action="store_true", default=False)
    p.add_argument("--lr_anneal", action="store_true", default=False)
    p.add_argument("--target_kl", type=float, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["MCI_REWARD_MODE"] = args.reward_mode
    os.environ.setdefault("MCI_OBS_VARIANT", "essential+load")
    os.environ.setdefault("MCI_CAP_GATE", "occ")
    print(f"[tmeta] obs={os.environ['MCI_OBS_VARIANT']} gate={os.environ.get('MCI_CAP_GATE')} "
          f"reward={args.reward_mode} norm_reward={args.norm_reward} T_set={T_SET_DEFAULT} "
          f"lr_anneal={args.lr_anneal} target_kl={args.target_kl} n_epochs={args.n_epochs}")

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i) for i in range(args.n_envs)]
    venv = (SubprocVecEnv if args.vec == "subproc" else DummyVecEnv)(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=args.norm_reward, clip_obs=10.0)

    lr = (lambda p: args.learning_rate * p) if args.lr_anneal else args.learning_rate
    hyg = {}
    if args.target_kl is not None:
        hyg["target_kl"] = args.target_kl
    if args.n_epochs is not None:
        hyg["n_epochs"] = args.n_epochs
    model = MaskablePPO("MlpPolicy", venv, learning_rate=lr, n_steps=args.n_steps,
                        batch_size=args.batch_size, ent_coef=args.ent_coef,
                        policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=args.seed,
                        tensorboard_log=os.path.join(args.log_dir, "tb"), **hyg)
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                              save_path=os.path.join(args.log_dir, "checkpoints"), name_prefix="tmeta")
    model.learn(total_timesteps=args.total_timesteps, callback=ckpt, tb_log_name="tmeta")
    model.save(os.path.join(args.log_dir, "final_model.zip"))
    venv.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
    print(f"Saved: {args.log_dir}/final_model.zip")
    try_plot_learning_curve(args.log_dir)
    ev = make_env_fn(args.config_path, seed=args.seed + 999)()
    mr, sr = masked_evaluate(model, ev, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mr:.3f} +/- {sr:.3f}")


if __name__ == "__main__":
    main()
