"""MaskablePPO 학습 스크립트.

예:
    python src/rl_src/train_ppo.py --config_path scenarios/exp_uav_seoul_test_uav/(37.5665,126.978)/config_(37.5665,126.978).yaml --total_timesteps 200000 --n_envs 4 --seed 0 --log_dir results/rl/ppo_seoul
"""
import argparse
import os
import sys

# repo-relative import
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_factory import make_base_env
from env_wrapper import FlattenAndDiscreteWrapper, HybridAMBHeurWrapper


def mask_fn(env):
    return env.action_masks()


def _build_rule(rule_args):
    """Universal_Rule 인스턴스 생성. RuleManager 에서 import (sim_src on sys.path)."""
    from RuleManager import Universal_Rule
    return Universal_Rule(*rule_args)


def make_env_fn(config_path: str, seed: int = 0, hybrid_amb_rule=None):
    def _f():
        base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
        if hybrid_amb_rule:
            env = HybridAMBHeurWrapper(base, _build_rule(hybrid_amb_rule))
        else:
            env = FlattenAndDiscreteWrapper(base)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _f


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="results/rl/ppo")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 학습: AMB 결정을 룰에 위임 (UAV 만 RL). "
                        "예: --hybrid_amb_rule START RedOnly Both_AMBFirst Both_AMBFirst")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.hybrid_amb_rule:
        print(f"[hybrid] AMB 결정 룰: {' / '.join(args.hybrid_amb_rule)}")
    env_fns = [make_env_fn(args.config_path, seed=args.seed + i,
                           hybrid_amb_rule=args.hybrid_amb_rule)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    model = MaskablePPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.log_dir, "tb"),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="ppo",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="ppo", progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")

    # 짧은 평가
    eval_env = make_env_fn(args.config_path, seed=args.seed + 999,
                           hybrid_amb_rule=args.hybrid_amb_rule)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
