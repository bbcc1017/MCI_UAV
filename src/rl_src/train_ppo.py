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
from reward_redesign_wrapper import RewardRedesignWrapper
from learning_curve_plot import try_plot_learning_curve


def mask_fn(env):
    return env.action_masks()


def _build_rule(rule_args):
    """Universal_Rule 인스턴스 생성. RuleManager 에서 import (sim_src on sys.path)."""
    from RuleManager import Universal_Rule
    return Universal_Rule(*rule_args)


def make_env_fn(config_path: str, seed: int = 0, hybrid_amb_rule=None, reward_mode="raw"):
    def _f():
        # config_path 가 .json 매니페스트면 전국 단일 정책용 멀티-지역 env
        if config_path.endswith(".json"):
            from multi_region_env import MultiRegionEnv
            env = MultiRegionEnv(config_path, seed=seed)
        else:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            if hybrid_amb_rule:
                env = HybridAMBHeurWrapper(base, _build_rule(hybrid_amb_rule))
            else:
                env = FlattenAndDiscreteWrapper(base)
        # 보상 모드 (raw=무변화 / woG=Green 제외 / rywt=Red·Yellow 가중). info['r_woG'] 사용.
        if reward_mode != "raw":
            env = RewardRedesignWrapper(env, mode=reward_mode)
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
    p.add_argument("--ent_coef", type=float, default=0.01,
                   help="엔트로피 계수 — 탐색 유지/조기수렴 방지 (개선 알고리즘)")
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 학습: AMB 결정을 룰에 위임 (UAV 만 RL). "
                        "예: --hybrid_amb_rule START RedOnly Both_AMBFirst Both_AMBFirst")
    p.add_argument("--reward_mode", choices=["raw", "woG", "rywt"], default="raw",
                   help="보상 모드: raw(원본·Green포함) / woG(Green제외) / rywt(R·Y 가중). "
                        "MultiRegionEnv 결합 가능 (RewardRedesignWrapper).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.hybrid_amb_rule:
        print(f"[hybrid] AMB 결정 룰: {' / '.join(args.hybrid_amb_rule)}")
    if args.reward_mode != "raw":
        print(f"[reward] mode={args.reward_mode} (info['r_woG'] 등으로 보상 치환)")
    env_fns = [make_env_fn(args.config_path, seed=args.seed + i,
                           hybrid_amb_rule=args.hybrid_amb_rule,
                           reward_mode=args.reward_mode)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    model = MaskablePPO(
        "MlpPolicy", venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[256, 256]),
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
    try_plot_learning_curve(args.log_dir)

    # 짧은 평가
    eval_env = make_env_fn(args.config_path, seed=args.seed + 999,
                           hybrid_amb_rule=args.hybrid_amb_rule,
                           reward_mode=args.reward_mode)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
