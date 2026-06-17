"""MaskablePPO 학습 + 피드백 5 보상 재설계/휴리스틱-advantage 활성화.

train_ppo.py 를 베이스로 RewardRedesignWrapper / HeuristicAdvantageWrapper 를 wrapping 체인에
선택적으로 끼워넣는다. train_ppo.py / env_wrapper.py 는 수정하지 않는다.

wrapping 순서 (외→내):
    Monitor → ActionMasker → [HeuristicAdvantageWrapper] →
    FlattenAndDiscreteWrapper(or HybridAMBHeurWrapper) → [RewardRedesignWrapper] → base_env

  - RewardRedesignWrapper 는 가장 안쪽: base 의 reward 를 woG/rywt 로 치환.
  - FlattenAndDiscreteWrapper 가 obs/action 평탄화.
  - HeuristicAdvantageWrapper 는 그 위에서 reward 에 baseline 을 뺀다.
  - ActionMasker / Monitor 는 마지막에.

사용 예 (woG + precomputed advantage, CPU 강제, plan1nat 매니페스트):
    CUDA_VISIBLE_DEVICES="" /home/RYU/anaconda3/envs/UAV/bin/python \\
        src/rl_src/train_ppo_reward.py \\
        --config_path scenarios/manifests/plan1nat_manifest.json \\
        --reward_mode woG \\
        --total_timesteps 100000 --n_envs 4 \\
        --log_dir results/rl/ppo_reward_woG
"""
import argparse
import os
import sys

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
from advantage_wrapper import HeuristicAdvantageWrapper
from learning_curve_plot import try_plot_learning_curve


def mask_fn(env):
    return env.action_masks()


def _build_rule(rule_args):
    from RuleManager import Universal_Rule
    return Universal_Rule(*rule_args)


def make_env_fn(config_path: str, seed: int = 0, hybrid_amb_rule=None, *,
                reward_mode="raw", red_w=2.0, yellow_w=1.0, green_w=0.0,
                advantage_mode=None, advantage_subtract_at="terminal",
                advantage_rule=None, advantage_csv=None, advantage_region=None,
                advantage_use_r_woG=False):
    """wrapping 체인 결합 factory.

    매니페스트(.json) 는 advantage rollout 모드를 지원하지 않는다 (config_path 가 단일이 아니므로).
    """
    is_manifest = config_path.endswith(".json")

    def _f():
        if is_manifest:
            from multi_region_env import MultiRegionEnv
            env = MultiRegionEnv(config_path, seed=seed)
            # MultiRegionEnv 는 이미 FlattenAndDiscreteWrapper 구조와 동일한 의무를 가짐
            # (자체에서 obs/action 평탄화). RewardRedesign / Advantage 적용은 별도 분석 필요 →
            # 본 launcher 는 우선 매니페스트면 advantage 비활성, reward redesign 도 미적용 안내.
            if reward_mode != "raw" or advantage_mode is not None:
                print(f"[train_ppo_reward] WARNING: manifest({config_path}) 모드에서는 "
                      f"reward_mode/advantage 옵션 미지원 — 무시")
            env = ActionMasker(env, mask_fn)
            env = Monitor(env)
            return env

        base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)

        # 1) reward redesign (가장 안쪽)
        if reward_mode != "raw":
            base = RewardRedesignWrapper(base, mode=reward_mode,
                                         red_w=red_w, yellow_w=yellow_w, green_w=green_w)

        # 2) flatten / hybrid
        if hybrid_amb_rule:
            env = HybridAMBHeurWrapper(base, _build_rule(hybrid_amb_rule))
        else:
            env = FlattenAndDiscreteWrapper(base)

        # 3) advantage (terminal/replace baseline subtraction)
        if advantage_mode is not None:
            env = HeuristicAdvantageWrapper(
                env, config_path,
                mode=advantage_mode,
                subtract_at=advantage_subtract_at,
                rule=advantage_rule,
                baseline_csv=advantage_csv,
                region=advantage_region,
                use_r_woG=advantage_use_r_woG,
            )

        # 4) mask + monitor
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
    p.add_argument("--log_dir", default="results/rl/ppo_reward")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"))
    # 보상 재설계
    p.add_argument("--reward_mode", choices=["raw", "woG", "rywt"], default="raw")
    p.add_argument("--red_w",    type=float, default=2.0)
    p.add_argument("--yellow_w", type=float, default=1.0)
    p.add_argument("--green_w",  type=float, default=0.0)
    # 휴리스틱 advantage
    p.add_argument("--advantage_mode", choices=["rollout", "precomputed"], default=None)
    p.add_argument("--advantage_subtract_at", choices=["terminal", "replace"], default="terminal")
    p.add_argument("--advantage_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="rollout 모드 휴리스틱 룰 (없으면 START YellowNearest Both_UAVFirst Both_UAVFirst)")
    p.add_argument("--advantage_csv", default=None,
                   help="precomputed 모드: region/heuristic_R[_woG] 컬럼이 있는 CSV "
                        "(예: results/plan1nat_f3_eval.csv)")
    p.add_argument("--advantage_region", default=None,
                   help="precomputed 모드 region key (예: 서울)")
    p.add_argument("--advantage_use_r_woG", action="store_true",
                   help="baseline 산출 시 R_woG 컬럼 사용 (reward_mode=woG 와 함께 권장)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[train_ppo_reward] reward_mode={args.reward_mode} "
          f"(red_w={args.red_w} yellow_w={args.yellow_w} green_w={args.green_w})")
    if args.advantage_mode:
        print(f"[train_ppo_reward] advantage={args.advantage_mode}/{args.advantage_subtract_at} "
              f"(use_r_woG={args.advantage_use_r_woG})")
    if args.hybrid_amb_rule:
        print(f"[train_ppo_reward] hybrid AMB rule: {' / '.join(args.hybrid_amb_rule)}")

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i,
                           hybrid_amb_rule=args.hybrid_amb_rule,
                           reward_mode=args.reward_mode,
                           red_w=args.red_w, yellow_w=args.yellow_w, green_w=args.green_w,
                           advantage_mode=args.advantage_mode,
                           advantage_subtract_at=args.advantage_subtract_at,
                           advantage_rule=args.advantage_rule,
                           advantage_csv=args.advantage_csv,
                           advantage_region=args.advantage_region,
                           advantage_use_r_woG=args.advantage_use_r_woG)
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
                tb_log_name="ppo_reward", progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")
    try_plot_learning_curve(args.log_dir)

    eval_env = make_env_fn(args.config_path, seed=args.seed + 999,
                           hybrid_amb_rule=args.hybrid_amb_rule,
                           reward_mode=args.reward_mode,
                           red_w=args.red_w, yellow_w=args.yellow_w, green_w=args.green_w,
                           advantage_mode=args.advantage_mode,
                           advantage_subtract_at=args.advantage_subtract_at,
                           advantage_rule=args.advantage_rule,
                           advantage_csv=args.advantage_csv,
                           advantage_region=args.advantage_region,
                           advantage_use_r_woG=args.advantage_use_r_woG)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward (under wrapped scheme): {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
