"""MaskablePPO 학습 스크립트 — EnrichedObsMaskWrapper 적용판.

train_ppo.py 의 파생본. env_wrapper.py / multi_region_env.py / env_factory.py
는 수정하지 않고, 여기서만 EnrichedObsMaskWrapper 로 base env 를 감싼다.

차이점:
  * FlattenAndDiscreteWrapper 대신 EnrichedObsMaskWrapper 사용
  * --topk 인자로 ETA top-k 마스크 제어 (None 이면 비활성, 기본 10)
  * 매니페스트(.json) 입력 시 각 지역 base env 를 EnrichedObsMaskWrapper 로
    감싸는 _EnrichedMultiRegionEnv 자체 구현 사용 (multi_region_env.py 무수정)

주의:
  * obs 차원이 기존(FlattenAndDiscreteWrapper) 과 달라 기존 학습 가중치와 비호환.
    반드시 새로 학습할 것.

예:
  CUDA_VISIBLE_DEVICES="" python src/rl_src/train_ppo_enriched.py \\
    --config_path scenarios/plan1nat_manifest.json --total_timesteps 200000 \\
    --n_envs 4 --log_dir results/rl/ppo_enriched --topk 10
"""
import argparse
import json
import os
import sys

# repo-relative import
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_factory import make_base_env
from enriched_env_wrapper import EnrichedObsMaskWrapper
from learning_curve_plot import try_plot_learning_curve


# ---------- 매니페스트 → 멀티 지역 enriched env ----------
class EnrichedMultiRegionEnv(gym.Env):
    """multi_region_env.MultiRegionEnv 의 enriched 판.

    각 지역 base env 를 EnrichedObsMaskWrapper 로 감싼 뒤 reset() 마다
    무작위 지역 하나에 step/reset/action_masks 를 위임한다.

    전제: 모든 지역의 H(병원 수)가 동일 (fixed_hos_num) — obs/action 차원 일치.
    """
    metadata = {"render_modes": []}

    def __init__(self, manifest_path: str, seed: int = 0, eval_mode: bool = False,
                 topk=None):
        super().__init__()
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        self.regions = list(manifest.keys())
        if not self.regions:
            raise ValueError(f"빈 manifest: {manifest_path}")

        self._envs = []
        for i, region in enumerate(self.regions):
            cfg = manifest[region]
            if not os.path.exists(cfg):
                raise FileNotFoundError(f"[{region}] config 미발견: {cfg}")
            base = make_base_env(cfg, seed=seed + i,
                                 rule_test=False, eval_mode=eval_mode)
            self._envs.append(EnrichedObsMaskWrapper(base, topk=topk))

        obs_shapes = {tuple(e.observation_space.shape) for e in self._envs}
        act_ns = {int(e.action_space.n) for e in self._envs}
        if len(obs_shapes) != 1 or len(act_ns) != 1:
            detail = "\n".join(
                f"  {r}: obs={e.observation_space.shape} act={e.action_space.n}"
                for r, e in zip(self.regions, self._envs)
            )
            raise ValueError(
                "지역별 obs/action 차원이 불일치합니다 — fixed_hos_num 으로 "
                f"시나리오를 재생성하세요.\n{detail}")

        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._cur = self._envs[0]

    @property
    def current_region(self) -> str:
        return self.regions[self._idx]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._idx = int(self._rng.integers(len(self._envs)))
        self._cur = self._envs[self._idx]
        return self._cur.reset(seed=seed, options=options)

    def step(self, action):
        return self._cur.step(action)

    def action_masks(self) -> np.ndarray:
        return self._cur.action_masks()

    @property
    def unwrapped(self):
        return self._cur.unwrapped

    def render(self):
        return None

    def close(self):
        for e in self._envs:
            e.close()


# ---------- ActionMasker 콜백 ----------
def mask_fn(env):
    return env.action_masks()


def make_env_fn(config_path: str, seed: int = 0, topk=None):
    def _f():
        if config_path.endswith(".json"):
            env = EnrichedMultiRegionEnv(config_path, seed=seed, topk=topk)
        else:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=False)
            env = EnrichedObsMaskWrapper(base, topk=topk)
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
    p.add_argument("--log_dir", default="results/rl/ppo_enriched")
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--topk", type=int, default=10,
                   help="ETA top-k mask 활성 (None 또는 -1 이면 비활성)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    topk = None if args.topk is None or args.topk < 1 else args.topk

    env_fns = [make_env_fn(args.config_path, seed=args.seed + i, topk=topk)
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
        name_prefix="ppo_enriched",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=ckpt_cb,
                tb_log_name="ppo_enriched", progress_bar=False)
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved: {final_path}")
    try_plot_learning_curve(args.log_dir)

    eval_env = make_env_fn(args.config_path, seed=args.seed + 999, topk=topk)()
    mean_r, std_r = masked_evaluate(model, eval_env, n_eval_episodes=10, use_masking=True)
    print(f"Eval mean reward: {mean_r:.3f} +/- {std_r:.3f}")


if __name__ == "__main__":
    main()
