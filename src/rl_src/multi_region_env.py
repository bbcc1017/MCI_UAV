"""전국 단일 정책 학습용 멀티-지역 env (피드백 2번).

매 reset() 마다 지역을 무작위로 샘플링한다. manifest(JSON: {region: config_path})
의 모든 지역 env 를 미리 구성해 두고, reset() 에서 하나를 골라 위임한다.

전제: 모든 시나리오가 fixed_hos_num 으로 hos_num(=H) 이 동일해야 obs/action
차원이 일치한다 (gen_regions.py --fixed_hos_num 으로 재생성).

학습 스크립트(train_ppo/dqn/reinforce.py)는 --config_path 가 .json 이면
이 env 를 사용하도록 분기한다.
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_factory import make_base_env  # noqa: E402
from env_wrapper import FlattenAndDiscreteWrapper  # noqa: E402


class MultiRegionEnv(gym.Env):
    """reset() 마다 지역을 랜덤 샘플링하는 멀티-시나리오 env.

    내부적으로 지역별 FlattenAndDiscreteWrapper env 를 미리 만들어두고,
    reset() 에서 하나를 선택해 step/reset/action_masks 를 위임한다.
    """
    metadata = {"render_modes": []}

    def __init__(self, manifest_path, seed: int = 0, eval_mode: bool = False):
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
            self._envs.append(FlattenAndDiscreteWrapper(base))

        # 모든 지역 obs/action 차원 일치 검증 (fixed_hos_num 필수)
        obs_shapes = {tuple(e.observation_space.shape) for e in self._envs}
        act_ns = {int(e.action_space.n) for e in self._envs}
        if len(obs_shapes) != 1 or len(act_ns) != 1:
            detail = "\n".join(f"  {r}: obs={e.observation_space.shape} "
                               f"act={e.action_space.n}"
                               for r, e in zip(self.regions, self._envs))
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
        # eval_policy 등이 .preventable / .preventable_woG 에 접근 →
        # 현재 선택된 지역의 MCIEnvironment_gym 으로 위임.
        return self._cur.unwrapped

    def render(self):
        return None

    def close(self):
        for e in self._envs:
            e.close()
