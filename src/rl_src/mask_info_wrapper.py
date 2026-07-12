"""v5 공정비교 하네스 — step info 에 다음 상태 action_mask 와 결정 간 경과시간 dt 주입.

flat obs(355)에는 helipad 등 마스크 재료가 없어 off-policy replay 에서 next-state
마스크를 재계산할 수 없다 → 수집 시점에 info 로 실어 버퍼(`masked_replay_buffer`)가
저장한다. dt(결정 간 sim 경과분)는 SMDP γ^Δt 프로브용.

체인 위치(트레이너 조립, train_zoo 참조):
  Monitor → ActionMasker → MaskInfoWrapper → HospitalFeatureWrapper(또는
  FeatureMultiRegionEnv) → RewardRedesignWrapper → base
obs/reward 무변형 — action_masks() 는 gym.Wrapper 의 속성 위임으로 그대로 노출된다.

주의: FeatureMultiRegionEnv 는 unwrapped 가 자기 자신이라 ev_manager 접근이 지역
env 에 따라 달라짐 → 시각은 info['time'](매 스텝 존재, MCIEnvironment_gymnasium)만
사용하고, 에피소드 첫 스텝의 dt 는 0 으로 둔다(첫 결정 이전 구간은 정책 무관).
"""
import gymnasium as gym
import numpy as np


class MaskInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._t_prev = None  # None = 에피소드 첫 스텝(dt 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t_prev = None
        info["action_mask"] = np.asarray(self.env.action_masks(), dtype=bool)
        info["dt"] = 0.0
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        t = info.get("time", None)
        if t is None or self._t_prev is None:
            info["dt"] = 0.0
        else:
            info["dt"] = max(float(t) - self._t_prev, 0.0)
        if t is not None:
            self._t_prev = float(t)
        # 종결 스텝의 마스크는 타깃에서 (1-done)으로 무시되지만 계산은 항상 가능(stateless)
        info["action_mask"] = np.asarray(self.env.action_masks(), dtype=bool)
        return obs, r, term, trunc, info
