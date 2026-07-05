"""T-메타 RL 래퍼 (플랜 v2 Phase 3-C) — 발송상한 T 를 상태의존적으로 학습.

RL 의 행동공간을 원래의 (class,dest,mode) 대신 **발송상한 T 선택(Discrete)** 으로 치환한다.
실행은 Phase 3-B 우승 프로그램(dest=적격 중 p_sent<T 최속, mode=시간절감형 UAV)이 그 T 로 수행.
obs 는 feature obs(essential+load) 그대로 통과 → RL 이 ρ·부하·시간을 보고 T 를 조절.

해석가능-by-construction: 학습된 정책은 "상태 → T" 매핑 하나뿐이라, info["T_selected"] 로깅으로
T=f(ρ, 시간, 지역) 규칙을 직접 추출할 수 있다. 표현력=고정 프로그램 < T-메타 < full-RL,
해석성은 역순 → 중간지대 탐색.

스택: base → RewardRedesign(pdrwog) → HospitalFeatureWrapper → TMetaWrapper → ActionMasker → Monitor.
클래스 우선순위 rule 은 지역 무관 generic(전국 단일정책 유지) — 프로그램이 dest·mode 를 덮으므로
rule 은 Red/Yellow 우선순위만 결정(어디서나 동일). make_program_policy 의 _sync 가 멀티지역 en_manager
교체를 자동 처리하므로 정책은 __init__ 1회 생성.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from program_policy import make_program_policy
from loadbalance_heuristic import H_DEFAULT

# 지역 무관 generic 클래스규칙(Red/Yellow 우선순위만 사용; dest·mode 는 프로그램이 덮음)
GENERIC_RULE = "START, YellowNearest, Red Both_AMBFirst, Yellow Both_AMBFirst"
T_SET_DEFAULT = (2.0, 3.0, 4.0, 6.0, 8.0, 1e9)  # 1e9 = 상한 없음(순수 시간절감 프로그램)


class TMetaWrapper(gym.Wrapper):
    def __init__(self, env, rule_name=GENERIC_RULE, t_set=T_SET_DEFAULT,
                 uav_time_factor=0.8, uav_red_only=False, H=H_DEFAULT):
        super().__init__(env)
        self._t_set = list(t_set)
        self.observation_space = env.observation_space          # passthrough
        self.action_space = spaces.Discrete(len(self._t_set))
        # T 별 프로그램 정책(멀티지역 자동 resync) — 1회 생성
        self._pols = [make_program_policy(rule_name, T=t, uav_time_factor=uav_time_factor,
                                          uav_red_only=uav_red_only, H=H) for t in self._t_set]
        self._t_hist = np.zeros(len(self._t_set), int)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._t_hist = np.zeros(len(self._t_set), int)
        return obs, info

    def step(self, action):
        ti = int(action)
        T = self._t_set[ti]
        mask = np.asarray(self.env.action_masks(), bool)         # inner (192/96)
        inner = self._pols[ti](None, mask, self.env.unwrapped)   # 프로그램이 T 로 실행
        obs, r, term, trunc, info = self.env.step(inner)
        self._t_hist[ti] += 1
        info = dict(info) if info else {}
        info["T_selected"] = T
        if term or trunc:
            info["T_hist"] = self._t_hist.tolist()
        return obs, r, term, trunc, info

    def action_masks(self):
        return np.ones(len(self._t_set), dtype=bool)             # 모든 T 합법
