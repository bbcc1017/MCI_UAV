"""SB3/DQN 호환 wrapper.

MCIEnvironment_gym 의 외부 액션 공간은 MultiDiscrete([3, H+1, 2]) 인데,
- DQN 은 Discrete 만 지원하므로 평탄화 필요
- MaskablePPO 는 MultiDiscrete 도 지원하지만 차원별 독립 마스크라 정합 결합 액션 보장 못 함

이 wrapper 는:
1. amb_num 또는 uav_num 이 0이면 mode 차원을 자동 제거
   - amb_num=0 → mode 항상 1(UAV) 고정, 액션 공간 절반 축소 (3*(H+1))
   - uav_num=0 → mode 항상 0(AMB) 고정
2. action_space 를 Discrete 로 평탄화 (효과적인 nvec 의 곱)
3. action_masks() 결합(joint) mask 노출 — 차원 제거 시 자동으로 슬라이싱
4. obs 를 평탄 Box(flat_dim,) float32 로 concat (DQN MlpPolicy 호환)

향후 AMB 재도입 시 wrapper 코드 수정 없이 amb_num>0 인 시나리오만 만들면
mode 차원이 자동으로 다시 활성화됨.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _flatten_dict_obs(obs: dict, key_order, dtype=np.float32) -> np.ndarray:
    parts = []
    for k in key_order:
        v = obs[k]
        parts.append(np.asarray(v, dtype=dtype).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


class FlattenAndDiscreteWrapper(gym.Wrapper):
    """
    - dict obs → flat Box float32
    - MultiDiscrete([3, H+1, 2]) → Discrete (사용 가능한 mode 만 반영)
    - action_masks() 노출
    """

    def __init__(self, env):
        super().__init__(env)
        # 원본 env nvec
        nvec = env.action_space.nvec.tolist()  # [3, H+1, 2]
        assert len(nvec) == 3, f"기대 형식 [class, dest, mode], got {nvec}"
        self._orig_nvec = nvec  # [3, H+1, 2] 그대로 보관 (마스크 슬라이싱에 필요)
        self.H = nvec[1] - 1

        # AMB / UAV 자원 가용성에 따라 mode 차원 자동 축소
        unwrapped = env.unwrapped
        amb_num = int(getattr(unwrapped, "amb_num", 0))
        uav_num = int(getattr(unwrapped, "uav_num", 0))

        if amb_num == 0 and uav_num > 0:
            # UAV 전용: mode 항상 1
            self._fixed_mode = 1
            self._effective_nvec = [nvec[0], nvec[1]]
            mode_label = "UAV-only (mode=1 고정)"
        elif uav_num == 0 and amb_num > 0:
            # AMB 전용: mode 항상 0
            self._fixed_mode = 0
            self._effective_nvec = [nvec[0], nvec[1]]
            mode_label = "AMB-only (mode=0 고정)"
        else:
            # 둘 다 있거나 둘 다 0인 경우 → 원본 mode 차원 유지
            self._fixed_mode = None
            self._effective_nvec = nvec
            mode_label = "AMB+UAV (mode 자유)"

        self._n_actions = int(np.prod(self._effective_nvec))
        self.action_space = spaces.Discrete(self._n_actions)
        print(f"[FlattenAndDiscreteWrapper] {mode_label}, action_space=Discrete({self._n_actions})")

        # obs 평탄화
        self._obs_keys = list(env.observation_space.spaces.keys())
        flat_dim = sum(int(np.prod(env.observation_space.spaces[k].shape)) for k in self._obs_keys)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    # ---------- action 변환 ----------
    def _decode(self, action: int):
        """평탄 정수 → [class, dest, mode]."""
        a = int(action)
        if self._fixed_mode is not None:
            n_dest = self._effective_nvec[1]
            c = a // n_dest
            d = a % n_dest
            return [c, d, self._fixed_mode]
        # 원본 3차원
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        d = rem // n_mode
        m = rem % n_mode
        return [c, d, m]

    def _encode(self, decoded):
        """[class, dest, mode] → 평탄 정수 (decode 의 역연산). mode 가 고정된 wrapper 면 mode 항 무시."""
        c, d, m = int(decoded[0]), int(decoded[1]), int(decoded[2])
        if self._fixed_mode is not None:
            return c * self._effective_nvec[1] + d
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        return c * (n_dest * n_mode) + d * n_mode + m

    # public aliases (외부 정책에서 사용)
    decode_action = _decode
    encode_action = _encode

    # ---------- gym API ----------
    def step(self, action):
        decoded = self._decode(action)
        obs, reward, terminated, truncated, info = self.env.step(decoded)
        return _flatten_dict_obs(obs, self._obs_keys), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return _flatten_dict_obs(obs, self._obs_keys), info

    # ---------- action mask ----------
    def action_masks(self) -> np.ndarray:
        """결합 mask: shape (n_actions,) bool. mode 차원 고정 시 해당 슬라이스만 평탄화."""
        full = self.env.unwrapped.action_masks_joint()  # (3 * (H+1) * 2,)
        full = full.reshape(self._orig_nvec[0], self._orig_nvec[1], self._orig_nvec[2])
        if self._fixed_mode is not None:
            sliced = full[:, :, self._fixed_mode]  # (3, H+1)
            return sliced.reshape(-1)
        return full.reshape(-1)


class HybridAMBHeurWrapper(FlattenAndDiscreteWrapper):
    """2안 학습용: AMB(mode=0) dispatch 는 휴리스틱 룰이 결정, UAV(mode=1) 는 RL 결정 그대로.

    RL 의 [c, d, m] 출력에서 m==0 이면 룰의 결정 [c2, d2, 0] 으로 swap (strict).
    m==1 이면 RL 결정 그대로 시뮬에 전달.
    """

    def __init__(self, env, rule):
        super().__init__(env)
        self.rule = rule
        self.rule.init_with_scenario({'EntityManager': env.unwrapped.en_manager})

    def _dict_obs(self):
        u = self.env.unwrapped
        obs = u.en_manager.get_full_obs()
        obs['time'] = u.ev_manager.time
        return obs

    def reset(self, **kwargs):
        seed = kwargs.get('seed', 0)
        self.rule.set_seed(np.random.default_rng(seed))
        return super().reset(**kwargs)

    def step(self, action):
        decoded = self._decode(action)  # [c, d, m]
        if decoded[2] == 0:
            rule_action = self.rule.select(self._dict_obs())
            final = [int(x) for x in rule_action]
        else:
            final = decoded
        obs, reward, terminated, truncated, info = self.env.step(final)
        return _flatten_dict_obs(obs, self._obs_keys), reward, terminated, truncated, info
