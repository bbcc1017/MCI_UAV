"""피드백 5 — 아이디어 3 (관측 정보 강화) + 4 (더 강한 action mask) 통합 wrapper.

휴리스틱이 활용하는 정보(helipad 보유, capacity 잔여, AMB/UAV ETA, 거리)를
RL 정책 obs 에 명시적으로 추가하고, 의미 없는 행동(헬기장 없는 병원으로 UAV,
capacity 0 병원, ETA 가 너무 먼 병원)을 더 강하게 마스킹한다.

설계 원칙:
  * src/sim_src/, env_wrapper.py, aggregate_obs.py, env_factory.py,
    multi_region_env.py, train_ppo.py 모두 **수정하지 않는다**.
  * AggregateObsWrapper 와 공존 — env 변수 MCI_REDUCED_OBS=1 이면 그대로 활용.
  * FlattenAndDiscreteWrapper 의 decode/encode/fixed-mode 로직과 동일한 동작을
    유지하면서 obs 와 action_masks 만 강화한다.

구성:
  EnrichedObsMaskWrapper(base_env, topk=None)
    1) 내부에서 AggregateObsWrapper 자동 적용 (MCI_REDUCED_OBS=1 일 때)
    2) base obs 를 평탄화 + 강화 feature concat
    3) action_masks() = base joint mask AND helipad/capacity/topk-ETA gate

obs / action 차원:
    self.observation_space.shape  ── (flat_base_dim + enrich_dim,)
    self.action_space             ── Discrete (mode 차원 슬라이싱 시 자동 축소)

train_ppo_enriched.py 에서 ActionMasker 와 함께 사용한다.
"""
from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------- 평탄화 유틸 (env_wrapper._flatten_dict_obs 와 동일 시그니처) ----------
def _flatten_dict_obs(obs: dict, key_order, dtype=np.float32) -> np.ndarray:
    parts = []
    for k in key_order:
        v = obs[k]
        parts.append(np.asarray(v, dtype=dtype).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


class EnrichedObsMaskWrapper(gym.Wrapper):
    """관측 정보 강화 + 더 강한 action mask.

    Parameters
    ----------
    env : MCIEnvironment_gym
        ``make_base_env(cfg)`` 로 생성한 base env (FlattenAndDiscreteWrapper 미적용).
    topk : Optional[int]
        ETA 가 짧은 상위 k 개 병원만 후보로 남긴다 (mode 별로 따로 적용).
        None 이면 ETA top-k 마스크는 비활성 (helipad/capacity 만 추가).
    """

    def __init__(self, env: gym.Env, topk: Optional[int] = 10):
        # 1) AggregateObsWrapper 자동 활성화 (FlattenAndDiscreteWrapper 와 동일 규약)
        if os.environ.get("MCI_REDUCED_OBS") == "1":
            from aggregate_obs import AggregateObsWrapper  # noqa: WPS433
            env = AggregateObsWrapper(env)
        super().__init__(env)
        self._topk = int(topk) if topk is not None else None

        # ---------- 2) base action 차원 (FlattenAndDiscreteWrapper 동등 로직) ----------
        nvec = env.action_space.nvec.tolist()
        assert len(nvec) == 3, f"기대 형식 [class, dest, mode], got {nvec}"
        self._orig_nvec = nvec
        self.H = nvec[1] - 1

        unwrapped = env.unwrapped
        amb_num = int(getattr(unwrapped, "amb_num", 0))
        uav_num = int(getattr(unwrapped, "uav_num", 0))
        if amb_num == 0 and uav_num > 0:
            self._fixed_mode = 1
            self._effective_nvec = [nvec[0], nvec[1]]
            mode_label = "UAV-only (mode=1 고정)"
        elif uav_num == 0 and amb_num > 0:
            self._fixed_mode = 0
            self._effective_nvec = [nvec[0], nvec[1]]
            mode_label = "AMB-only (mode=0 고정)"
        else:
            self._fixed_mode = None
            self._effective_nvec = nvec
            mode_label = "AMB+UAV (mode 자유)"

        self._n_actions = int(np.prod(self._effective_nvec))
        self.action_space = spaces.Discrete(self._n_actions)

        # ---------- 3) 정적 enrich feature (시나리오 메타) ----------
        hos_props = unwrapped.en_manager.en_properties['hospital']
        self._hos_num = int(hos_props['hos_num'])
        self._hos_max_send = np.asarray(hos_props['hos_max_send'], dtype=np.float32)
        # helipad 보유 여부 (1.0/0.0)
        helipad_idx = np.asarray(hos_props.get('hos_helipad_idx', np.array([])), dtype=int)
        self._helipad_mask = np.zeros(self._hos_num, dtype=np.float32)
        if helipad_idx.size > 0:
            self._helipad_mask[helipad_idx] = 1.0
        # 병원→사고지점 거리 (UAV 직선 / AMB road)
        self._d_HtoS_euc = np.asarray(hos_props.get('d_HtoS_euc',
                                                    hos_props.get('d_HtoS_road', np.zeros(self._hos_num))),
                                       dtype=np.float32)
        self._d_HtoS_road = np.asarray(hos_props.get('d_HtoS_road',
                                                     self._d_HtoS_euc),
                                       dtype=np.float32)
        # 평균 transport time (단위: 분, mean of lognormal underlying)
        amb_props = unwrapped.en_manager.en_properties.get('ambulance', {})
        uav_props = unwrapped.en_manager.en_properties.get('uav', {})

        amb_t = amb_props.get('amb_HtoS_t', None)
        if amb_t is not None and len(amb_t[0]) == self._hos_num:
            self._eta_amb = np.asarray(amb_t[0], dtype=np.float32)
        else:
            v_amb = float(amb_props.get('amb_v', 40)) or 40.0
            self._eta_amb = self._d_HtoS_road * 60.0 / v_amb
        uav_t = uav_props.get('uav_HtoS_t', None)
        if uav_t is not None and len(uav_t[0]) == self._hos_num:
            self._eta_uav = np.asarray(uav_t[0], dtype=np.float32)
        else:
            v_uav = float(uav_props.get('uav_v', 80)) or 80.0
            self._eta_uav = self._d_HtoS_euc * 60.0 / v_uav

        # ---------- 4) obs key 순서 + base flat dim ----------
        self._obs_keys = list(env.observation_space.spaces.keys())
        flat_base_dim = sum(int(np.prod(env.observation_space.spaces[k].shape))
                            for k in self._obs_keys)

        # ---------- 5) enrich feature dim ----------
        # 정적 per-hospital (3*H): helipad, eta_amb, eta_uav  ─ 학습 시 정규화 위해 scale 조정
        # 동적 per-hospital (1*H): capacity_remaining = max(hos_max_send - p_sent, 0)
        # 글로벌 (4): [n_red_avail, n_yellow_avail, n_amb_idle, n_uav_idle]
        self._enrich_dim = 3 * self._hos_num + 1 * self._hos_num + 4
        self._flat_base_dim = flat_base_dim
        self._flat_dim = flat_base_dim + self._enrich_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._flat_dim,), dtype=np.float32,
        )

        topk_label = f"topk={self._topk}" if self._topk else "topk=off"
        print(
            f"[EnrichedObsMaskWrapper] {mode_label}, action_space=Discrete({self._n_actions}), "
            f"obs={self._flat_dim} (base={flat_base_dim} + enrich={self._enrich_dim}), "
            f"H={self._hos_num}, helipad={int(self._helipad_mask.sum())}/"
            f"{self._hos_num}, {topk_label}"
        )

    # ---------- decode/encode (FlattenAndDiscreteWrapper 와 동치) ----------
    def _decode(self, action: int):
        a = int(action)
        if self._fixed_mode is not None:
            n_dest = self._effective_nvec[1]
            c = a // n_dest
            d = a % n_dest
            return [c, d, self._fixed_mode]
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        d = rem // n_mode
        m = rem % n_mode
        return [c, d, m]

    def _encode(self, decoded):
        c, d, m = int(decoded[0]), int(decoded[1]), int(decoded[2])
        if self._fixed_mode is not None:
            return c * self._effective_nvec[1] + d
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        return c * (n_dest * n_mode) + d * n_mode + m

    decode_action = _decode
    encode_action = _encode

    # ---------- enrichment feature 생성 ----------
    def _enrich_features(self) -> np.ndarray:
        """현재 시뮬 상태로부터 동적 강화 feature 생성."""
        full = self.env.unwrapped.en_manager.get_full_obs()
        p_sent = np.asarray(full['p_sent'], dtype=np.float32)
        # capacity 잔여 (음수 방지)
        cap_remain = np.maximum(self._hos_max_send - p_sent, 0.0)

        # ETA 는 시뮬 진행에 따라 변하지 않는 정적 값 (사고지점 ↔ 병원 거리/속도)
        # 작은 정규화: 분단위 그대로 두되 큰 값을 줄이기 위해 /60 (시간 단위)
        eta_amb_h = self._eta_amb / 60.0
        eta_uav_h = self._eta_uav / 60.0

        # 글로벌: 가용 환자/자원 수
        p_at_site = np.asarray(full.get('p_wait', None), dtype=object)
        try:
            n_red = float(len(full['p_wait'][0][0]))
            n_yel = float(len(full['p_wait'][1][0]))
        except Exception:
            n_red = n_yel = 0.0
        try:
            n_amb_idle = float(len(full['amb_wait'][0]))
            n_uav_idle = float(len(full['uav_wait'][0]))
        except Exception:
            n_amb_idle = n_uav_idle = 0.0

        glob = np.array([n_red, n_yel, n_amb_idle, n_uav_idle], dtype=np.float32)
        return np.concatenate([
            self._helipad_mask,   # (H,)
            eta_amb_h,            # (H,)
            eta_uav_h,            # (H,)
            cap_remain,           # (H,)
            glob,                 # (4,)
        ]).astype(np.float32)

    def _make_flat_obs(self, base_obs: dict) -> np.ndarray:
        base = _flatten_dict_obs(base_obs, self._obs_keys)
        enrich = self._enrich_features()
        return np.concatenate([base, enrich]).astype(np.float32)

    # ---------- gym API ----------
    def step(self, action):
        decoded = self._decode(action)
        obs, reward, terminated, truncated, info = self.env.step(decoded)
        return self._make_flat_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._make_flat_obs(obs), info

    # ---------- action mask (강화판) ----------
    def action_masks(self) -> np.ndarray:
        """기본 joint mask 위에 helipad / topk-ETA 제약을 추가로 적용.

        - UAV(mode=1) 는 helipad 보유 병원만 합법
        - capacity 0 (= p_sent >= max_send) 병원 제외 — base mask 가 이미 처리
        - top-k 만 후보: mode 별 ETA 짧은 k 개 병원만 합법 (stay 제외)
        """
        # (3, H+1, 2) 결합 mask
        full = self.env.unwrapped.action_masks_joint().reshape(
            self._orig_nvec[0], self._orig_nvec[1], self._orig_nvec[2]
        ).copy()

        H = self._hos_num
        non_helipad = self._helipad_mask < 0.5  # bool (H,)
        # UAV (mode=1): helipad 없는 병원 → 금지
        if non_helipad.any():
            # dest 1..H 가 병원 1..H 에 대응
            full[:, 1:1 + H, 1][:, non_helipad] = False

        # top-k ETA gate
        if self._topk is not None and self._topk < H:
            k = self._topk
            # mode=0 (AMB) ETA: amb HtoS
            amb_order = np.argsort(self._eta_amb)
            amb_allowed = np.zeros(H, dtype=bool)
            amb_allowed[amb_order[:k]] = True
            full[:, 1:1 + H, 0] &= amb_allowed[None, :]
            # mode=1 (UAV) ETA: uav HtoS — helipad 보유 병원 중 가까운 k 개
            uav_eta_eff = self._eta_uav.copy()
            uav_eta_eff[non_helipad] = np.inf  # 헬기장 없음 → 사실상 후보에서 제외
            uav_order = np.argsort(uav_eta_eff)
            uav_allowed = np.zeros(H, dtype=bool)
            uav_allowed[uav_order[:k]] = True
            full[:, 1:1 + H, 1] &= uav_allowed[None, :]

        # mode 차원 슬라이싱 (UAV-only / AMB-only)
        if self._fixed_mode is not None:
            sliced = full[:, :, self._fixed_mode]  # (3, H+1)
            mask = sliced.reshape(-1)
        else:
            mask = full.reshape(-1)

        # 모든 액션이 금지되면 stay 만 허용 (정책 분포 NaN 방지)
        if not mask.any():
            # stay 는 (c, 0, m). 어떤 c, m 이든 stay 1 개 이상 허용해야 함.
            if self._fixed_mode is not None:
                # (3, H+1) → flatten: index = c * (H+1) + 0
                mask = mask.copy()
                for c in range(self._orig_nvec[0]):
                    mask[c * (H + 1)] = True
            else:
                mask = mask.copy()
                for c in range(self._orig_nvec[0]):
                    for m in range(self._orig_nvec[2]):
                        # idx = c * (H+1) * 2 + 0 * 2 + m
                        idx = c * (H + 1) * 2 + m
                        mask[idx] = True
        return mask
