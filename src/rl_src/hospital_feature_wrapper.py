"""Phase 3a — 특징기반 병원 obs 래퍼 (Option A: 전체 H 병원, 슬롯축소 없음).

랩 피드백 #1(ETA=lognormal 평균)·#2(tier를 obs로)·#3(local/comms 정보수준)을 RL obs 로
해결한다. 인덱스 기반 h_states/p_sent 대신 **병원당 특징 엔티티 행렬 (H, F)** 로 표현 →
정책이 병원 "특징"을 읽어 일반화하고, VIPER 트리가 해석 가능한 특징으로 분기한다.

설계 원칙 (Phase 1/2 와 동일):
  * sim_src 동역학·env_wrapper.py 코어·multi_region_env.py **무수정**.
  * FlattenAndDiscreteWrapper 의 decode/encode/fixed-mode·tier 마스킹과 **동치** 동작.
  * action 은 Discrete(H+1=25) 유지(슬롯→idx 역매핑 없음 — H 고정이라 차원통일 이미 해결).
  * 자체적으로 compact obs 를 만들므로 MCI_REDUCED_OBS(AggregateObsWrapper) 불필요
    — 다만 글로벌 집계는 AggregateObsWrapper._patient_agg/_fleet_agg 를 재사용한다.

병원당 특징 F (MCI_OBS_VARIANT 로 토글):
  essential(기본): [is_tier3, cap_remain, eta_amb, eta_uav] — 중복 제거 최소핵심(F=4).
  full(ablation):  [is_tier3, helipad, eta_amb, eta_uav, idle, queue, occ, cap_remain] (F=8).
  local/comms(ablation): 위 8열을 정적4/실시간4 로 분리.
  - 미설정/"essential" → essential. helipad 는 UAV 마스크가 강제(중복), idle/occ 는
    cap_remain 과 affine 중복, queue 는 ablation 무신호라 essential 에서 제외.
  - ETA = amb/uav_HtoS_t[0](=lognormal 평균=사전계산 deterministic, #1). 시나리오 최소
    ETA 로 정규화(최근접=1) 후 MCI_ETA_CLIP(기본 10배) 클립. 정적이라 1회 캐시.
  - cap_remain = max(hos_max_send - cap_used, 0). cap_used=occ(기본)|p_sent(psent게이트).

글로벌 특징 (병원 비의존): patient_agg(R/Y 2등급×5단계=10) + vehicle_agg(10) + time(1) = 21.
  - Green/Black 은 행동대상 아님(R/Y 소진+구조완료 시 sim 코어가 자동일괄 이송) → patient_agg
    에서 제거(R/Y 만). p_at_site·n_amb/uav_at_site 는 patient_agg stage1·vehicle_agg n_avail
    의 부분집합이라 제거(중복). raw h_states/p_sent 는 엔티티로 흡수.

행동 마스크: tier(Red→Tier3, MCI_TIER_MASK) + Green 이송 차단(MCI_GREEN_MASK) + helipad/capa
  (joint). train 스크립트에서 ActionMasker 와 함께 사용(FlattenAndDiscreteWrapper 대체).
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aggregate_obs import AggregateObsWrapper  # _patient_agg / _fleet_agg 재사용

# 병원당 특징 열 정의 (순서 고정)
# essential(기본): 중복 제거 최소핵심. helipad(마스크중복)·idle/occ(cap_remain과 affine중복)·
#   queue(ablation 무신호) 제거 → capability+여유+AMB도달+UAV도달.
_ESSENTIAL_COLS = ["is_tier3", "cap_remain", "eta_amb", "eta_uav"]
_LOCAL_COLS = ["is_tier3", "helipad", "eta_amb", "eta_uav"]   # 정적 사전지식 (ablation)
_COMMS_COLS = ["idle", "queue", "occ", "cap_remain"]          # 실시간 동적 (ablation)
# 글로벌: patient_agg(R/Y 2등급×5단계=10) + vehicle_agg(10) + time(1).
#   p_at_site·n_amb_at_site·n_uav_at_site 는 각각 patient_agg stage1·vehicle_agg n_avail 의
#   정확한 부분집합이라 제거(0손실). Green/Black 은 행동대상 아님(자동일괄 start_GB_transport)이라
#   patient_agg 에서 제거 — R/Y 만 유지.
_GLOBAL_DIM = 10 + 10 + 1


def _parse_variant():
    raw = os.environ.get("MCI_OBS_VARIANT", "").strip().lower()
    return set(t for t in raw.replace(",", "+").split("+") if t)


class HospitalFeatureWrapper(gym.Wrapper):
    """병원당 특징 엔티티 obs + Discrete action + 결합 마스크 (FlattenAndDiscreteWrapper 대체).

    Parameters
    ----------
    env : MCIEnvironment_gym
        ``make_base_env(cfg)`` 로 만든 raw base env (FlattenAndDiscreteWrapper 미적용).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # ---------- 1) action 차원 (FlattenAndDiscreteWrapper 동등 로직) ----------
        nvec = env.action_space.nvec.tolist()  # [3, H+1, 2]
        assert len(nvec) == 3, f"기대 형식 [class, dest, mode], got {nvec}"
        self._orig_nvec = nvec
        self.H = nvec[1] - 1

        u = env.unwrapped
        amb_num = int(getattr(u, "amb_num", 0))
        uav_num = int(getattr(u, "uav_num", 0))
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

        # ---------- 2) 정적 병원 특징 (시나리오 상수, 1회 캐시) ----------
        hp = u.en_manager.en_properties['hospital']
        hos_tier = np.asarray(hp['hos_tier'], dtype=np.float32).reshape(-1)  # 3=Tier3, 2=그외
        self._is_tier3 = (hos_tier == 3).astype(np.float32)                  # (H,)
        helipad_idx = np.asarray(hp.get('hos_helipad_idx', np.array([])), dtype=int)
        self._helipad = np.zeros(self.H, dtype=np.float32)
        if helipad_idx.size > 0:
            self._helipad[helipad_idx] = 1.0
        self._max_send = np.asarray(hp['hos_max_send'], dtype=np.float32).reshape(-1)

        # ETA(분) = lognormal 평균(amb/uav_HtoS_t[0]) — 없으면 거리/속도로 폴백. (#1)
        ambp = u.en_manager.en_properties.get('ambulance', {})
        uavp = u.en_manager.en_properties.get('uav', {})
        d_road = np.asarray(hp.get('d_HtoS_road', hp.get('d_HtoS_euc', np.zeros(self.H))), dtype=np.float32)
        d_euc = np.asarray(hp.get('d_HtoS_euc', d_road), dtype=np.float32)
        amb_t = ambp.get('amb_HtoS_t', None)
        if amb_t is not None and len(amb_t[0]) == self.H:
            eta_amb = np.asarray(amb_t[0], dtype=np.float32)
        else:
            eta_amb = d_road * 60.0 / (float(ambp.get('amb_v', 40)) or 40.0)
        uav_t = uavp.get('uav_HtoS_t', None)
        if uav_t is not None and len(uav_t[0]) == self.H:
            eta_uav = np.asarray(uav_t[0], dtype=np.float32)
        else:
            eta_uav = d_euc * 60.0 / (float(uavp.get('uav_v', 80)) or 80.0)
        # 시나리오 최소 ETA 로 정규화(>0 기준, 최근접=1) → 지역간 스케일 제거.
        # + 외곽 병원 이상치 클립(최근접의 MCI_ETA_CLIP 배, 기본 10) → VecNorm std 왜곡 방지.
        eta_clip = float(os.environ.get("MCI_ETA_CLIP", "10.0"))
        self._eta_amb = np.minimum(self._norm_by_min(eta_amb), eta_clip).astype(np.float32)
        self._eta_uav = np.minimum(self._norm_by_min(eta_uav), eta_clip).astype(np.float32)

        # ---------- 3) MCI_OBS_VARIANT → 특징 열 선택 (local/comms/full) ----------
        toks = _parse_variant()
        if "full" in toks:
            self._cols = _LOCAL_COLS + _COMMS_COLS
            var_label = "full(ablation)"
        elif "local" in toks and "comms" not in toks:
            self._cols = list(_LOCAL_COLS)
            var_label = "local(ablation)"
        elif "comms" in toks and "local" not in toks:
            self._cols = list(_COMMS_COLS)
            var_label = "comms(ablation)"
        else:  # 기본 = essential (essential 토큰 또는 미설정)
            self._cols = list(_ESSENTIAL_COLS)
            var_label = "essential"
        self._F = len(self._cols)

        # ---------- 4) obs space ----------
        self._flat_dim = self.H * self._F + _GLOBAL_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._flat_dim,), dtype=np.float32,
        )
        self._ct_cache = None  # 등급-tier 치료가능 마스크 (3, H)

        print(f"[HospitalFeatureWrapper] {mode_label}, action=Discrete({self._n_actions}), "
              f"obs={self._flat_dim} (entity {self.H}x{self._F} + global {_GLOBAL_DIM}), "
              f"variant={var_label}, helipad={int(self._helipad.sum())}/{self.H}")

    @staticmethod
    def _norm_by_min(eta: np.ndarray) -> np.ndarray:
        pos = eta[eta > 0]
        denom = float(pos.min()) if pos.size else 1.0
        return (eta / denom).astype(np.float32)

    # ---------- decode/encode (FlattenAndDiscreteWrapper 와 동치) ----------
    def _decode(self, action: int):
        a = int(action)
        if self._fixed_mode is not None:
            n_dest = self._effective_nvec[1]
            return [a // n_dest, a % n_dest, self._fixed_mode]
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        c = a // (n_dest * n_mode)
        rem = a % (n_dest * n_mode)
        return [c, rem // n_mode, rem % n_mode]

    def _encode(self, decoded):
        c, d, m = int(decoded[0]), int(decoded[1]), int(decoded[2])
        if self._fixed_mode is not None:
            return c * self._effective_nvec[1] + d
        n_dest, n_mode = self._orig_nvec[1], self._orig_nvec[2]
        return c * (n_dest * n_mode) + d * n_mode + m

    decode_action = _decode
    encode_action = _encode

    # ---------- obs 구성 ----------
    def _entity(self, obs: dict) -> np.ndarray:
        """병원당 특징 행렬 (H, F)."""
        h = np.asarray(obs['h_states'], dtype=np.float32)  # (H,3) = [idle, queue, occ]
        p_sent = np.asarray(obs['p_sent'], dtype=np.float32).reshape(-1)
        # cap_remain 도 게이트 기준에 맞춤: occ(실시간 잔여) | psent(보낸 만큼 차감). 마스크와 동일 의미.
        cap_used = h[:, 2] if os.environ.get("MCI_CAP_GATE", "occ").strip().lower() != "psent" else p_sent
        cap_remain = np.maximum(self._max_send - cap_used, 0.0)
        col_map = {
            "is_tier3": self._is_tier3,
            "helipad": self._helipad,
            "eta_amb": self._eta_amb,
            "eta_uav": self._eta_uav,
            "idle": h[:, 0],
            "queue": h[:, 1],
            "occ": h[:, 2],
            "cap_remain": cap_remain,
        }
        return np.stack([col_map[c] for c in self._cols], axis=1).astype(np.float32)  # (H, F)

    def _globals(self, obs: dict) -> np.ndarray:
        # patient_agg 4등급×5단계(20) 중 R/Y(앞 2등급=10)만 — Green/Black 은 행동대상 아님(자동일괄).
        pa = AggregateObsWrapper._patient_agg(np.asarray(obs['p_states']))[:10]       # (10,) R/Y
        va = np.concatenate([
            AggregateObsWrapper._fleet_agg(np.asarray(obs['amb_states'])),
            AggregateObsWrapper._fleet_agg(np.asarray(obs['uav_states'])),
        ])                                                                            # (10,)
        # p_at_site/n_amb_at_site/n_uav_at_site 는 pa·va 의 부분집합이라 제거(중복 0손실).
        return np.concatenate([
            pa, va,
            np.asarray(obs['time'], dtype=np.float32).reshape(-1),                    # (1,)
        ]).astype(np.float32)

    def _flat_obs(self, obs: dict) -> np.ndarray:
        return np.concatenate([self._entity(obs).reshape(-1), self._globals(obs)]).astype(np.float32)

    # ---------- gym API ----------
    def step(self, action):
        decoded = self._decode(action)
        obs, reward, terminated, truncated, info = self.env.step(decoded)
        return self._flat_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flat_obs(obs), info

    # ---------- action mask (env_wrapper.action_masks 와 동치: joint + tier) ----------
    def _can_treat_mask(self) -> np.ndarray:
        if self._ct_cache is None:
            ep = self.env.unwrapped.en_manager.en_properties
            hos_tier = np.asarray(ep['hospital']['hos_tier']).reshape(-1)
            pinfo = ep['patient']['patient_info']
            t3 = np.asarray(pinfo['treat_tier3']).astype(bool)
            t2 = np.asarray(pinfo['treat_tier2']).astype(bool)
            ct = np.zeros((3, self.H), dtype=bool)
            for h in range(self.H):
                ht = int(hos_tier[h])
                col = t3 if ht == 3 else (t2 if ht == 2 else np.zeros(4, dtype=bool))
                ct[:, h] = col[:3]
            self._ct_cache = ct
        return self._ct_cache

    def action_masks(self) -> np.ndarray:
        full = self.env.unwrapped.action_masks_joint()
        full = full.reshape(self._orig_nvec[0], self._orig_nvec[1], self._orig_nvec[2]).copy()
        if os.environ.get("MCI_TIER_MASK", "1") != "0":
            ct = self._can_treat_mask()           # (3, H) bool
            full[:, 1:, :] &= ct[:, :, None]      # dest 1..H 만 차단, stay(0) 유지
        # Green(class=2) 이송 차단 → 자동일괄(start_GB_transport)에 위임. R/Y 만 행동대상.
        # (Black 은 행동공간(class dim=3)에 애초에 없음. stay(dest=0)는 유지해 합법행동 보장.)
        if os.environ.get("MCI_GREEN_MASK", "1") != "0":
            full[2, 1:, :] = False
        if self._fixed_mode is not None:
            return full[:, :, self._fixed_mode].reshape(-1)
        return full.reshape(-1)
