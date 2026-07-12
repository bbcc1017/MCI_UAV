"""v5 공정비교 하네스 — 마스크·dt 를 함께 저장하는 ReplayBuffer.

off-policy(MaskedDQN/MaskedQRDQN/SACDiscrete)의 타깃 계산에는 next-state 유효행동
마스크가 필요한데 flat obs(355)로는 재계산 불가(helipad 열 부재) → 수집 시점 저장.

배선 계약(구현자 준수):
  - next_masks/dts: `MaskInfoWrapper` 가 주입한 infos[i]["action_mask"]/["dt"] 에서 추출.
  - masks(현재 상태 s_t 의 마스크, 행동선택에 실제 쓴 것): 트레이너의 `_sample_action`
    오버라이드가 행동선택 직후 `buffer._pending_cur_masks = masks(np.bool_, (n_envs, A))`
    로 걸어두면 같은 스텝의 add() 가 소비한다(OffPolicyAlgorithm 은 _sample_action →
    env.step → _store_transition(add) 순서가 한 스텝 안에서 보장됨). 미세팅 시
    all-True 폴백 + 1회 경고(SAC actor loss 는 all-True 면 무마스크로 퇴화하므로 주의).
  - optimize_memory_usage=False 전제(assert). SB3 2.9.0 의 `_get_samples` 를
    복제·확장(부모가 env_indices 를 내부 난수로 뽑아 마스크 정렬이 불가능하므로).
"""
import warnings
from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer


class MaskedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor        # (B,1) — timeout 제외 순수 종결
    rewards: th.Tensor      # (B,1)
    masks: th.Tensor        # (B,A) bool — s_t 유효행동
    next_masks: th.Tensor   # (B,A) bool — s_{t+1} 유효행동(타깃 max/기대값용)
    dts: th.Tensor          # (B,1) float — 결정 간 sim 경과분(SMDP γ^Δt 용)


class MaskedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True):
        assert not optimize_memory_usage, "MaskedReplayBuffer 는 memopt 미지원(마스크 정렬)"
        assert isinstance(action_space, spaces.Discrete), "Discrete 행동공간 전용"
        super().__init__(buffer_size, observation_space, action_space, device=device,
                         n_envs=n_envs, optimize_memory_usage=False,
                         handle_timeout_termination=handle_timeout_termination)
        self.mask_dim = int(action_space.n)
        self.masks = np.ones((self.buffer_size, self.n_envs, self.mask_dim), dtype=np.bool_)
        self.next_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dim), dtype=np.bool_)
        self.dts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._pending_cur_masks: Optional[np.ndarray] = None
        self._warned_no_cur = False
        self._warned_no_next = False

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos  # super().add 가 pos 를 증가시키므로 먼저 고정
        ones = np.ones(self.mask_dim, dtype=np.bool_)

        cur = self._pending_cur_masks
        if cur is None:
            if not self._warned_no_cur:
                warnings.warn("MaskedReplayBuffer: _pending_cur_masks 미세팅 — all-True 폴백"
                              "(트레이너 _sample_action 배선 확인)")
                self._warned_no_cur = True
            cur = np.ones((self.n_envs, self.mask_dim), dtype=np.bool_)
        self.masks[pos] = np.asarray(cur, dtype=np.bool_).reshape(self.n_envs, self.mask_dim)
        self._pending_cur_masks = None

        nxt, dts = [], []
        for info in infos:
            m = info.get("action_mask")
            if m is None:
                if not self._warned_no_next:
                    warnings.warn("MaskedReplayBuffer: infos 에 action_mask 없음 — "
                                  "MaskInfoWrapper 체인 확인(all-True 폴백)")
                    self._warned_no_next = True
                m = ones
            nxt.append(np.asarray(m, dtype=np.bool_))
            dts.append(float(info.get("dt", 0.0)))
        self.next_masks[pos] = np.stack(nxt)
        self.dts[pos] = np.asarray(dts, dtype=np.float32)

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> MaskedReplayBufferSamples:
        # SB3 2.9.0 ReplayBuffer._get_samples 복제 + 마스크·dt 확장(env_indices 정렬 유지)
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices]
             * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        obs_t, act_t, nobs_t, done_t, rew_t = tuple(map(self.to_torch, data))
        masks_t = th.as_tensor(self.masks[batch_inds, env_indices], device=self.device)
        nmasks_t = th.as_tensor(self.next_masks[batch_inds, env_indices], device=self.device)
        dts_t = th.as_tensor(self.dts[batch_inds, env_indices].reshape(-1, 1),
                             device=self.device)
        return MaskedReplayBufferSamples(obs_t, act_t, nobs_t, done_t, rew_t,
                                         masks_t, nmasks_t, dts_t)
