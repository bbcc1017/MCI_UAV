"""v5 공정비교 하네스 — 하드 마스킹 Double-DQN(off-policy 베이스라인).

목적
----
챔피언(MaskablePPO)과 **동일 env 스택**(essential+load obs 355 · Discrete 192 ·
occ 게이트) 위에서 학습·평가 **양쪽에 action mask 를 하드 적용**하는 DQN.
v1 `train_dqn.py` 의 DoubleDQN(과대추정 완화)을 계승하되, v1 이 "무효행동=시뮬
no-op 자연 페널티"에 의존했던 것과 달리 여기서는 **행동선택·타깃 계산 모두에서
무효행동을 −inf 로 배제**한다("masked 동물원"의 기여점, 계획 §3.1-2).

계약(트레이너↔버퍼, `masked_replay_buffer.py` docstring 과 짝)
--------------------------------------------------------------
  - `_sample_action` 이 행동선택 직전 live env(`self._mask_env.env_method`)에서
    현재 상태 s_t 의 마스크를 계산해 `self.replay_buffer._pending_cur_masks`
    (np.bool_, (n_envs, A))로 걸어두면, 같은 스텝의 `add()` 가 이를 `masks[pos]`
    로 소비한다(OffPolicyAlgorithm 은 _sample_action → env.step → add 를 한 스텝
    안에서 보장). next-state 마스크·dt 는 `MaskInfoWrapper` 가 info 로 실어 버퍼가
    저장 → `train()` 이 타깃에서 소비.
  - obs 정규화: 수집 obs 는 VecNormalize 가 정규화한 값이 `self._last_obs` 로
    들어오고, 버퍼는 원본(비정규화)을 저장 후 sample 시 `_normalize_obs` 로 재적용
    (SB3 표준). 따라서 `_sample_action`/`train` 의 q_net 입력은 항상 정규화 obs.

재사용처
--------
  - 트레이너 `train_zoo.py --algo dqn [--smdp]` 가 net(pointer/mlp)·하이퍼를 넘겨 생성.
  - 평가 어댑터(`evaluate.dqn_policy` 후속)가 `predict_masked(obs, mask)` 호출
    — obs 는 호출자(동결 VecNormalize)가 이미 정규화해 전달하므로 내부 정규화 없음.

SB3 2.9.0 전제(서브클래스 지점): `off_policy_algorithm.collect_rollouts`(env 첫
인자)·`_sample_action`(반환 (action, buffer_action))·`dqn.DQN.train`. super() 최소침습.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import DQN

from masked_replay_buffer import MaskedReplayBuffer


class MaskedDQN(DQN):
    """하드 마스킹 Double-DQN.

    Parameters
    ----------
    policy, env : SB3 DQN 규약과 동일.
    smdp : bool
        True 면 타깃 할인을 결정 간 sim 경과분 Δt 로 γ^Δt(SMDP) 적용
        (버퍼가 저장한 dts 사용). False(기본)=결정당 고정 γ.
    **kwargs : DQN 하이퍼 그대로 전달. replay_buffer_class 는 미지정 시
        MaskedReplayBuffer 로 강제.
    """

    def __init__(self, policy, env, smdp: bool = False, **kwargs):
        kwargs.setdefault("replay_buffer_class", MaskedReplayBuffer)
        super().__init__(policy, env, **kwargs)
        self.smdp = bool(smdp)
        self._mask_env = None  # collect_rollouts 가 매 롤아웃 시작에 스태시

    # ---------- 롤아웃: live 마스크 접근용 env 스태시 ----------
    def collect_rollouts(self, env, *args, **kwargs):
        # env = self.env(VecNormalize) — env_method("action_masks") 위임 접근 가능.
        self._mask_env = env
        return super().collect_rollouts(env, *args, **kwargs)

    # ---------- 행동선택(warmup/ε=유효행동 균등, else masked-greedy) ----------
    def _sample_action(self, learning_starts, action_noise=None, n_envs: int = 1):
        mask_env = self._mask_env if self._mask_env is not None else self.env
        # s_t 의 마스크(env.step 이전이라 현재 상태) → 버퍼가 이 스텝의 masks 로 소비.
        masks = np.stack(mask_env.env_method("action_masks")).astype(bool)  # (n_envs, A)
        self.replay_buffer._pending_cur_masks = masks

        warmup = self.num_timesteps < learning_starts
        if warmup or np.random.rand() < self.exploration_rate:
            # 유효행동(마스크 True) 중 env 별 균등 샘플 — 무효행동 탐색 낭비 제거.
            actions = np.array([self._sample_valid(masks[i]) for i in range(n_envs)],
                               dtype=np.int64)
        else:
            actions = self._masked_greedy(self._last_obs, masks)
        return actions, actions  # Discrete: buffer_action == action

    @staticmethod
    def _sample_valid(mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0  # 이론상 stay 항상 유효라 도달 불가(방어)
        return int(np.random.choice(valid))

    def _masked_greedy(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """정규화 obs → q_net → 무효행동 −inf → argmax (env 별)."""
        self.policy.set_training_mode(False)
        obs_t, _ = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            q = self.q_net(obs_t).cpu().numpy()  # (n_envs, A)
        q = np.where(masks, q, -np.inf)
        a = np.argmax(q, axis=1)
        # 전부 −inf 인 행(유효행동 0개) 방어: stay(0)
        allbad = ~np.isfinite(q).any(axis=1)
        a[allbad] = 0
        return a.astype(np.int64)

    # ---------- 학습(Double DQN + next_masks 마스킹 + γ^Δt 훅) ----------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # SMDP: 결정 간 Δt 로 γ^Δt(Δt=분, γ^0=1 정합). 아니면 고정 γ.
            discounts = (self.gamma ** rd.dts) if self.smdp else self.gamma

            with th.no_grad():
                # Double DQN: 행동 선택은 online q_net(+마스킹), 가치 평가는 target net.
                next_q_online = self.q_net(rd.next_observations)  # (B, A)
                neg = th.full_like(next_q_online, float("-inf"))
                next_q_online = th.where(rd.next_masks, next_q_online, neg)
                # 유효행동 0개 행 방어(이론상 stay 로 없음): argmax 안전화(0 대체).
                no_valid = ~th.isfinite(next_q_online).any(dim=1)
                if no_valid.any():
                    next_q_online[no_valid] = 0.0
                next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)
                next_q = self.q_net_target(rd.next_observations)          # (B, A)
                next_q = th.gather(next_q, dim=1, index=next_actions)     # (B, 1)
                # done 행은 (1-done) 으로 0화 → no_valid(=주로 종결) 무해.
                target_q = rd.rewards + (1 - rd.dones) * discounts * next_q

            current_q = self.q_net(rd.observations)
            current_q = th.gather(current_q, dim=1, index=rd.actions.long())

            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    # ---------- 평가용 마스크드 예측(전 v5 알고 공통 계약) ----------
    def predict_masked(self, obs: np.ndarray, mask: np.ndarray,
                       deterministic: bool = True) -> int:
        """단일 상태에서 마스크드 argmax 행동(int) 반환.

        obs 는 호출자가 이미 정규화(동결 VecNormalize)해 넘긴다 → 내부 정규화 없음.
        deterministic 인자는 규약 통일용(항상 greedy).
        """
        obs_t, _ = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            q = self.q_net(obs_t).cpu().numpy().reshape(-1)
        q_masked = np.where(np.asarray(mask, dtype=bool), q, -np.inf)
        if not np.isfinite(q_masked).any():
            return 0  # stay 폴백(도달 불가)
        return int(np.argmax(q_masked))

    # ---------- 저장 제외(smdp 는 data 로 자동 저장·복원) ----------
    def _excluded_save_params(self):
        # _mask_env(VecEnv 참조)는 피클 불가/불요 → 제외. smdp(bool)은 __dict__ 에
        # 남아 SB3 save/load 로 자동 왕복.
        return [*super()._excluded_save_params(), "_mask_env"]
