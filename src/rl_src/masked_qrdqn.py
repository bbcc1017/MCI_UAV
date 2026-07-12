"""v5 공정비교 하네스 — 하드 마스킹 QR-DQN(분포형 off-policy 베이스라인).

목적
----
`MaskedDQN` 과 동일한 마스킹 계약을 sb3-contrib QR-DQN 위에 이식한 분포형 가치기반
베이스라인. 행동선택·타깃 계산 모두에서 무효행동을 −inf 로 배제하고, 큰 행동공간
(192)에서의 CPU 병목을 피하려 `n_quantiles` 를 트레이너(train_zoo)가 넘긴다
(기본 강제 금지 — 계획 §3.2 #5, sb3 QRDQN 기본 200 유지). 값 추정은 분위수 평균 Q.

MaskedDQN 과의 차이(주석으로 명시)
----------------------------------
  - Q = 분위수(quantile) 평균. 행동선택/타깃 next-action 모두 **quantile 평균 Q +
    마스크 −inf argmax**.
  - 타깃 next-action 선택은 **target net** 으로 수행(표준 QR-DQN 타깃 스타일).
    DoubleDQN(online 선택·target 평가) 변형은 MaskedDQN 에만 적용, 여기선 미적용.
  - 손실은 `quantile_huber_loss`(분위수 합). SMDP 할인 γ^Δt 는 (B,1) 로 분위수축
    브로드캐스트.

계약·재사용처·SB3 전제는 `masked_dqn.py`/`masked_replay_buffer.py` docstring 과 동일.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch as th

from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss

from masked_replay_buffer import MaskedReplayBuffer


class MaskedQRDQN(QRDQN):
    """하드 마스킹 QR-DQN.

    Parameters
    ----------
    policy, env : SB3 QRDQN 규약과 동일.
    smdp : bool
        True 면 타깃 할인을 γ^Δt(SMDP, 버퍼 dts). False(기본)=고정 γ.
    **kwargs : QRDQN 하이퍼 그대로. `n_quantiles` 는 policy_kwargs 로 넘김
        (클래스에서 강제하지 않음). replay_buffer_class 미지정 시 MaskedReplayBuffer.
    """

    def __init__(self, policy, env, smdp: bool = False, **kwargs):
        kwargs.setdefault("replay_buffer_class", MaskedReplayBuffer)
        super().__init__(policy, env, **kwargs)
        self.smdp = bool(smdp)
        self._mask_env = None

    # ---------- 롤아웃: live 마스크 접근용 env 스태시 ----------
    def collect_rollouts(self, env, *args, **kwargs):
        self._mask_env = env
        return super().collect_rollouts(env, *args, **kwargs)

    # ---------- 행동선택(warmup/ε=유효행동 균등, else masked-greedy) ----------
    def _sample_action(self, learning_starts, action_noise=None, n_envs: int = 1):
        mask_env = self._mask_env if self._mask_env is not None else self.env
        masks = np.stack(mask_env.env_method("action_masks")).astype(bool)  # (n_envs, A)
        self.replay_buffer._pending_cur_masks = masks

        warmup = self.num_timesteps < learning_starts
        if warmup or np.random.rand() < self.exploration_rate:
            actions = np.array([self._sample_valid(masks[i]) for i in range(n_envs)],
                               dtype=np.int64)
        else:
            actions = self._masked_greedy(self._last_obs, masks)
        return actions, actions

    @staticmethod
    def _sample_valid(mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return 0
        return int(np.random.choice(valid))

    def _masked_greedy(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """정규화 obs → quantile 평균 Q → 무효행동 −inf → argmax (env 별)."""
        self.policy.set_training_mode(False)
        obs_t, _ = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            # quantile_net(obs): (n_envs, n_quantiles, A) → 분위수 평균 = Q(s,·)
            q = self.quantile_net(obs_t).mean(dim=1).cpu().numpy()  # (n_envs, A)
        q = np.where(masks, q, -np.inf)
        a = np.argmax(q, axis=1)
        allbad = ~np.isfinite(q).any(axis=1)
        a[allbad] = 0
        return a.astype(np.int64)

    # ---------- 학습(표준 QR-DQN 타깃 + next_masks 마스킹 + γ^Δt 훅) ----------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # SMDP: (B,1) γ^Δt → 아래 (B, n_quantiles) 에 브로드캐스트. 아니면 고정 γ.
            discounts = (self.gamma ** rd.dts) if self.smdp else self.gamma

            with th.no_grad():
                # target net 의 다음 상태 분위수 (B, n_q, A)
                next_quantiles = self.quantile_net_target(rd.next_observations)
                # 분위수 평균 Q 로 next-action 선택(+마스크). DoubleDQN 변형은 MaskedDQN 전용.
                mean_q = next_quantiles.mean(dim=1)  # (B, A)
                neg = th.full_like(mean_q, float("-inf"))
                mean_q = th.where(rd.next_masks, mean_q, neg)
                no_valid = ~th.isfinite(mean_q).any(dim=1)
                if no_valid.any():
                    mean_q[no_valid] = 0.0
                next_actions = mean_q.argmax(dim=1)  # (B,)
                # (B,) → (B, n_q, 1) 로 확장해 선택 행동의 분위수 추출
                next_actions = next_actions.view(batch_size, 1, 1).expand(
                    batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(
                    dim=2, index=next_actions).squeeze(dim=2)  # (B, n_q)
                # rewards/dones (B,1), discounts scalar|(B,1) → (B, n_q) 브로드캐스트
                target_quantiles = rd.rewards + (1 - rd.dones) * discounts * next_quantiles

            current_quantiles = self.quantile_net(rd.observations)  # (B, n_q, A)
            actions = rd.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(
                current_quantiles, dim=2, index=actions).squeeze(dim=2)  # (B, n_q)

            loss = quantile_huber_loss(current_quantiles, target_quantiles,
                                       sum_over_quantiles=True)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    # ---------- 평가용 마스크드 예측(quantile 평균 Q 기준, 공통 계약) ----------
    def predict_masked(self, obs: np.ndarray, mask: np.ndarray,
                       deterministic: bool = True) -> int:
        obs_t, _ = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            q = self.quantile_net(obs_t).mean(dim=1).cpu().numpy().reshape(-1)
        q_masked = np.where(np.asarray(mask, dtype=bool), q, -np.inf)
        if not np.isfinite(q_masked).any():
            return 0
        return int(np.argmax(q_masked))

    # ---------- 저장 제외 ----------
    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "_mask_env"]
