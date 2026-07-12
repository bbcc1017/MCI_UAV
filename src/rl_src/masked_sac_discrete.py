"""v5 공정비교 하네스 — 마스크드 이산 SAC(SAC-Discrete).

목적:
  Christodoulou(2019) "SAC for Discrete Action Settings" 를 재난 대응 MCI 도메인의
  하드 마스킹(무효행동 원천 차단) + per-state 목표 엔트로피로 확장한 off-policy
  베이스라인. v5 "masked 동물원"의 SAC 행 — MaskablePPO/Masked-DQN/QRDQN/REINFORCE 와
  동일 env 스택·동일 마스킹 규약(학습·평가 모두 하드 적용)으로 공정비교한다.

계약(v5 하네스 공유):
  * env 체인(outer→inner): VecNormalize → Vec → Monitor → ActionMasker →
    MaskInfoWrapper → HospitalFeatureWrapper → RewardRedesignWrapper → base.
    obs=Box(355, essential+load) / action=Discrete(192, R·Y × dest48 × mode2).
  * 마스크 재료(helipad 등)가 flat obs 에 없어 next-state 마스크를 재계산 불가 →
    `MaskInfoWrapper` 가 info 로 실어 `MaskedReplayBuffer` 가 저장(next_masks).
    현재상태 마스크(masks)는 `_sample_action` 이 행동선택 직후
    `replay_buffer._pending_cur_masks` 로 걸어두면 같은 스텝 add() 가 소비.
  * `MaskedReplayBufferSamples`(observations/actions/next_observations/dones/rewards/
    masks/next_masks/dts) 를 소비 — 이 파일은 masks/next_masks 만 사용(dts=SMDP 는 DQN 전용).
  * `predict_masked(obs, mask, deterministic=True) → int`: 전 v5 드라이버 공통 계약.
    obs 는 호출자가 VecNormalize 로 정규화해 전달(내부 정규화 금지).

재사용처:
  * train_zoo.py(--algo sacd) 가 학습 구동, evaluate.py 의 sacd_policy 어댑터가
    predict_masked 를 fn(obs,mask,unwrapped)→int 규약으로 감싼다.
  * features_extractor 는 policy_kwargs 로 교체 가능(pointer_policy.HospitalTokenExtractor
    공유 — 공정비교 네트 용량 통일). 기본은 FlattenExtractor + MLP[256,256].
  * ⚠️ 평가 로드 시 `from masked_sac_discrete import SACDiscrete` +
    (pointer 사용 시) `from pointer_policy import HospitalTokenExtractor` 선행 필수
    (SB3 zip 이 policy_class/features_extractor_class 를 모듈경로로 역직렬화).

SB3 2.9.0 OffPolicyAlgorithm 서브클래스 관례(off_policy_algorithm.py / sac/ 참조):
  * _setup_model → 버퍼·정책·lr스케줄 생성(super) 후 alias·log_alpha·alpha_opt 생성.
  * _sample_action(off_policy_algorithm.py:367) 오버라이드로 마스킹 샘플/warmup 균등.
  * collect_rollouts 씬 오버라이드로 라이브 마스크 재계산용 env 를 스태시.
  * save/load: _excluded_save_params/_get_torch_save_params(sac.py:322-332 관례).
"""
import os
import sys
import warnings
from typing import Any, ClassVar, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(__file__))  # 형제 모듈(masked_replay_buffer) import 보장

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update

from masked_replay_buffer import MaskedReplayBuffer


# ---------------------------------------------------------------------------
# 네트워크 (BaseModel = nn.Module + extract_features, ABC 아님 → _predict 불요)
# ---------------------------------------------------------------------------
class DiscreteActor(BaseModel):
    """이산 SAC 액터: extractor → MLP → logits(A). 마스킹은 알고리즘 단계에서 적용."""

    def __init__(self, observation_space, action_space, net_arch,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 activation_fn=nn.ReLU, normalize_images: bool = True):
        super().__init__(observation_space, action_space, features_extractor=features_extractor,
                         normalize_images=normalize_images)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.action_dim = int(action_space.n)
        self.latent_pi = nn.Sequential(*create_mlp(features_dim, self.action_dim, net_arch, activation_fn))

    def logits(self, obs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.latent_pi(features)

    def forward(self, obs) -> th.Tensor:
        return self.logits(obs)


class DiscreteCritic(BaseModel):
    """트윈 Q: extractor(내부 공유) → 각 Q-헤드 → Q(A) 벡터(행동별 Q값 동시 출력)."""

    def __init__(self, observation_space, action_space, net_arch,
                 features_extractor: BaseFeaturesExtractor, features_dim: int,
                 activation_fn=nn.ReLU, n_critics: int = 2, normalize_images: bool = True):
        super().__init__(observation_space, action_space, features_extractor=features_extractor,
                         normalize_images=normalize_images)
        self.n_critics = n_critics
        self.action_dim = int(action_space.n)
        self.q_networks: list[nn.Module] = []
        for i in range(n_critics):
            qnet = nn.Sequential(*create_mlp(features_dim, self.action_dim, net_arch, activation_fn))
            self.add_module(f"qf{i}", qnet)
            self.q_networks.append(qnet)

    def forward(self, obs) -> tuple[th.Tensor, ...]:
        # extractor 는 한 번만 통과(두 Q-헤드 공유)
        features = self.extract_features(obs, self.features_extractor)
        return tuple(qnet(features) for qnet in self.q_networks)


# ---------------------------------------------------------------------------
# 정책(actor + twin critic + critic target) — SB3 SACPolicy 관례 미러
# ---------------------------------------------------------------------------
class SACDiscretePolicy(BasePolicy):
    """이산 SAC 정책. actor/critic 은 각자 features_extractor 소유(share=False 기본,
    SB3 SAC 관례 — 단순성 우선). critic_target 은 별도 extractor + polyak 추적."""

    def __init__(self, observation_space, action_space, lr_schedule: Schedule,
                 net_arch: "list[int] | None" = None, activation_fn=nn.ReLU,
                 features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: "dict[str, Any] | None" = None,
                 normalize_images: bool = True,
                 optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: "dict[str, Any] | None" = None,
                 n_critics: int = 2, share_features_extractor: bool = False):
        super().__init__(observation_space, action_space,
                         features_extractor_class=features_extractor_class,
                         features_extractor_kwargs=features_extractor_kwargs,
                         optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs,
                         normalize_images=normalize_images, squash_output=False)
        self.net_arch = [256, 256] if net_arch is None else net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # --- actor (자기 extractor) ---
        actor_fe = self.make_features_extractor()
        self.actor = DiscreteActor(self.observation_space, self.action_space, self.net_arch,
                                   actor_fe, actor_fe.features_dim, self.activation_fn,
                                   self.normalize_images)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)
        # --- critic (share 여부에 따라 extractor 공유/전용) ---
        if self.share_features_extractor:
            self.critic = DiscreteCritic(self.observation_space, self.action_space, self.net_arch,
                                         actor_fe, actor_fe.features_dim, self.activation_fn,
                                         self.n_critics, self.normalize_images)
            # 공유 extractor 는 critic loss 로 학습하지 않음(actor loss 전용)
            critic_params = [p for n, p in self.critic.named_parameters()
                             if "features_extractor" not in n]
        else:
            critic_fe = self.make_features_extractor()
            self.critic = DiscreteCritic(self.observation_space, self.action_space, self.net_arch,
                                         critic_fe, critic_fe.features_dim, self.activation_fn,
                                         self.n_critics, self.normalize_images)
            critic_params = list(self.critic.parameters())
        # --- critic target (extractor 비공유, eval 고정) ---
        target_fe = self.make_features_extractor()
        self.critic_target = DiscreteCritic(self.observation_space, self.action_space, self.net_arch,
                                            target_fe, target_fe.features_dim, self.activation_fn,
                                            self.n_critics, self.normalize_images)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(critic_params, lr=lr_schedule(1),
                                                    **self.optimizer_kwargs)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(net_arch=self.net_arch, activation_fn=self.activation_fn,
                         n_critics=self.n_critics,
                         share_features_extractor=self.share_features_extractor,
                         lr_schedule=self._dummy_schedule,
                         optimizer_class=self.optimizer_class,
                         optimizer_kwargs=self.optimizer_kwargs,
                         features_extractor_class=self.features_extractor_class,
                         features_extractor_kwargs=self.features_extractor_kwargs))
        return data

    def _predict(self, observation, deterministic: bool = True) -> th.Tensor:
        # ⚠️ 마스크 미적용(BasePolicy.predict/SB3 API 최소 호환용). 실사용은 알고리즘의
        # predict_masked() — 마스크드 argmax/샘플.
        logits = self.actor.logits(observation)
        if deterministic:
            return th.argmax(logits, dim=1)
        return th.distributions.Categorical(logits=logits).sample()

    def forward(self, obs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)  # critic_target 은 항상 eval
        self.training = mode


# ---------------------------------------------------------------------------
# 알고리즘
# ---------------------------------------------------------------------------
class SACDiscrete(OffPolicyAlgorithm):
    """마스킹 이산 SAC.

    타깃    y = r + γ(1−d)·Σ_a′ π(a′|s′)[min(Q1t,Q2t)(s′,a′) − α·logπ(a′|s′)]
            (π 는 next_masks 마스킹 → 무효행동 확률 0 → Σ 는 유효행동만).
    critic  MSE(Q1(s).gather(a), y) + MSE(Q2(s).gather(a), y).
    actor   E_s Σ_a π(a|s)[α·logπ(a|s) − min(Q1,Q2)(s,a)] (π 는 masks 마스킹).
    α 자동조정(재난특화): per-state 목표 엔트로피 H̄(s)=target_entropy_coef·log(n_valid(s))
            — 유효행동 수가 상태마다 다른 도메인 반영. n_valid=masks.sum(1).
            alpha_loss = E_s Σ_a π(a|s)·(−log_α·(logπ(a|s)+H̄(s)).detach()).
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {"MlpPolicy": SACDiscretePolicy}
    policy: SACDiscretePolicy
    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        policy: "str | type[SACDiscretePolicy]" = "MlpPolicy",
        env: "GymEnv | str | None" = None,
        learning_rate: "float | Schedule" = 3e-4,
        buffer_size: int = 500_000,
        learning_starts: int = 50_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: "int | tuple[int, str]" = 1,
        gradient_steps: int = 1,
        target_entropy_coef: float = 0.5,
        init_alpha: float = 1.0,
        replay_buffer_class: "type[ReplayBuffer] | None" = None,
        replay_buffer_kwargs: "dict[str, Any] | None" = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: "dict[str, Any] | None" = None,
        stats_window_size: int = 100,
        tensorboard_log: "str | None" = None,
        verbose: int = 0,
        seed: "int | None" = None,
        device: "th.device | str" = "auto",
        _init_setup_model: bool = True,
    ):
        # 마스크 저장 버퍼 강제(계약) — None 이든 명시든 항상 MaskedReplayBuffer.
        if replay_buffer_class is None:
            replay_buffer_class = MaskedReplayBuffer
        super().__init__(
            policy, env, learning_rate,
            buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size,
            tau=tau, gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps,
            action_noise=None, replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs, optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs, stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log, verbose=verbose, device=device, seed=seed,
            sde_support=False, use_sde=False,  # gSDE 미지원(이산) → policy_kwargs 오염 방지
            supported_action_spaces=(spaces.Discrete,), support_multi_env=True,
        )
        self.target_entropy_coef = float(target_entropy_coef)
        self.init_alpha = float(init_alpha)
        self.log_alpha: Optional[nn.Parameter] = None
        self.alpha_optimizer: Optional[th.optim.Optimizer] = None
        self._mask_env = None            # collect_rollouts 가 스태시(라이브 마스크 재계산용)
        self._warned_unmasked_predict = False
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()  # lr스케줄·시드·replay_buffer(MaskedReplayBuffer)·policy 생성
        self._create_aliases()
        # α = exp(log_α), log_α 는 nn.Parameter(알고리즘 소유 — nn.Module 아니므로 state_dict
        # 에 안 잡힘 → _get_torch_save_params 의 pytorch_variables 로 별도 저장, SAC 관례).
        self.log_alpha = nn.Parameter(th.log(th.ones(1, device=self.device) * self.init_alpha))
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=self.lr_schedule(1))

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # ---- 롤아웃: 라이브 마스크 재계산용 env 스태시 후 super ----
    def collect_rollouts(self, env, *args, **kwargs):
        self._mask_env = env
        return super().collect_rollouts(env, *args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs: int = 1):
        """행동 선택(계약: (action, buffer_action) 반환). 현재상태 마스크를 버퍼에 전달."""
        mask_env = self._mask_env if self._mask_env is not None else self.env
        masks = np.stack(mask_env.env_method("action_masks")).astype(np.bool_)  # (n_envs, A)
        # 같은 스텝의 add() 가 소비(_sample_action → env.step → _store_transition 순서 보장)
        self.replay_buffer._pending_cur_masks = masks

        if self.num_timesteps < learning_starts:
            # warmup: 유효행동 균등 샘플(마스크드)
            actions = np.array([int(np.random.choice(np.flatnonzero(m))) for m in masks])
        else:
            # actor 마스킹 Categorical 샘플(탐색). _last_obs 는 VecNormalize 정규화 obs.
            self.policy.set_training_mode(False)
            with th.no_grad():
                obs_t, _ = self.policy.obs_to_tensor(self._last_obs)
                logits = self.actor.logits(obs_t)
                mask_t = th.as_tensor(masks, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e9)
                actions = th.distributions.Categorical(logits=logits).sample().cpu().numpy()
        actions = actions.astype(np.int64)
        return actions, actions  # discrete: action == buffer_action

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.alpha_optimizer])

        actor_losses, critic_losses, alpha_losses, alphas, entropies = [], [], [], [], []

        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            masks = batch.masks             # (B, A) bool — s_t 유효행동
            next_masks = batch.next_masks   # (B, A) bool — s_{t+1} 유효행동
            actions = batch.actions.long()  # (B, 1)
            rewards = batch.rewards          # (B, 1)
            dones = batch.dones              # (B, 1)

            alpha = th.exp(self.log_alpha.detach())  # 크리틱/액터엔 detach(α 학습은 alpha_loss 전용)

            # ---- 타깃 y (soft value bootstrap, next_masks 마스킹) ----
            with th.no_grad():
                next_logits = self.actor.logits(batch.next_observations).masked_fill(~next_masks, -1e9)
                next_logp = F.log_softmax(next_logits, dim=1)   # (B, A) — 무효행동 ≈ -1e9
                next_p = next_logp.exp()                        # 무효행동 확률 ≈ 0
                nq1, nq2 = self.critic_target(batch.next_observations)
                min_nq = th.min(nq1, nq2)                       # (B, A)
                next_v = (next_p * (min_nq - alpha * next_logp)).sum(dim=1, keepdim=True)  # (B,1)
                target_q = rewards + (1.0 - dones) * self.gamma * next_v

            # ---- critic loss (twin, 실제 취한 행동 Q gather) ----
            q1, q2 = self.critic(batch.observations)
            q1a = q1.gather(1, actions)
            q2a = q2.gather(1, actions)
            critic_loss = F.mse_loss(q1a, target_q) + F.mse_loss(q2a, target_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ---- actor loss (masks 마스킹, 전 행동 기대값) ----
            logits = self.actor.logits(batch.observations).masked_fill(~masks, -1e9)
            logp = F.log_softmax(logits, dim=1)                 # (B, A)
            p = logp.exp()
            with th.no_grad():
                pq1, pq2 = self.critic(batch.observations)
                min_pq = th.min(pq1, pq2)                       # (B, A)
            actor_loss = (p * (alpha * logp - min_pq)).sum(dim=1).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ---- α 자동조정: per-state 목표 엔트로피 H̄(s)=coef·log(n_valid) ----
            n_valid = masks.sum(dim=1).clamp(min=1).float()             # (B,)
            target_ent = self.target_entropy_coef * th.log(n_valid)     # (B,)
            pd, logpd = p.detach(), logp.detach()
            # E_s Σ_a π(−log_α·(logπ+H̄))  (π,logπ,H̄ detached — log_α 만 학습)
            #   = −log_α · E_s[ Σ_a π·logπ + H̄ ] = −log_α · E_s[ H̄ − entropy ]
            inner = (pd * (logpd + target_ent.unsqueeze(1))).sum(dim=1)  # (B,) = H̄ − entropy
            alpha_loss = (-self.log_alpha * inner).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ---- critic target polyak ----
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            alpha_losses.append(alpha_loss.item())
            alphas.append(alpha.item())
            entropies.append((-(pd * logpd).sum(dim=1)).mean().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/alpha", float(np.mean(alphas)))
        self.logger.record("train/alpha_loss", float(np.mean(alpha_losses)))
        self.logger.record("train/entropy", float(np.mean(entropies)))

    # ---- 예측 ----
    def predict_masked(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> int:
        """전 v5 드라이버 공통 계약. obs=(obs_dim,) 는 호출자가 이미 VecNormalize 정규화해
        전달(내부 정규화 금지). mask=bool(A,). → 유효행동 flat int."""
        self.policy.set_training_mode(False)
        with th.no_grad():
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(1, -1)
            mask_t = th.as_tensor(np.asarray(mask, dtype=bool), device=self.device).reshape(1, -1)
            logits = self.actor.logits(obs_t).masked_fill(~mask_t, -1e9)
            if deterministic:
                return int(th.argmax(logits, dim=1).item())
            return int(th.distributions.Categorical(logits=logits).sample().item())

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        # ⚠️ SB3 API 호환 최소 구현 — 마스크 미적용 argmax/샘플(무효행동 선택 가능).
        # 평가는 반드시 predict_masked() 사용.
        if not self._warned_unmasked_predict:
            warnings.warn("SACDiscrete.predict() 는 마스크 미적용(무효행동 가능) — "
                          "평가엔 predict_masked(obs, mask) 사용")
            self._warned_unmasked_predict = True
        return self.policy.predict(observation, state, episode_start, deterministic)

    # ---- save/load (SB3 zip 관례, sac.py:322-332 참조) ----
    def _excluded_save_params(self) -> list[str]:
        # actor/critic/critic_target 은 policy 서브모듈 alias(중복 저장 회피),
        # _mask_env 는 VecEnv(피클 불가) → 제외.
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "_mask_env"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "alpha_optimizer"]
        return state_dicts, ["log_alpha"]  # log_alpha 는 pytorch_variable 로 저장/복원
