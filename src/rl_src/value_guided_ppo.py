"""v7 Value-Guided PPO — NCRP 개선가치를 크리틱으로 흡수하는 MaskablePPO 서브클래스 (설계 §4·부록 B).

설계 정본: docs/v7_value_guided_설계_2026-07-22.md.

핵심(방법 A): forward search(NCRP)가 노출한 개선가치 q(s)(pdrwog 단위, 미래 난수 위 기댓값 =
obs 의 함수라 크리틱이 회귀 가능)를 **크리틱 V(s) 의 보조 회귀 타깃**으로 PPO 손실에 상시 혼합한다.
공유 크리틱이 개선정책 가치(≈V^π+dpdr)로 당겨지면 PPO 의 advantage 가 재형성되어(§4.3) 정책이
개선 방향으로 이동 — **행동을 지목(BC)하지 않으므로** v3·v4·v6 의 행동-BC 천장을 우회 시도.
방법 B(옵션): CRR/AWAC 식 dpdr 가중 masked-NLL(a* 모방) 을 보조로 결합(고신뢰 스위치 앵커).

이 파일은 **스켈레톤**(배관 검증까지):
  - ValueGuidedMaskablePPO.train() 오버라이드로 aux value(+ CRR) 손실 합산(단일 옵티마이저 스텝).
  - load_value_labels(): 공유 스키마(ncrp_value_labels_*.pkl) 로드·차원검증·단위환산(할인·/σ_ret).
  - __main__ 스모크: 더미 라벨·더미 env 로 로드→aux 1 스텝→저장/재로드.

단위환산(부록 A): 크리틱은 VecNormalize(norm_reward=True) 하에서 정규화·할인 return 을 예측하므로,
무할인·비정규화 pdrwog 인 q 를 반드시 (1) 할인 (2) /σ_ret 로 환산해야 스케일이 맞는다. 스키마에
할인 suffix(q_*_disc)가 있으면 그대로, 없으면 무할인 폴백(상태별 편향 경고 — planner 1줄 수정 권장).

의존: sb3_contrib.MaskablePPO(train 본문 복제 후 aux 삽입), stable_baselines3.F.mse_loss.
역직렬화: pointer 챔피언 로드 시 호출자가 `from pointer_policy import ...` 선행(train_ppo_feature 관례).
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


# ============================================================ 가중함수(방법 B, CRR/AWAC)
def crr_weight(dpdr: np.ndarray, mode: str, beta: float = 1.0, eps: float = 5e-3,
               clip: float = 20.0) -> "np.ndarray | None":
    """dpdr(개선분, pdrwog)로 CRR/AWAC 가중 w 계산.

    mode: 'off'/None → None(방법 B 미사용) / 'binary' → 1[dpdr>eps] / 'exp' → min(exp(dpdr/β), clip).
    binary 는 노이즈 스위치(dpdr≈0)를 컷, exp 는 CRR 관례로 20 클립(설계 §5.1).
    """
    if mode in (None, "off"):
        return None
    dpdr = np.asarray(dpdr, dtype=np.float32)
    if mode == "binary":
        return (dpdr > eps).astype(np.float32)
    if mode == "exp":
        return np.minimum(np.exp(dpdr / max(beta, 1e-6)), clip).astype(np.float32)
    raise ValueError(f"crr mode 는 off|binary|exp, got {mode!r}")


def _champion_values(model, obs_np, device, batch=4096):
    """챔피언 크리틱 V(s)(정규화 return 공간) 배치 예측 — relative baseline 앵커."""
    import torch as _th
    model.policy.set_training_mode(False)
    outs = []
    with _th.no_grad():
        for i in range(0, len(obs_np), batch):
            ob = _th.as_tensor(obs_np[i:i + batch], device=device)
            outs.append(model.policy.predict_values(ob).flatten().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


# ============================================================ 스키마 로더 + 단위환산
def load_value_labels(path: str, model: MaskablePPO, *, aux_target: str = "q_best",
                      sigma_ret: "float | None" = None, baseline: str = "relative",
                      crr: "str | None" = None, crr_beta: float = 1.0,
                      clip_q: float = 1.5, device: "str | None" = None) -> dict:
    """공유 스키마 pkl 로드 → 모델 차원검증 → 타깃 단위환산 → torch 텐서 dict 반환.

    스키마: obs(N,obs_dim 정규화본), q_greedy/q_best/q_exec(N, pdrwog **h절단** suffix),
            dpdr(N,≥0), q_*_disc/dpdr_disc(할인), actions/masks, obs_dim, n_actions.
    반환: {obs, y_tilde, act, mask, w, meta}.

    ★baseline 모드(핵심 — 2026-07-22 실데이터서 확정, 설계 R4):
      NCRP q 는 h=10 **절단** 리프無 롤아웃이라 **완전 가치의 ~1/3**(무편향 크리틱 V≈3.5σ 대비
      q/σ≈1.0). 절대 회귀(V→q/σ)는 크리틱을 3배 끌어내려 **왜곡**. 절단 리프를 챔피언 V(s)로
      근사(MVE γ^h·V̂ 항의 앵커 — 개선분에서 tail 상쇄되어 유효):
        - "relative"(기본): y = V_champ(s) + dpdr_disc/σ_ret  (개선정책 가치 = 현 가치+측정 개선분).
                            대부분 dpdr=0 → 타깃=V_champ(무변). switched 상태만 V 상향 = §4.3 재형성.
        - "absolute"     : y = q_target_disc/σ_ret  (진단/대조군 — 절단이면 스케일 편향 알려짐).
    sigma_ret: sqrt(ret_rms.var). None → 1.0(norm_reward=False 가정, 경고).
    """
    dev = device or model.device
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "obs" not in d:
        raise KeyError(f"라벨 스키마 누락: obs (키={sorted(d.keys())})")

    obs = np.asarray(d["obs"], dtype=np.float32)
    N = obs.shape[0]
    obs_dim_m = int(model.observation_space.shape[0])
    n_act_m = int(model.action_space.n)
    obs_dim_lbl = int(d.get("obs_dim", obs.shape[1]))
    if obs.shape[1] != obs_dim_m or obs_dim_lbl != obs_dim_m:
        raise ValueError(f"obs_dim 불일치 — 라벨 {obs.shape[1]}/{obs_dim_lbl} vs 모델 {obs_dim_m} "
                         f"(챔피언·라벨수집 obs 변형/패딩 일치 확인)")
    if sigma_ret is None:
        sigma_ret = 1.0
        print("[vg] ⚠️ sigma_ret 미지정 → 1.0 (norm_reward=False 가정). norm_reward=True 챔피언이면 "
              "sqrt(ret_rms.var) 를 반드시 전달할 것(부록 A).", flush=True)
    sigma_ret = float(sigma_ret)

    # ---- 개선분 dpdr(할인 우선) — relative 앵커의 신호 ----
    dpdr_raw = np.asarray(d.get("dpdr_disc", d.get("dpdr", np.zeros(N))), dtype=np.float32)
    imp = np.clip(dpdr_raw, 0.0, clip_q) / sigma_ret          # 정규화 개선분(≥0)

    if baseline == "relative":
        v_champ = _champion_values(model, obs, dev)           # 정규화 return 공간(≈3.5)
        y_tilde = (v_champ + imp).astype(np.float32)
        disc_note = f"relative: V_champ(mean={v_champ.mean():.3f}) + dpdr_disc/σ(mean={imp.mean():.4f})"
    elif baseline == "absolute":
        disc_key = f"{aux_target}_disc"
        q = np.asarray(d.get(disc_key, d.get(aux_target)), dtype=np.float32)
        q = np.clip(q, 0.0, clip_q)
        y_tilde = (q / sigma_ret).astype(np.float32)
        disc_note = f"absolute: {disc_key if disc_key in d else aux_target}/σ (⚠️절단 스케일편향 가능)"
    else:
        raise ValueError(f"baseline 은 relative|absolute, got {baseline!r}")

    # ---- 방법 B용 action/mask (스키마 최종: 복수형 actions/masks; 구 단수 키도 폴백) ----
    act = d.get("actions", d.get("action"))
    mask = d.get("masks", d.get("mask"))
    w = crr_weight(d.get("dpdr", np.zeros(N)), crr, crr_beta) if crr not in (None, "off") else None
    if crr not in (None, "off") and (act is None or mask is None):
        raise ValueError(f"crr={crr} 인데 라벨에 actions/masks 없음 (키={sorted(d.keys())})")

    out = {
        "obs": th.as_tensor(obs, device=dev),
        "y_tilde": th.as_tensor(y_tilde, device=dev),
        "act": th.as_tensor(np.asarray(act, dtype=np.int64), device=dev) if act is not None else None,
        "mask": th.as_tensor(np.asarray(mask, dtype=bool), device=dev) if mask is not None else None,
        "w": th.as_tensor(w, device=dev) if w is not None else None,
        "meta": {"N": N, "baseline": baseline, "aux_target": aux_target, "note": disc_note,
                 "sigma_ret": sigma_ret, "n_actions": int(d.get("n_actions", n_act_m)),
                 "imp_mean": float(imp.mean()), "imp_std": float(imp.std()),
                 "imp_nonzero_frac": float((imp > 1e-6).mean()),
                 "y_tilde_mean": float(y_tilde.mean()), "y_tilde_std": float(y_tilde.std())},
    }
    print(f"[vg] 라벨 로드 N={N} baseline={baseline} [{disc_note}] σ_ret={sigma_ret:.4f} "
          f"ỹ[mean={out['meta']['y_tilde_mean']:.3f} std={out['meta']['y_tilde_std']:.3f}] "
          f"imp[mean={out['meta']['imp_mean']:.4f} nonzero={out['meta']['imp_nonzero_frac']:.3f}] "
          f"crr={crr}", flush=True)
    return out


# ============================================================ 서브클래스
class ValueGuidedMaskablePPO(MaskablePPO):
    """MaskablePPO + NCRP 가치유도 보조손실. train() 만 오버라이드(collect_rollouts 등 상속).

    set_value_guidance() 로 라벨·계수를 주입하면 매 미니배치에서 오프라인 라벨 미니배치를 샘플해
    aux value(+CRR) 손실을 PPO 손실에 합산. 미주입(또는 aux_coef=crr_coef=0)이면 바닐라 PPO(회귀보증).
    """

    def set_value_guidance(self, labels: "dict | None", *, aux_coef: float = 0.5,
                           crr_coef: float = 0.0, aux_batch: "int | None" = None,
                           seed: int = 0) -> None:
        """labels=load_value_labels 결과. aux_coef=value 회귀 계수, crr_coef=CRR BC 계수(방법 B).
        aux_batch=aux 미니배치 크기(None → PPO batch_size). """
        self._vg = labels
        self._vg_aux_coef = float(aux_coef)
        self._vg_crr_coef = float(crr_coef)
        self._vg_bs = int(aux_batch) if aux_batch else int(self.batch_size)
        self._vg_rng = np.random.default_rng(seed)
        self._ev_history = []  # 진단 곡선(train() 이 매 업데이트 append) — 드라이버가 CSV 로 덤프
        if labels is not None:
            print(f"[vg] value guidance on: aux_coef={aux_coef} crr_coef={crr_coef} "
                  f"aux_batch={self._vg_bs} N={labels['meta']['N']}", flush=True)

    def _aux_losses(self):
        """오프라인 라벨 미니배치에서 (aux_value_loss, aux_crr_loss) 계산. 미설정 시 (0,0)."""
        vg = getattr(self, "_vg", None)
        zero = th.zeros((), device=self.device)
        if vg is None or (self._vg_aux_coef <= 0 and self._vg_crr_coef <= 0):
            return zero, zero, None
        N = vg["obs"].shape[0]
        idx = self._vg_rng.integers(0, N, size=min(self._vg_bs, N))
        idx_t = th.as_tensor(idx, device=self.device)
        obs_b = vg["obs"].index_select(0, idx_t)
        # evaluate_actions 로 values(+log_prob) 를 grad 유지하여 획득(train 루프와 동일 경로).
        # act/mask 없으면(A 단독) 더미 action=0·mask=None 로 values 만 사용.
        act_b = vg["act"].index_select(0, idx_t) if vg["act"] is not None \
            else th.zeros(len(idx), dtype=th.long, device=self.device)
        mask_b = vg["mask"].index_select(0, idx_t) if vg["mask"] is not None else None
        values, log_prob, _ = self.policy.evaluate_actions(obs_b, act_b, action_masks=mask_b)
        values = values.flatten()
        # (A) value 회귀
        aux_v = zero
        if self._vg_aux_coef > 0:
            aux_v = F.mse_loss(values, vg["y_tilde"].index_select(0, idx_t))
        # (B) CRR/AWAC 가중 masked-NLL — **선택집합(w>0) 평균**으로 정규화.
        # sparse positive(switched·dpdr>eps ~4.4%)를 batch 평균이 ~22배 희석하는 것을 방지 →
        # crr_coef 를 "flip-BC 강도"(PPO policy loss 대비)로 해석 가능하게. binary w 면
        # = switched 결정들의 평균 NLL, exp w 면 가중평균. 선택 0개면 0(무기여).
        aux_crr = zero
        if self._vg_crr_coef > 0 and vg["w"] is not None:
            w_b = vg["w"].index_select(0, idx_t)
            denom = w_b.sum().clamp(min=1.0)
            aux_crr = -(w_b * log_prob).sum() / denom
        # 진단: 라벨셋 크리틱 EV(신규 롤아웃 아님 — 라벨 분포 기준 근사)
        with th.no_grad():
            ev = explained_variance(values.detach().cpu().numpy(),
                                    vg["y_tilde"].index_select(0, idx_t).detach().cpu().numpy())
        return aux_v, aux_crr, float(ev)

    def train(self) -> None:
        """MaskablePPO.train() 본문 복제 + aux value/CRR 손실 합산(단일 옵티마이저 스텝)."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        aux_v_losses, aux_crr_losses, aux_evs = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ---- v7: NCRP 가치유도 보조손실(A: value 회귀 / B: CRR BC) ----
                aux_v, aux_crr, aux_ev = self._aux_losses()
                if aux_ev is not None:
                    aux_v_losses.append(float(aux_v.item()))
                    aux_crr_losses.append(float(aux_crr.item()))
                    aux_evs.append(aux_ev)

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                        + getattr(self, "_vg_aux_coef", 0.0) * aux_v
                        + getattr(self, "_vg_crr_coef", 0.0) * aux_crr)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                            self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if aux_v_losses:  # v7 진단 로깅
            self.logger.record("train/aux_value_loss", float(np.mean(aux_v_losses)))
            self.logger.record("train/aux_crr_loss", float(np.mean(aux_crr_losses)))
            self.logger.record("train/critic_ev_labels", float(np.mean(aux_evs)))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # v7 진단 곡선(핵심 계기: 신규 롤아웃 EV 가 오르면 "가치는 배운다" = 가설 1차 지지).
        if not hasattr(self, "_ev_history"):
            self._ev_history = []
        self._ev_history.append({
            "t": int(self.num_timesteps),
            "ev": float(explained_var),                                   # on-policy 크리틱 EV(신규 롤아웃)
            "value_loss": float(np.mean(value_losses)),
            "aux_value_loss": float(np.mean(aux_v_losses)) if aux_v_losses else 0.0,
            "critic_ev_labels": float(np.mean(aux_evs)) if aux_evs else float("nan"),
            "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else float("nan"),
        })


# ============================================================ 스모크(더미 배관 검증)
def _smoke():
    """더미 라벨·더미 env 로 배관 검증: 라벨생성→로드→학습(aux 1+ 스텝)→저장→재로드."""
    import tempfile
    import gymnasium as gym
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    OBS_DIM, N_ACT, N_LBL = 16, 8, 200

    class _DummyEnv(gym.Env):
        """랜덤 obs·보상·마스크(전부 유효)의 최소 gym env — SB3 배관 테스트용."""
        def __init__(self, seed=0):
            self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
            self.action_space = spaces.Discrete(N_ACT)
            self._rng = np.random.default_rng(seed)
            self._t = 0

        def action_masks(self):
            return np.ones(N_ACT, dtype=bool)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._t = 0
            return self._rng.standard_normal(OBS_DIM).astype(np.float32), {}

        def step(self, action):
            self._t += 1
            obs = self._rng.standard_normal(OBS_DIM).astype(np.float32)
            r = float(self._rng.standard_normal() * 0.1)
            term = self._t >= 10
            return obs, r, term, False, {}

    def _mk():
        return Monitor(ActionMasker(_DummyEnv(0), lambda e: e.action_masks()))

    tmp = tempfile.mkdtemp(prefix="vg_smoke_")
    # 1) 더미 라벨(공유 스키마 최종) 생성 — 복수형 actions/masks + 할인 suffix(_disc) 포함해
    #    방법 A(q_best_disc 경로)·방법 B 모두 배관 검증.
    rng = np.random.default_rng(0)
    q_greedy = rng.uniform(0.3, 0.6, N_LBL).astype(np.float32)
    dpdr = np.abs(rng.normal(0, 0.02, N_LBL)).astype(np.float32)
    disc = 0.85  # 할인 suffix 는 무할인보다 작다(γ^depth 누적)
    labels_pkl = os.path.join(tmp, "ncrp_value_labels_smoke.pkl")
    with open(labels_pkl, "wb") as f:
        pickle.dump({
            "obs": rng.standard_normal((N_LBL, OBS_DIM)).astype(np.float32),
            "q_greedy": q_greedy, "q_best": (q_greedy + dpdr).astype(np.float32),
            "q_exec": (q_greedy + dpdr * (dpdr > 0.005)).astype(np.float32),
            "q_greedy_disc": (q_greedy * disc).astype(np.float32),
            "q_best_disc": ((q_greedy + dpdr) * disc).astype(np.float32),
            "q_exec_disc": ((q_greedy + dpdr * (dpdr > 0.005)) * disc).astype(np.float32),
            "dpdr": dpdr, "dpdr_disc": (dpdr * disc).astype(np.float32),
            "switched": (dpdr > 0.005), "n_cand": np.full(N_LBL, 8, np.int64),
            "regions": np.zeros(N_LBL, np.int64),
            "actions": rng.integers(0, N_ACT, N_LBL).astype(np.int64),
            "greedy_actions": rng.integers(0, N_ACT, N_LBL).astype(np.int64),
            "masks": np.ones((N_LBL, N_ACT), bool),
            "obs_dim": OBS_DIM, "n_actions": N_ACT,
        }, f)

    # 2) 모델 생성(VecNormalize norm_reward=True → σ_ret 경로 검증)
    venv = VecNormalize(DummyVecEnv([_mk]), norm_obs=True, norm_reward=True, gamma=0.99)
    model = ValueGuidedMaskablePPO("MlpPolicy", venv, n_steps=32, batch_size=16, n_epochs=2,
                                   ent_coef=0.01, verbose=0, seed=0,
                                   policy_kwargs=dict(net_arch=[32, 32]), device="cpu")
    sigma_ret = float(np.sqrt(venv.ret_rms.var + 1e-8))

    # 3) 라벨 로드 + 단위환산 + 방법 A(value)만 / A+B 둘 다 검증
    lab_a = load_value_labels(labels_pkl, model, aux_target="q_best",
                              sigma_ret=sigma_ret, crr=None)
    lab_ab = load_value_labels(labels_pkl, model, aux_target="q_best",
                               sigma_ret=sigma_ret, crr="binary")
    assert lab_a["obs"].shape == (N_LBL, OBS_DIM)
    assert abs(lab_a["meta"]["sigma_ret"] - sigma_ret) < 1e-9
    assert lab_ab["w"] is not None and lab_ab["act"] is not None

    # 4) 학습(collect→train with aux) 1+ 업데이트 — aux value + CRR 둘 다 경로 태움
    model.set_value_guidance(lab_ab, aux_coef=0.5, crr_coef=0.05, seed=0)
    model.learn(total_timesteps=64, progress_bar=False)

    # 5) 저장/재로드
    zip_path = os.path.join(tmp, "vg_smoke.zip")
    model.save(zip_path)
    reloaded = ValueGuidedMaskablePPO.load(zip_path, device="cpu")

    # 6) off 회귀: guidance 미설정 시 aux 항 0 (loss 경로 바닐라와 동일)
    av, ac, ev = model._aux_losses()  # guidance 설정된 model → 유한 손실(grad 유지 → detach 후 출력)
    reloaded.set_value_guidance(None)
    av0, ac0, ev0 = reloaded._aux_losses()
    assert float(av0) == 0.0 and float(ac0) == 0.0 and ev0 is None, "off 경로가 0 이 아님"

    print(f"[vg smoke] OK — labels N={N_LBL} obs_dim={OBS_DIM} n_act={N_ACT} "
          f"σ_ret={sigma_ret:.4f} aux_v={float(av.detach()):.5f} aux_crr={float(ac.detach()):.5f} "
          f"ev_labels={ev:.3f} | off 경로 0 확인 | 저장/재로드 OK ({zip_path})")


if __name__ == "__main__":
    _smoke()
