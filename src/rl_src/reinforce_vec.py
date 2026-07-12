"""v5 공정비교 하네스 — 벡터화 REINFORCE(정직한 vanilla policy-gradient + 이동 베이스라인).

목적: 논문 baseline 축. v1 `reinforce_agent.py`(hidden64·베이스라인 없음·단일 env)의
"정직한 REINFORCE" 성격을 유지하되, 챔피언과 공정비교가 되도록 (i) 벡터화 수집,
(ii) 공유 torso(`HospitalTokenExtractor` wide) 또는 mlp256, (iii) 배치 표준화 advantage
(=이동 베이스라인)만 현대화한다. **클리핑·중요도비·에폭 재사용은 없다** — 이 부재가
PPO 와의 차별성(순수 on-policy MC PG)이자 비교연구의 관찰 대상이다.

전 v5 알고리즘 공통 계약(평가 하네스가 의존):
  * `predict_masked(obs(obs_dim,), mask(A,) bool, deterministic=True) -> int`
    - obs 는 **호출자가 정규화해 전달**(평가 env 팩토리의 _NormObs 가 담당). 내부 정규화 금지.
  * `save(path.pt)` / `@classmethod load(path, device="cpu")`.

학습 입력 venv: 챔피언 체인 + MaskInfoWrapper 를 씌운 VecEnv 를 `VecNormalize(norm_obs=True,
norm_reward=False)` 로 감싼 것(train_zoo 가 조립). REINFORCE 는 next-mask/dt(off-policy용)를
쓰지 않고 매 스텝 `venv.env_method("action_masks")` 로 라이브 마스크를 받아 샘플링한다.
보상 정규화를 끄는 근거: pdrwog 는 0~1 유계라 정규화 불요(계획 §3.1).

재사용: pointer_policy.HospitalTokenExtractor(torso, nn.Module 로 직접 사용).
"""
from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

# torso 재사용(순열등변 wide 임베딩). 이 import 는 sb3_contrib 를 끌어오지만 학습·로드 양쪽에 필요.
from pointer_policy import HospitalTokenExtractor


class _PolicyNet(nn.Module):
    """flat obs → 마스킹 전 logits (B, A).

    net="pointer": HospitalTokenExtractor(torso) → Linear(features_dim,256)+ReLU+Linear(256,A).
    net="mlp256" : [256,256] MLP → Linear(256,A) (torso 없이 평탄 obs 직접).
    """

    def __init__(self, obs_dim: int, n_actions: int, net: str = "pointer",
                 H: int = 47, entity_f: int = 7, global_dim: int = 26,
                 embed_dim: int = 64, ctx_dim: int = 128):
        super().__init__()
        self.net_kind = net
        if net == "pointer":
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.extractor = HospitalTokenExtractor(
                obs_space, n_hospitals=H, entity_f=entity_f, global_dim=global_dim,
                embed_dim=embed_dim, ctx_dim=ctx_dim)
            feat = int(self.extractor.features_dim)
            self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(),
                                      nn.Linear(256, n_actions))
        elif net == "mlp256":
            self.extractor = None
            self.head = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, n_actions))
        else:
            raise ValueError(f"net 은 pointer|mlp256, got {net!r}")

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if self.extractor is not None:
            obs = self.extractor(obs)
        return self.head(obs)


class ReinforceVec:
    """벡터화 REINFORCE 에이전트.

    Parameters
    ----------
    obs_dim, n_actions : env obs/action 차원(train_zoo 가 space 에서 역산해 전달).
    net : "pointer"(공유 torso) | "mlp256".
    H, entity_f, global_dim, embed_dim, ctx_dim : pointer torso 형상(net="mlp256"면 무시).
    """

    def __init__(self, obs_dim: int, n_actions: int, net: str = "pointer",
                 H: int = 47, entity_f: int = 7, global_dim: int = 26,
                 embed_dim: int = 64, ctx_dim: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99, ent_coef: float = 0.01,
                 max_grad_norm: float = 10.0, device: str = "cpu"):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.net_kind = net
        # torso 형상(load 때 그대로 복원) — mlp256 이면 미사용이나 보존.
        self._arch = dict(H=int(H), entity_f=int(entity_f), global_dim=int(global_dim),
                          embed_dim=int(embed_dim), ctx_dim=int(ctx_dim))
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = th.device(device)
        self.net = _PolicyNet(self.obs_dim, self.n_actions, net=net, **self._arch).to(self.device)
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=self.lr)

    # ---------- 학습 ----------
    @staticmethod
    def _returns_to_go(rews, gamma: float):
        """MC returns-to-go G_t = Σ_{k≥t} γ^{k-t} r_k."""
        out = [0.0] * len(rews)
        g = 0.0
        for t in reversed(range(len(rews))):
            g = float(rews[t]) + gamma * g
            out[t] = g
        return out

    def _update(self, b_obs, b_act, b_msk, b_ret):
        """배치(≥batch_episodes 완결 에피소드) 1회 vanilla PG 업데이트."""
        obs_t = th.as_tensor(np.asarray(b_obs, dtype=np.float32), device=self.device)
        act_t = th.as_tensor(np.asarray(b_act, dtype=np.int64), device=self.device)
        msk_t = th.as_tensor(np.asarray(b_msk), dtype=th.bool, device=self.device)
        ret_t = th.as_tensor(np.asarray(b_ret, dtype=np.float32), device=self.device)
        # 이동 베이스라인 = 배치 표준화 advantage. std=0(단일 에피소드 등) 가드.
        std = ret_t.std()
        adv = (ret_t - ret_t.mean()) / (std + 1e-8)
        logits = self.net(obs_t).masked_fill(~msk_t, -1e9)
        dist = th.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_t)
        entropy = dist.entropy().mean()
        # 정직한 vanilla PG: 중요도비·클리핑·에폭 재사용 없음(수집시=업데이트시 파라미터 동일).
        loss = -(logp * adv).mean() - self.ent_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item()), float(entropy.item())

    def train(self, venv, total_timesteps: int, batch_episodes: int = 16,
              lr: float = 3e-4, ent_coef: float = 0.01, gamma: float = 0.99,
              max_grad_norm: float = 10.0, log_dir: str | None = None,
              checkpoint_freq: int = 500_000, debug_mask_check: bool = True):
        """venv(VecNormalize)에서 마스킹 샘플링으로 에피소드를 병렬 수집 →
        완결 에피소드 ≥batch_episodes 마다 1회 업데이트. TB 에 {rollout/ep_rew_mean,
        train/entropy, train/loss} 기록.
        """
        from torch.utils.tensorboard import SummaryWriter

        # train() 인자를 인스턴스에 반영(spec 시그니처 준수) + 옵티마이저 lr 재설정.
        self.lr, self.ent_coef, self.gamma, self.max_grad_norm = \
            float(lr), float(ent_coef), float(gamma), float(max_grad_norm)
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=self.lr)

        writer = None
        ckpt_dir = None
        if log_dir:
            writer = SummaryWriter(os.path.join(log_dir, "tb"))
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

        n_envs = venv.num_envs
        obs = venv.reset()  # (n_envs, obs_dim) — VecNormalize 정규화 obs
        # env 별 진행 중 에피소드 버퍼
        ep_obs = [[] for _ in range(n_envs)]
        ep_act = [[] for _ in range(n_envs)]
        ep_msk = [[] for _ in range(n_envs)]
        ep_rew = [[] for _ in range(n_envs)]
        # 업데이트 대기 배치(완결 에피소드 누적)
        b_obs, b_act, b_msk, b_ret = [], [], [], []
        n_done_in_batch = 0
        ep_rew_hist = deque(maxlen=100)  # Monitor info["episode"]["r"]

        num_ts = 0
        last_ckpt = 0
        n_updates = 0
        n_mask_checked = 0
        t0 = time.time()

        while num_ts < total_timesteps:
            masks = np.asarray(venv.env_method("action_masks"), dtype=bool)  # (n_envs, A)
            obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            with th.no_grad():
                logits = self.net(obs_t)
                mask_t = th.as_tensor(masks, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e9)
                actions = th.distributions.Categorical(logits=logits).sample()
            a_np = actions.cpu().numpy().astype(int)
            if debug_mask_check:
                for i in range(n_envs):
                    assert masks[i, a_np[i]], f"마스크 위반: env{i} action={a_np[i]}"
                    n_mask_checked += 1

            next_obs, rewards, dones, infos = venv.step(a_np)
            num_ts += n_envs
            for i in range(n_envs):
                ep_obs[i].append(np.asarray(obs[i], dtype=np.float32))
                ep_act[i].append(int(a_np[i]))
                ep_msk[i].append(masks[i].copy())
                ep_rew[i].append(float(rewards[i]))
                if dones[i]:
                    rets = self._returns_to_go(ep_rew[i], self.gamma)
                    b_obs.extend(ep_obs[i]); b_act.extend(ep_act[i])
                    b_msk.extend(ep_msk[i]); b_ret.extend(rets)
                    info = infos[i]
                    if isinstance(info, dict) and "episode" in info:
                        ep_rew_hist.append(float(info["episode"]["r"]))
                    ep_obs[i], ep_act[i], ep_msk[i], ep_rew[i] = [], [], [], []
                    n_done_in_batch += 1
            obs = next_obs

            if n_done_in_batch >= batch_episodes:
                loss_val, ent_val = self._update(b_obs, b_act, b_msk, b_ret)
                n_updates += 1
                if writer is not None:
                    if ep_rew_hist:
                        writer.add_scalar("rollout/ep_rew_mean",
                                          float(np.mean(ep_rew_hist)), num_ts)
                    writer.add_scalar("train/loss", loss_val, num_ts)
                    writer.add_scalar("train/entropy", ent_val, num_ts)
                b_obs, b_act, b_msk, b_ret = [], [], [], []
                n_done_in_batch = 0

            if ckpt_dir and (num_ts - last_ckpt) >= checkpoint_freq:
                self.save(os.path.join(ckpt_dir, f"reinforce_{num_ts}_steps.pt"))
                last_ckpt = num_ts

        if writer is not None:
            writer.close()
        return {"num_timesteps": num_ts, "n_updates": n_updates,
                "n_mask_checked": n_mask_checked, "sec": time.time() - t0}

    # ---------- 평가 계약 ----------
    def predict_masked(self, obs, mask, deterministic: bool = True) -> int:
        """전 v5 알고 공통 어댑터가 호출. obs 는 호출자가 정규화 전달(내부 정규화 없음)."""
        obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32),
                             device=self.device).unsqueeze(0)
        mask_t = th.as_tensor(np.asarray(mask, dtype=bool),
                              device=self.device).unsqueeze(0)
        with th.no_grad():
            logits = self.net(obs_t).masked_fill(~mask_t, -1e9)
            if deterministic:
                a = int(th.argmax(logits, dim=1).item())
            else:
                a = int(th.distributions.Categorical(logits=logits).sample().item())
        return a

    # ---------- 저장/로드 ----------
    def save(self, path: str):
        th.save({
            "net_state": self.net.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "net": self.net_kind,
            "arch": self._arch,
            "hypers": {"lr": self.lr, "gamma": self.gamma, "ent_coef": self.ent_coef,
                       "max_grad_norm": self.max_grad_norm},
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        # 자작 체크포인트(신뢰 로컬 파일) — arch/hypers dict 를 담으므로 weights_only=False.
        ckpt = th.load(path, map_location=device, weights_only=False)
        arch = dict(ckpt.get("arch", {}))
        hy = dict(ckpt.get("hypers", {}))
        agent = cls(obs_dim=int(ckpt["obs_dim"]), n_actions=int(ckpt["n_actions"]),
                    net=ckpt.get("net", "pointer"),
                    lr=float(hy.get("lr", 3e-4)), gamma=float(hy.get("gamma", 0.99)),
                    ent_coef=float(hy.get("ent_coef", 0.01)),
                    max_grad_norm=float(hy.get("max_grad_norm", 10.0)),
                    device=device, **arch)
        agent.net.load_state_dict(ckpt["net_state"])
        agent.net.eval()
        return agent
