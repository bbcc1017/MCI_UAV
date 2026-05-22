"""REINFORCE agent — baseline 포함 개선판 (피드백 3 / 5).

기존: 순수 REINFORCE (baseline 없음, advantage 정규화 없음) → 분산이 커서
일부 지역에서 정책 붕괴(예: 경북 R=3.0).

개선:
  - Critic(가치함수) baseline → advantage = return - V(s)
  - advantage 정규화 (평균 0, 표준편차 1)
  - 엔트로피 보너스 → 조기 수렴/붕괴 방지
  - gradient clipping
  - actor-critic 공유 trunk, 은닉폭 128 × 2층 (기존 64 × 1층)

인터페이스(act/store_reward/train_step/save/load)는 train_reinforce.py 와
호환 유지. 단 net 구조가 바뀌어 구버전 .pt 는 로드 불가(arch 마커로 구분).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "actor_critic_v2"


class ActorCriticNet(nn.Module):
    """공유 trunk + 정책 head + 가치 head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class ReinforceAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 1.0, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.net = ActorCriticNet(obs_dim, n_actions).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._reset_episode_buffer()

    def _reset_episode_buffer(self):
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.rewards = []

    def _forward(self, obs, mask=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        logits = logits.squeeze(0)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_t, float("-1e9"))
        return logits, value.squeeze(0)

    def act(self, obs, mask=None, deterministic: bool = False, store: bool = True):
        logits, value = self._forward(obs, mask=mask)
        if deterministic:
            return int(torch.argmax(logits).item()), None
        dist = torch.distributions.Categorical(logits=logits)
        a_t = dist.sample()
        log_prob = dist.log_prob(a_t)
        if store:
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.entropies.append(dist.entropy())
        return int(a_t.item()), log_prob

    def store_reward(self, reward: float):
        self.rewards.append(float(reward))

    def train_step(self):
        T = len(self.rewards)
        if T == 0 or len(self.log_probs) == 0:
            self._reset_episode_buffer()
            return 0.0

        # 할인 누적 return
        rets = torch.zeros(T, dtype=torch.float32, device=self.device)
        future = 0.0
        for t in reversed(range(T)):
            future = self.rewards[t] + self.gamma * future
            rets[t] = future

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)

        # advantage = return - V(s),  정규화로 분산 축소
        adv = rets - values.detach()
        if T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(log_probs * adv).mean()
        value_loss = F.mse_loss(values, rets)
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optim.step()

        loss_val = float(loss.item())
        self._reset_episode_buffer()
        return loss_val

    def save(self, path: str):
        torch.save({
            "arch": ARCH,
            "net_state": self.net.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if ckpt.get("arch") != ARCH:
            raise ValueError(
                f"{path}: arch={ckpt.get('arch')} — 이 버전(ReinforceAgent {ARCH})과 "
                f"불일치. 해당 모델은 구버전 코드로 재학습/평가하세요.")
        agent = cls(obs_dim=int(ckpt["obs_dim"]),
                    n_actions=int(ckpt["n_actions"]),
                    gamma=float(ckpt.get("gamma", 0.99)),
                    device=device)
        agent.net.load_state_dict(ckpt["net_state"])
        return agent
