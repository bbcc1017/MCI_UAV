"""REINFORCE agent — v1 (피드백 3 이전 원본, 분리 실험용).

피드백 3 에서 reinforce_agent.py 를 actor-critic baseline + 엔트로피 보너스 판
(v2, arch="actor_critic_v2")으로 바꿨으나 plan1nat_f3 평가에서 오히려 악화
(-3.93 → -9.41)했다. 이 악화가 (a) 알고리즘 변경 때문인지 (b) obs 축소 때문인지
가르려고, obs 축소는 유지하되 알고리즘만 이 v1(순수 REINFORCE)로 재학습해 비교한다.

이 파일은 git HEAD 의 reinforce_agent.py 원본을 그대로 보존한 것이다:
  순수 REINFORCE — baseline 없음, advantage 정규화 없음, PolicyNet 64×1층.

train_reinforce.py --agent_version v1 로 선택. 결정이 끝나면(원복/유지) 이 파일과
플래그는 정리한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "reinforce_v1"


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.model(x)


class ReinforceAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.net = PolicyNet(obs_dim, n_actions).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._reset_episode_buffer()

    def _reset_episode_buffer(self):
        self.log_probs = []
        self.rewards = []

    def _logits(self, obs, mask=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.net(obs_t).squeeze(0)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_t, float("-1e9"))
        return logits

    def act(self, obs, mask=None, deterministic: bool = False, store: bool = True):
        logits = self._logits(obs, mask=mask)
        if deterministic:
            action = int(torch.argmax(logits).item())
            return action, None
        dist = torch.distributions.Categorical(logits=logits)
        a_t = dist.sample()
        log_prob = dist.log_prob(a_t)
        if store:
            self.log_probs.append(log_prob)
        return int(a_t.item()), log_prob

    def store_reward(self, reward: float):
        self.rewards.append(float(reward))

    def train_step(self):
        T = len(self.rewards)
        if T == 0 or len(self.log_probs) == 0:
            self._reset_episode_buffer()
            return 0.0

        rets = torch.zeros(T, dtype=torch.float32, device=self.device)
        future_ret = 0.0
        for t in reversed(range(T)):
            future_ret = self.rewards[t] + self.gamma * future_ret
            rets[t] = future_ret

        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * rets).sum()

        self.optim.zero_grad()
        loss.backward()
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
        agent = cls(obs_dim=int(ckpt["obs_dim"]),
                    n_actions=int(ckpt["n_actions"]),
                    gamma=float(ckpt.get("gamma", 0.99)),
                    device=device)
        agent.net.load_state_dict(ckpt["net_state"])
        return agent
