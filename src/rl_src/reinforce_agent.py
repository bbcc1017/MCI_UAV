"""REINFORCE agent.

Per-episode update, no baseline, no advantage normalization.
Action mask는 환경 제약(없는 자원에 dispatch = no-op)이라 적용.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
