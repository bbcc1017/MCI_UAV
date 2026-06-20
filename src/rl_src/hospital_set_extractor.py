"""Phase 3c — 병원 엔티티 집합 인코더 (SB3 BaseFeaturesExtractor).

HospitalFeatureWrapper 의 평탄 obs = [entity(H,F) 평탄화 | global(gdim)] 을 받아,
병원당 **가중치공유 임베딩 + 자기어텐션**으로 관계추론한 뒤 per-hospital 임베딩을
flatten + global 과 concat 해 정책망에 넘긴다.

설계 주의 (Option A 호환):
  * 행동 dest 는 병원 **위치 인덱스**(Discrete(H+1)). 따라서 순수 pooling(순열불변)은
    어느 병원을 고를지 정보를 없애 행동과 충돌한다. 여기서는 **순열 등변(equivariant)** —
    병원을 재배열하면 per-hospital 출력도 같이 재배열되며, flatten 으로 위치 인덱스를 보존.
  * 일반화 이득은 **가중치공유 φ**(모든 병원을 같은 변환으로 처리 → "좋은 병원의 특징"을
    학습) + **자기어텐션**(각 병원이 다른 병원을 참조한 상대적 판단)에서 온다.

train_ppo_feature.py --extractor deepsets 에서 features_extractor_class 로 주입.
"""
from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HospitalSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, n_hospitals: int, entity_f: int,
                 global_dim: int, embed_dim: int = 32, n_heads: int = 4,
                 features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.H = int(n_hospitals)
        self.F = int(entity_f)
        self.gdim = int(global_dim)
        assert self.H * self.F + self.gdim == int(observation_space.shape[0]), (
            f"obs dim 불일치: H*F+g={self.H*self.F+self.gdim} != {observation_space.shape[0]}")

        self.embed = nn.Sequential(nn.Linear(self.F, embed_dim), nn.ReLU())
        # 자기어텐션 (배치 우선, 1층) — 순열 등변
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.H * embed_dim + self.gdim, features_dim), nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.shape[0]
        ent = obs[:, : self.H * self.F].reshape(B, self.H, self.F)  # (B,H,F)
        g = obs[:, self.H * self.F:]                                # (B,gdim)
        e = self.embed(ent)                                         # (B,H,embed)
        a, _ = self.attn(e, e, e)                                   # 자기어텐션(등변)
        e = self.norm(e + a)                                        # residual + norm
        flat = e.reshape(B, -1)                                     # (B, H*embed) 위치 보존
        return self.head(th.cat([flat, g], dim=1))
