"""포인터 스코어링 head — dest(병원선택)를 flat 인덱스 분류가 아닌 per-hospital 랭킹으로 (플랜 v2 L3).

진단(연구방향 재점검 §3.3): 증류 축별 난이도 실측 class 1.000 / mode 0.916 / dest 0.444 —
병원선택은 본질이 "후보를 특징으로 점수화해 최선을 고르는" 랭킹 문제인데, 기존 구조는
위치 인덱스 47-way 분류라 병원 i 를 고르는 지식이 병원 j 로 전이되지 않는다(deepsets 도
입력만 순열등변, 출력 head 는 flat Linear).

원리: logit(c,d,m) = f_class(ctx) + s_{d,m}(h_d, ctx) + g_mode(ctx) 는 flat categorical 의
**재매개변수화**일 뿐 — MaskableCategoricalDistribution 은 최종 (B, n_actions) logits 만
받으므로 log_prob/entropy/KL/마스킹(apply_masking)·기존 action_masks()·encode/decode 를
전부 무수정 재사용한다. 마스크가 표현 못 하는 결합 제약(helipad×UAV 등)도 apply_masking 이
사후에 덮어쓰므로 무관. 스코어러는 병원 간 가중치 공유 → H 불변·순열등변(지역 전이).

구성:
  - HospitalTokenExtractor : per-hospital 임베딩(embed→self-attn→LN)을 **flat features 벡터
    안에 레이아웃 규약으로 실어** head 까지 전달(features_dim = H*embed + ctx_dim).
    SB3 의 "features_dim 은 flat" 제약 우회 — share_features_extractor=True 전제.
  - PointerActionNet       : 토큰/ctx 분리 → class·mode 소형 head + per-hospital×mode 스코어
    (dest=0 현장대기는 ctx 의존 스칼라) → (B, n_class*(H+1)*n_mode) logits 합성.
    flatten 순서 = (c,d,m) row-major = HospitalFeatureWrapper._encode 와 일치(등변 테스트로 봉인).
  - PointerMaskablePolicy  : net_arch=dict(pi=[], vf=[256,256]) 강제(latent_pi=features 통과),
    _build 에서 action_net 교체. ⚠️ super()._build 가 옵티마이저를 먼저 만들므로 교체 후
    **옵티마이저 재생성 필수**(신규 파라미터 등록). 최종 logit 층은 gain 0.01(초기 near-uniform).

주의: eval/증류 스크립트에서 모델 zip 로드 시 이 모듈이 import 가능해야 함
(train_ppo_feature 의 deepsets 패턴과 동일 — `from pointer_policy import *` 배선).
"""
import numpy as np
import torch as th
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HospitalTokenExtractor(BaseFeaturesExtractor):
    """flat obs → [per-hospital 토큰 (H×embed) | 전역 ctx] 를 한 벡터로 반환.

    입력 레이아웃(HospitalFeatureWrapper): [entity (H*F) | global (gdim)].
    출력 레이아웃(PointerActionNet 과의 계약): [tokens (H*embed_dim) | ctx (ctx_dim)].
    """

    def __init__(self, observation_space, n_hospitals: int, entity_f: int, global_dim: int,
                 embed_dim: int = 32, n_heads: int = 4, ctx_dim: int = 64):
        super().__init__(observation_space, features_dim=n_hospitals * embed_dim + ctx_dim)
        assert observation_space.shape[0] == n_hospitals * entity_f + global_dim, \
            f"obs dim {observation_space.shape[0]} != H*F+g ({n_hospitals}*{entity_f}+{global_dim})"
        self.H, self.F, self.gdim = n_hospitals, entity_f, global_dim
        self.embed_dim, self.ctx_dim = embed_dim, ctx_dim
        self.embed = nn.Sequential(nn.Linear(entity_f, embed_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ctx = nn.Sequential(nn.Linear(embed_dim + global_dim, ctx_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        B = observations.shape[0]
        ent = observations[:, : self.H * self.F].reshape(B, self.H, self.F)
        glob = observations[:, self.H * self.F:]
        t = self.embed(ent)                                   # (B, H, e)
        a, _ = self.attn(t, t, t, need_weights=False)
        t = self.ln(t + a)                                    # (B, H, e) 순열등변
        ctx = self.ctx(th.cat([t.mean(dim=1), glob], dim=1))  # (B, ctx) 순열불변
        return th.cat([t.reshape(B, -1), ctx], dim=1)


class PointerActionNet(nn.Module):
    """latent(=extractor 출력) → flat logits (B, n_class*(H+1)*n_mode).

    L[b,c,d,m] = f_class[b,c] + S[b,d,m] + g_mode[b,m]
      S[b,0,:]  = s0(ctx) broadcast (dest=0 현장대기)
      S[b,1+i,:]= scorer([h_i; ctx]) — 병원 간 가중치 공유(순열등변), mode 별 스코어라
                  dest×mode 상호작용(helipad·원거리 UAV)을 자연 표현.
    """

    def __init__(self, H: int, embed_dim: int, ctx_dim: int,
                 n_class: int = 2, n_mode: int = 2, hidden: int = 64):
        super().__init__()
        self.H, self.e, self.c = H, embed_dim, ctx_dim
        self.n_class, self.n_mode = n_class, n_mode
        self.f_class = nn.Linear(ctx_dim, n_class)
        self.g_mode = nn.Linear(ctx_dim, n_mode)
        self.s0 = nn.Linear(ctx_dim, 1)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_mode),
        )

    def forward(self, latent: th.Tensor) -> th.Tensor:
        B = latent.shape[0]
        tokens = latent[:, : self.H * self.e].reshape(B, self.H, self.e)
        ctx = latent[:, self.H * self.e:]
        fc = self.f_class(ctx)                                        # (B, C)
        gm = self.g_mode(ctx)                                         # (B, M)
        s0 = self.s0(ctx)                                             # (B, 1)
        ctx_e = ctx.unsqueeze(1).expand(-1, self.H, -1)               # (B, H, ctx)
        s = self.scorer(th.cat([tokens, ctx_e], dim=2))               # (B, H, M)
        S = th.cat([s0.unsqueeze(2).expand(-1, 1, self.n_mode), s], dim=1)  # (B, H+1, M)
        L = fc[:, :, None, None] + S[:, None, :, :] + gm[:, None, None, :]  # (B, C, H+1, M)
        # row-major flatten: idx = c*(H+1)*M + d*M + m == HospitalFeatureWrapper._encode
        return L.reshape(B, -1)


class PointerMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy 의 action_net 만 PointerActionNet 으로 교체한 정책."""

    def __init__(self, *args, **kwargs):
        # latent_pi = features 그대로 통과(pi=[]) — head 가 토큰 레이아웃을 온전히 받도록.
        # value 브랜치는 flat features 를 [256,256] MLP 로 처리.
        kwargs["net_arch"] = dict(pi=[], vf=[256, 256])
        super().__init__(*args, **kwargs)
        assert self.share_features_extractor, \
            "PointerMaskablePolicy 는 share_features_extractor=True 전제(토큰 레이아웃 계약)"

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)  # mlp_extractor/value_net/기본 action_net/옵티마이저 생성
        fx = self.features_extractor
        assert isinstance(fx, HospitalTokenExtractor), \
            "features_extractor_class=HospitalTokenExtractor 필요"
        n_class, n_mode = 2, 2
        assert self.action_space.n == n_class * (fx.H + 1) * n_mode, \
            (f"action {self.action_space.n} != {n_class}x{fx.H + 1}x{n_mode} — "
             f"mode 자동축소(uav=0) 구성은 pointer 미지원")
        self.action_net = PointerActionNet(fx.H, fx.embed_dim, fx.ctx_dim,
                                           n_class=n_class, n_mode=n_mode)
        # 초기화: trunk gain √2, 최종 logit 층 gain 0.01(초기 정책 near-uniform — PPO 위생)
        for m in self.action_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        for lin in (self.action_net.f_class, self.action_net.g_mode,
                    self.action_net.s0, self.action_net.scorer[-1]):
            nn.init.orthogonal_(lin.weight, gain=0.01)
            nn.init.constant_(lin.bias, 0.0)
        # ⚠️ super()._build 가 구 action_net 기준으로 옵티마이저를 이미 생성 → 재생성 필수
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)
