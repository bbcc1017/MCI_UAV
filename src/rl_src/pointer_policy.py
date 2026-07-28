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
    (v6 A3) valid_col!=None 이면 그 엔티티 열(0/1)로 패딩 병원을 식별 — attention
    key_padding_mask 로 패딩 키를 무시하고 ctx 는 유효 병원만 마스크드 평균(패딩 불변·
    순열등변 보존). None(기본)=구 동작·구 zip state_dict 완전 호환(파라미터 shape 불변).
  - PointerActionNet       : 토큰/ctx 분리 → class·mode 소형 head + per-hospital×mode 스코어
    (dest=0 현장대기는 ctx 의존 스칼라) → (B, n_class*(H+1)*n_mode) logits 합성.
    flatten 순서 = (c,d,m) row-major = HospitalFeatureWrapper._encode 와 일치(등변 테스트로 봉인).
  - JointPointerActionNet  : 위 기준선의 extractor·value branch 는 그대로 두고, 병원별 스코어를
    mode 에서 class×mode 로만 확장한다. 따라서 class×destination×mode 3원 상호작용을 직접
    표현하면서 H 불변·순열등변·단일 categorical·하드 마스킹 계약은 그대로 유지한다.
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
                 embed_dim: int = 32, n_heads: int = 4, ctx_dim: int = 64,
                 n_attn_blocks: int = 1, valid_col: int = None):
        super().__init__(observation_space, features_dim=n_hospitals * embed_dim + ctx_dim)
        assert observation_space.shape[0] == n_hospitals * entity_f + global_dim, \
            f"obs dim {observation_space.shape[0]} != H*F+g ({n_hospitals}*{entity_f}+{global_dim})"
        self.H, self.F, self.gdim = n_hospitals, entity_f, global_dim
        self.embed_dim, self.ctx_dim = embed_dim, ctx_dim
        # (v6 A3) valid_col: 패딩 식별 열 인덱스(0/1). None=구 동작(패딩 인지 없음). 신규
        # 파라미터 없음 → 구 zip state_dict 로드 시 이 kwarg 부재로 None 복원(완전 호환).
        self.valid_col = None if valid_col is None else int(valid_col)
        assert self.valid_col is None or 0 <= self.valid_col < entity_f, \
            f"valid_col {self.valid_col} 범위 밖 (entity_f={entity_f})"
        self.embed = nn.Sequential(nn.Linear(entity_f, embed_dim), nn.ReLU())
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        # (v4) 추가 attention 블록: [self-attn+LN → FFN(2e)+LN] × (n_attn_blocks−1).
        # 기본 1 = 구 아키텍처와 state_dict 완전 동일(구 zip 로드 호환) — 1블록째는 위
        # 고정 경로(attn/ln)가 담당. 병원 간 혼잡 결합(포인터의 병원별 독립 스코어가
        # ctx 로만 보던 상호작용)을 직접 표현하는 증축.
        self.blocks = None
        if int(n_attn_blocks) > 1:
            def _blk():
                return nn.ModuleDict({
                    "attn": nn.MultiheadAttention(embed_dim, n_heads, batch_first=True),
                    "ln1": nn.LayerNorm(embed_dim),
                    "ffn": nn.Sequential(nn.Linear(embed_dim, 2 * embed_dim), nn.ReLU(),
                                         nn.Linear(2 * embed_dim, embed_dim)),
                    "ln2": nn.LayerNorm(embed_dim),
                })
            self.blocks = nn.ModuleList([_blk() for _ in range(int(n_attn_blocks) - 1)])
        self.ctx = nn.Sequential(nn.Linear(embed_dim + global_dim, ctx_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        B = observations.shape[0]
        ent = observations[:, : self.H * self.F].reshape(B, self.H, self.F)
        glob = observations[:, self.H * self.F:]
        # (v6 A3) 패딩 인지: valid_col 이 있으면 그 열(0/1)로 패딩 병원을 식별.
        #   kpm(key_padding_mask): True=무시할 키(패딩) → 유효 병원 쿼리는 패딩 키를 안 봄
        #   → 유효 토큰·ctx 가 패딩 특징 교란에 불변. None 이면 kpm=None(구 동작 동일).
        if self.valid_col is not None:
            valid = ent[:, :, self.valid_col] > 0.5           # (B, H) bool
            kpm = ~valid                                      # (B, H) True=패딩
        else:
            valid, kpm = None, None
        t = self.embed(ent)                                   # (B, H, e) — valid 열 포함(무해)
        a, _ = self.attn(t, t, t, need_weights=False, key_padding_mask=kpm)
        t = self.ln(t + a)                                    # (B, H, e) 순열등변
        if self.blocks is not None:
            for blk in self.blocks:
                a2, _ = blk["attn"](t, t, t, need_weights=False, key_padding_mask=kpm)
                t = blk["ln1"](t + a2)
                t = blk["ln2"](t + blk["ffn"](t))             # 순열등변 유지
        if valid is not None:                                 # 마스크드 평균(유효 병원만)
            v = valid.float().unsqueeze(2)                    # (B, H, 1)
            pooled = (t * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)  # 실병원≥34→분모>0
        else:
            pooled = t.mean(dim=1)                            # 구 동작(전체 평균)
        ctx = self.ctx(th.cat([pooled, glob], dim=1))         # (B, ctx) 순열불변
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


class JointPointerActionNet(nn.Module):
    """중증도×목적지×수단 3원 상호작용을 복원한 포인터 head.

    기준선 PointerActionNet 은 병원 스코어 ``S[d,m]`` 를 두 class 가 공유하므로
    ``L[Red,d,m] - L[Yellow,d,m]`` 가 모든 d,m 에서 상수다. 즉 하드 마스크로 제거되지 않은
    후보 사이에서는 Red와 Yellow가 같은 병원·수단 순위를 갖는 구조적 제약이 있다.

    이 head 는 공유 병원 토큰마다 ``S[d,c,m]`` 를 출력한다::

        L[b,c,d,m] = f_class[b,c] + S[b,d,c,m] + g_mode[b,m]

    병원 scorer 와 stay head 의 마지막 출력 차원만 C×M 으로 바꾸므로, extractor·critic·PPO
    하이퍼·관측·행동 마스크·flat action codec 은 기준선과 동일하다. 병원 축 가중치 공유도
    유지되어 가변 H 패딩과 병원 순열에 계속 등변이다.
    """

    def __init__(self, H: int, embed_dim: int, ctx_dim: int,
                 n_class: int = 2, n_mode: int = 2, hidden: int = 64):
        super().__init__()
        self.H, self.e, self.c = H, embed_dim, ctx_dim
        self.n_class, self.n_mode = n_class, n_mode
        self.f_class = nn.Linear(ctx_dim, n_class)
        self.g_mode = nn.Linear(ctx_dim, n_mode)
        self.s0 = nn.Linear(ctx_dim, n_class * n_mode)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_class * n_mode),
        )

    def forward(self, latent: th.Tensor) -> th.Tensor:
        B = latent.shape[0]
        tokens = latent[:, : self.H * self.e].reshape(B, self.H, self.e)
        ctx = latent[:, self.H * self.e:]
        fc = self.f_class(ctx)                                      # (B, C)
        gm = self.g_mode(ctx)                                       # (B, M)
        s0 = self.s0(ctx).reshape(B, 1, self.n_class, self.n_mode)  # (B, 1, C, M)
        ctx_e = ctx.unsqueeze(1).expand(-1, self.H, -1)
        s = self.scorer(th.cat([tokens, ctx_e], dim=2))
        s = s.reshape(B, self.H, self.n_class, self.n_mode)         # (B, H, C, M)
        S = th.cat([s0, s], dim=1).permute(0, 2, 1, 3)             # (B, C, H+1, M)
        L = fc[:, :, None, None] + S + gm[:, None, None, :]
        # row-major flatten: idx = c*(H+1)*M + d*M + m
        return L.reshape(B, -1)


class ClassModeResidualPointerActionNet(PointerActionNet):
    """기준 Pointer 위에 상태의존 class×mode 잔차만 더하는 최소 확장.

    ``S[d,m]`` 병원 랭킹은 그대로 공유하고 ``R[c,m|ctx]`` 만 추가한다. 따라서 의료 적합성
    마스크가 이미 담당하는 class×destination 차이를 중복 학습하지 않으면서, Red가 UAV 시간
    절감에 더 민감한 식의 중증도별 수단 선호를 표현한다. ``r_cm`` 0 초기화 시 기준 Pointer와
    수치적으로 완전히 동일해 warm-start 정책을 훼손하지 않는다.
    """

    def __init__(self, H: int, embed_dim: int, ctx_dim: int,
                 n_class: int = 2, n_mode: int = 2, hidden: int = 64):
        super().__init__(H, embed_dim, ctx_dim, n_class, n_mode, hidden)
        self.r_cm = nn.Linear(ctx_dim, n_class * n_mode)
        nn.init.constant_(self.r_cm.weight, 0.0)
        nn.init.constant_(self.r_cm.bias, 0.0)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        B = latent.shape[0]
        base = super().forward(latent).reshape(B, self.n_class, self.H + 1, self.n_mode)
        ctx = latent[:, self.H * self.e:]
        residual = self.r_cm(ctx).reshape(B, self.n_class, 1, self.n_mode)
        return (base + residual).reshape(B, -1)


class LowRankResidualPointerActionNet(PointerActionNet):
    """기준 Pointer + rank-R class×destination×mode 잔차.

    완전 자유 C×D×M 출력 대신
    ``Δ[c,d,m]=Σ_r tanh(U[c,r|ctx])·V[d,r,m|h_d,ctx]`` 로 제한한다. 병원별 V scorer는
    가중치를 공유하므로 H 불변·순열등변이다. V 최종층과 stay 잔차를 0 초기화해 시작 정책은
    기준 Pointer와 정확히 같고, rank=1/2가 표현력-정규화 사다리를 이룬다.
    """

    def __init__(self, H: int, embed_dim: int, ctx_dim: int,
                 n_class: int = 2, n_mode: int = 2, hidden: int = 64,
                 rank: int = 1):
        super().__init__(H, embed_dim, ctx_dim, n_class, n_mode, hidden)
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError(f"residual rank는 1 이상이어야 함: {rank}")
        self.r_u = nn.Linear(ctx_dim, n_class * self.rank)
        # token/ctx 자체가 이미 비선형 표현이므로 V는 단일 저랭크 투영만 둔다. 별도 hidden
        # MLP를 복제하면 rank-1이어도 기준 scorer만큼 파라미터가 늘어 "저랭크" 규제가 약해진다.
        self.r_v = nn.Linear(embed_dim + ctx_dim, self.rank * n_mode)
        self.r0 = nn.Linear(ctx_dim, n_class * n_mode)
        # 기준 정책 포함: 병원 residual V와 stay residual 모두 정확히 0에서 시작.
        nn.init.constant_(self.r_v.weight, 0.0)
        nn.init.constant_(self.r_v.bias, 0.0)
        nn.init.constant_(self.r0.weight, 0.0)
        nn.init.constant_(self.r0.bias, 0.0)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        B = latent.shape[0]
        base = super().forward(latent).reshape(B, self.n_class, self.H + 1, self.n_mode)
        tokens = latent[:, : self.H * self.e].reshape(B, self.H, self.e)
        ctx = latent[:, self.H * self.e:]
        u = th.tanh(self.r_u(ctx).reshape(B, self.n_class, self.rank))       # (B,C,R)
        ctx_e = ctx.unsqueeze(1).expand(-1, self.H, -1)
        v = self.r_v(th.cat([tokens, ctx_e], dim=2))
        v = v.reshape(B, self.H, self.rank, self.n_mode)                    # (B,H,R,M)
        hospital = th.einsum("bcr,bhrm->bchm", u, v)                        # (B,C,H,M)
        stay = self.r0(ctx).reshape(B, self.n_class, 1, self.n_mode)
        residual = th.cat([stay, hospital], dim=2)                          # (B,C,H+1,M)
        return (base + residual).reshape(B, -1)


class PointerMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy 의 action_net 만 PointerActionNet 으로 교체한 정책."""

    def __init__(self, *args, head_hidden: int = 64, **kwargs):
        # head_hidden: PointerActionNet scorer 은닉폭(기본 64 = 구 L3 아키텍처). _build 전에
        # 저장(super().__init__ 이 _build 를 호출하므로). 구 저장 zip 은 이 인자 없이 로드되어
        # 기본 64 로 복원 → 기존 모델 호환 유지.
        self._head_hidden = int(head_hidden)
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
                                           n_class=n_class, n_mode=n_mode,
                                           hidden=getattr(self, "_head_hidden", 64))
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


class JointPointerMaskablePolicy(PointerMaskablePolicy):
    """기준선과 동일한 torso/critic 위에 JointPointerActionNet 만 사용하는 실험 정책."""

    def _build(self, lr_schedule) -> None:
        # 부모의 extractor·critic 검증과 optimizer 계약을 그대로 거친 뒤 action head 만 교체한다.
        super()._build(lr_schedule)
        fx = self.features_extractor
        self.action_net = JointPointerActionNet(
            fx.H, fx.embed_dim, fx.ctx_dim, n_class=2, n_mode=2,
            hidden=getattr(self, "_head_hidden", 64))
        for m in self.action_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        for lin in (self.action_net.f_class, self.action_net.g_mode,
                    self.action_net.s0, self.action_net.scorer[-1]):
            nn.init.orthogonal_(lin.weight, gain=0.01)
            nn.init.constant_(lin.bias, 0.0)
        # 부모 optimizer 는 기존 PointerActionNet 파라미터를 가리키므로 신규 head 기준 재생성.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


class ResidualPointerMaskablePolicy(PointerMaskablePolicy):
    """기준 Pointer를 포함하는 class×mode 또는 저랭크 3원 잔차 실험 정책."""

    def __init__(self, *args, residual_kind: str = "cm", residual_rank: int = 1, **kwargs):
        self._residual_kind = str(residual_kind)
        self._residual_rank = int(residual_rank)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        fx = self.features_extractor
        common = dict(H=fx.H, embed_dim=fx.embed_dim, ctx_dim=fx.ctx_dim,
                      n_class=2, n_mode=2, hidden=getattr(self, "_head_hidden", 64))
        if self._residual_kind == "cm":
            self.action_net = ClassModeResidualPointerActionNet(**common)
        elif self._residual_kind == "lowrank":
            self.action_net = LowRankResidualPointerActionNet(
                **common, rank=self._residual_rank)
        else:
            raise ValueError(f"residual_kind은 cm|lowrank: {self._residual_kind!r}")

        # 기준 head는 기존 Pointer와 같은 초기화, residual 경로만 마지막에 다시 0으로 봉인.
        for m in self.action_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        for lin in (self.action_net.f_class, self.action_net.g_mode,
                    self.action_net.s0, self.action_net.scorer[-1]):
            nn.init.orthogonal_(lin.weight, gain=0.01)
            nn.init.constant_(lin.bias, 0.0)
        if self._residual_kind == "cm":
            nn.init.constant_(self.action_net.r_cm.weight, 0.0)
            nn.init.constant_(self.action_net.r_cm.bias, 0.0)
        else:
            nn.init.constant_(self.action_net.r_v.weight, 0.0)
            nn.init.constant_(self.action_net.r_v.bias, 0.0)
            nn.init.constant_(self.action_net.r0.weight, 0.0)
            nn.init.constant_(self.action_net.r0.bias, 0.0)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)
