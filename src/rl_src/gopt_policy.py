# -*- coding: utf-8 -*-
"""GOPT식 크로스어텐션 정책 (v12) — 수요(등급×수단) 토큰 × 목적지 토큰 bilinear 채점.

동기(진단): 기준선 Pointer head 는
    L[c,d,m] = f_class(ctx)[c] + S(token_d, ctx)[m] + g_mode(ctx)[m]
로 **S 에 class 축이 없다**. 따라서 같은 상태에서 Red 와 Yellow 의 목적지 선호 순위가
수학적으로 동일하고, 등급 차이는 action mask(Red→Tier3 한정)로만 들어간다. 그런데 생존곡선
(Red 0.56/((t/91)^1.58+1) vs Yellow 0.81/((t/160)^2.41+1)) 은 Red 가 훨씬 급해서 Red 는 ETA 를,
Yellow 는 부하를 더 무겁게 봐야 한다 — 현 구조로는 표현 불가.

v8 은 scorer 를 class 별로 쪼개(S[c,d,m]) 이를 시도했다가 **표본공유를 잃고 실패**(+0.0026 악화).
GOPT(IEEE RA-L 2024, github.com/Xiong5Heng/GOPT)는 같은 문제를 쪼개지 않고 푼다:
"무엇을 처리할지"를 query 토큰, 후보를 key 로 두고 bilinear 점수를 낸다
(`logits = bmm(layer_1(item), layer_2(ems).permute(0,2,1))`). 후보 쪽 파라미터가 전 query 에
공유되므로 표본 단절이 없다. 이 모듈이 그 구조를 MCI 문제로 이식한다.

대응관계:
  GOPT item 토큰(2)        → 수요 토큰 (class × mode) 4개   "어떤 등급을 어떤 수단으로"
  GOPT EMS 후보 K개        → 목적지 토큰 H+1개 (병원 H + 학습 stay 토큰 = dest 0)
  GOPT EMS 유효성 mask     → valid 열에서 유도한 dest 패딩 마스크
  GOPT 1/max(container)    → ETA _norm_by_min (hospital_feature_wrapper, 이미 존재)
  GOPT 빈 크기 일반화      → 병원수 H 일반화 (v6 MCI_H_PAD, 이미 존재)

★obs 무변경: 수요 토큰 입력은 **기존 글로벌 26차원의 재배치**로 만든다(GLOBAL_LAYOUT 참조).
   따라서 obs=402 그대로이고 v10 챔피언과 완전 동일 조건 비교가 성립한다(기존 scoreboard
   cube 의 PPO 에피소드별 값을 paired 기준으로 직접 사용 가능).

설계 결정:
  - `pointer_policy.py` 는 수정하지 않는다. v10/v11 산출물 전부가 그 모듈로 역직렬화되므로
    회귀 위험을 0 으로 유지하고, 여기서 상속만 한다.
  - 병원 토큰 경로(embed/attn/ln)는 부모 `HospitalTokenExtractor` 의 것을 **그대로 상속** →
    X1(head 만 교체)이 v10 인코더와 파라미터 이름·shape 동일.
  - critic 은 SB3 vf=[256,256] 유지(부모 계약). GOPT 의 순열불변 pooled critic 은 별도 팔(X6)로
    격리한다 — 우리 병원 슬롯은 현장 거리순 정렬(Spearman(슬롯,road_dist)=+0.966~+0.995,
    전 지역 일관)이라 합 풀링이 실신호를 파괴할 수 있어 actor 변경과 섞지 않는다.
  - GOPT 는 bmm 에 스케일링이 없고 최종 logit 층도 없다. 이 프로젝트의 PPO 위생 관례
    (초기 정책 near-uniform)를 지키기 위해 **1/√e 스케일 + key 층 gain 0.01** 을 넣는다.
    의도적 편차이며 아래 주석에 표시했다.

주의: eval/증류 스크립트에서 모델 zip 로드 시 이 모듈이 import 가능해야 한다
(pointer_policy 전례 — paired_eval_ladder._worker 의 역직렬화 import 목록에 추가).
"""
import math

import numpy as np
import torch as th
from torch import nn

from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy

# ---------------------------------------------------------------------------
# 글로벌 26차원 레이아웃 (hospital_feature_wrapper._globals 와 1:1 — 단일 진실원)
#   parts = [patient_agg(R/Y 2등급×5단계=10), fleet_agg(AMB 5 + UAV 5=10), time(1)]
#           (+ ctx_static 6, essential+load+ctx 변형만) (+ load extra 5)
#   → 인덱스 0..9 pa(class-major: agg[class,stage].reshape(-1) 앞 10), 10..19 va(AMB 먼저),
#     20.. 이후는 등급·수단 무관 공유 신호(time, ρ, avail_frac, uav_frac, t_norm, [ctx_static])
# fleet_agg 5열 = [가용수, 운행수, 최단복귀, 평균복귀, 중증수송수] (aggregate_obs._fleet_agg)
# ---------------------------------------------------------------------------
PA_LEN, VA_LEN = 10, 10          # 등급별 5열 × 2등급 / 수단별 5열 × 2수단
PER_CLASS, PER_MODE = 5, 5
SHARED_START = PA_LEN + VA_LEN + 0  # 20 — 이 뒤는 전부 공유 신호(time/load/ctx_static)


def demand_input_dim(global_dim: int) -> int:
    """수요 토큰 입력 차원 = 등급 5 + 수단 5 + 공유(global_dim−20)."""
    if global_dim < SHARED_START + 1:
        raise ValueError(f"global_dim {global_dim} < {SHARED_START + 1} — "
                         f"GOPT 수요 토큰은 patient_agg(10)+fleet_agg(10)+공유≥1 레이아웃 전제")
    return PER_CLASS + PER_MODE + (global_dim - SHARED_START)


def build_demand_input(glob: th.Tensor, n_class: int = 2, n_mode: int = 2) -> th.Tensor:
    """글로벌 벡터 → (B, n_class*n_mode, demand_input_dim) 수요 토큰 입력.

    토큰 순서 q = c*n_mode + m (head 의 reshape 계약과 일치).
    정보 추가 없음 — 평탄 글로벌을 등급/수단 축으로 **재배치**할 뿐이다(obs 무변경 근거).
    """
    B = glob.shape[0]
    pa = glob[:, :PA_LEN].reshape(B, n_class, PER_CLASS)               # (B,C,5)
    va = glob[:, PA_LEN:PA_LEN + VA_LEN].reshape(B, n_mode, PER_MODE)  # (B,M,5)
    sh = glob[:, SHARED_START:]                                        # (B,S)
    pa_e = pa.unsqueeze(2).expand(B, n_class, n_mode, PER_CLASS)
    va_e = va.unsqueeze(1).expand(B, n_class, n_mode, PER_MODE)
    sh_e = sh[:, None, None, :].expand(B, n_class, n_mode, sh.shape[1])
    return th.cat([pa_e, va_e, sh_e], dim=-1).reshape(B, n_class * n_mode, -1)


class TBlock(nn.Module):
    """GOPT `TransformerBlock` 대응: MHA(q,kv,kv) + residual + LN + FFN + LN."""

    def __init__(self, embed_dim: int, n_heads: int, ff_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_expansion * embed_dim), nn.LeakyReLU(),
            nn.Linear(ff_expansion * embed_dim, embed_dim),
        )
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: th.Tensor, kv: th.Tensor, key_padding_mask=None) -> th.Tensor:
        a, _ = self.attn(q, kv, kv, need_weights=False, key_padding_mask=key_padding_mask)
        x = self.ln1(q + self.do(a))
        return self.ln2(x + self.do(self.ffn(x)))


class GoptEncoderBlock(nn.Module):
    """GOPT `EncoderBlock` 대응: self×2 + cross×2 (수요 스트림 ↔ 목적지 스트림).

    ⚠️ 마스크 배치 교정: 패딩 마스크는 **목적지가 key 인** 어텐션에만 준다. GOPT 원문은
    item 이 key 인 cross 에도 mask 를 넘기지만 그쪽 스트림엔 패딩이 없어 무해한 quirk 다.
    우리는 목적지 쪽에 실제 패딩(H_pad>실H)이 있으므로 정확히 배치해야 한다.
    stay 토큰(dest 0)은 항상 유효.
    """

    def __init__(self, embed_dim: int, n_heads: int, ff_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        kw = dict(embed_dim=embed_dim, n_heads=n_heads, ff_expansion=ff_expansion, dropout=dropout)
        self.self_dem = TBlock(**kw)      # 수요 self-attention
        self.self_dst = TBlock(**kw)      # 목적지 self-attention (패딩 마스크)
        self.dst_on_dem = TBlock(**kw)    # 공급←수요 cross (key=수요, 마스크 없음)
        self.dem_on_dst = TBlock(**kw)    # 수요←공급 cross (key=목적지, 마스크)

    def forward(self, dem: th.Tensor, dst: th.Tensor, dst_kpm=None):
        dem_s = self.self_dem(dem, dem)
        dst_s = self.self_dst(dst, dst, dst_kpm)
        dst_o = self.dst_on_dem(dst_s, dem_s)
        dem_o = self.dem_on_dst(dem_s, dst_s, dst_kpm)
        return dem_o, dst_o


class GoptTokenExtractor(HospitalTokenExtractor):
    """flat obs → [목적지 토큰 (H+1)*e | 수요 토큰 (C*M)*e | ctx] 한 벡터.

    병원 토큰 경로는 부모와 **동일 모듈**(embed/attn/ln/[blocks]/ctx)을 사용한다 →
    n_gopt_blocks=0 이면 인코더가 v10 과 파라미터 이름·shape 동일(X1: head 만의 효과 격리).
    features_dim = (H + 1 + C*M) * embed_dim + ctx_dim.
    """

    def __init__(self, observation_space, n_hospitals: int, entity_f: int, global_dim: int,
                 embed_dim: int = 32, n_heads: int = 4, ctx_dim: int = 64,
                 n_attn_blocks: int = 1, valid_col: int = None,
                 n_class: int = 2, n_mode: int = 2,
                 n_gopt_blocks: int = 0, ff_expansion: int = 4, dropout: float = 0.0):
        super().__init__(observation_space, n_hospitals=n_hospitals, entity_f=entity_f,
                         global_dim=global_dim, embed_dim=embed_dim, n_heads=n_heads,
                         ctx_dim=ctx_dim, n_attn_blocks=n_attn_blocks, valid_col=valid_col)
        self.n_class, self.n_mode = int(n_class), int(n_mode)
        self.n_dem = self.n_class * self.n_mode
        self.n_dst = n_hospitals + 1                       # 병원 H + stay(dest 0)
        # 부모가 features_dim = H*e + ctx 로 잡아둔 것을 재설정(BaseFeaturesExtractor 규약)
        self._features_dim = (self.n_dst + self.n_dem) * embed_dim + ctx_dim
        # GOPT item_encoder 대응 (Linear→LeakyReLU→Linear)
        self.demand_encoder = nn.Sequential(
            nn.Linear(demand_input_dim(global_dim), 32), nn.LeakyReLU(),
            nn.Linear(32, embed_dim),
        )
        # dest=0 '현장 대기' 후보 토큰 — 학습 파라미터 + ctx 의존 성분(v10 의 s0(ctx) 상태의존성 보존)
        self.stay_base = nn.Parameter(th.zeros(1, 1, embed_dim))
        self.stay_ctx = nn.Linear(ctx_dim, embed_dim)
        self.gopt_blocks = nn.ModuleList([
            GoptEncoderBlock(embed_dim, n_heads, ff_expansion, dropout)
            for _ in range(int(n_gopt_blocks))
        ]) if int(n_gopt_blocks) > 0 else None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 부모 forward 의 병원 토큰·ctx 계산을 그대로 재현한다(중간 텐서가 필요해 super() 호출
        # 대신 동일 연산을 반복 — 부모 수정 금지 결정에 따른 의도적 중복).
        B = observations.shape[0]
        ent = observations[:, : self.H * self.F].reshape(B, self.H, self.F)
        glob = observations[:, self.H * self.F:]
        if self.valid_col is not None:
            valid = ent[:, :, self.valid_col] > 0.5
            kpm = ~valid
        else:
            valid, kpm = None, None
        t = self.embed(ent)
        if self.attn is not None:
            a, _ = self.attn(t, t, t, need_weights=False, key_padding_mask=kpm)
            t = self.ln(t + a)
        else:
            t = self.ln(t)
        if self.blocks is not None:                        # (v4) 추가 self-attn 블록
            for blk in self.blocks:
                a2, _ = blk["attn"](t, t, t, need_weights=False, key_padding_mask=kpm)
                t = blk["ln1"](t + a2)
                t = blk["ln2"](t + blk["ffn"](t))
        if valid is not None:
            v = valid.float().unsqueeze(2)
            pooled = (t * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)
        else:
            pooled = t.mean(dim=1)
        ctx = self.ctx(th.cat([pooled, glob], dim=1))       # (B, ctx)

        # ---- 목적지 스트림: stay 토큰을 dest 0 으로 앞에 붙인다 ----
        stay = self.stay_base.expand(B, 1, self.embed_dim) + self.stay_ctx(ctx).unsqueeze(1)
        dst = th.cat([stay, t], dim=1)                      # (B, H+1, e) — index 0 = dest 0
        if kpm is not None:
            stay_kpm = th.zeros(B, 1, dtype=kpm.dtype, device=kpm.device)  # stay 는 항상 유효
            dst_kpm = th.cat([stay_kpm, kpm], dim=1)
        else:
            dst_kpm = None

        # ---- 수요 스트림: (class×mode) 토큰 ----
        dem = self.demand_encoder(build_demand_input(glob, self.n_class, self.n_mode))

        if self.gopt_blocks is not None:
            for blk in self.gopt_blocks:
                dem, dst = blk(dem, dst, dst_kpm)

        return th.cat([dst.reshape(B, -1), dem.reshape(B, -1), ctx], dim=1)


class GoptBilinearActionNet(nn.Module):
    """GOPT `ActorHead` 대응 — logits = bmm(q(수요), k(목적지)ᵀ)/√e + b(ctx)[c,m].

    latent 레이아웃 계약(GoptTokenExtractor 출력) = [dest (H+1)*e | demand (C*M)*e | ctx].
    flatten 은 (c, d, m) row-major = `HospitalFeatureWrapper._encode`
    (`idx = c*(H+1)*M + d*M + m`) 와 일치시킨다.

    b(ctx)[c,m] = 목적지에 무관한 (등급,수단) 오프셋. 순수 bilinear 로는 만들 수 없는 rank-1
    항이며 v10 의 f_class+g_mode 역할을 유지한다(표현력 상 v10 의 상위집합).
    """

    def __init__(self, H: int, embed_dim: int, ctx_dim: int, n_class: int = 2, n_mode: int = 2):
        super().__init__()
        self.H, self.e, self.c = H, embed_dim, ctx_dim
        self.n_class, self.n_mode = n_class, n_mode
        self.n_dst, self.n_dem = H + 1, n_class * n_mode
        self.layer_1 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())  # query
        self.layer_2 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())  # key
        self.bias_cm = nn.Linear(ctx_dim, n_class * n_mode)
        # ⚠️ GOPT 원문엔 없는 스케일 — 초기 logit 을 작게 유지(PPO 위생: near-uniform 초기정책)
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        B = latent.shape[0]
        n_dst_e = self.n_dst * self.e
        n_dem_e = self.n_dem * self.e
        dst = latent[:, :n_dst_e].reshape(B, self.n_dst, self.e)
        dem = latent[:, n_dst_e:n_dst_e + n_dem_e].reshape(B, self.n_dem, self.e)
        ctx = latent[:, n_dst_e + n_dem_e:]
        q = self.layer_1(dem)                                            # (B, C*M, e)
        k = self.layer_2(dst)                                            # (B, H+1, e)
        S = th.bmm(q, k.transpose(1, 2)) * self.scale                    # (B, C*M, H+1)
        S = S + self.bias_cm(ctx).unsqueeze(2)                           # (B, C*M, 1) 브로드캐스트
        S = S.reshape(B, self.n_class, self.n_mode, self.n_dst)
        return S.permute(0, 1, 3, 2).reshape(B, -1)                      # (B, C, H+1, M) → flat


class PooledCriticNet(nn.Module):
    """GOPT `CriticHead` 대응 — 순열불변 합 풀링 critic (X6 격리 팔).

    SB3 `MlpExtractor.value_net` 자리에 끼워 features → (B, out_dim) 을 낸다
    (정책의 최종 `value_net = Linear(out_dim, 1)` 은 그대로 유지).

    ⚠️ 한계: features 에 valid 열이 실려 있지 않아 **합 풀링에 마스크를 못 쓴다**. 이 라운드의
    학습·평가 시나리오는 고정 H=47(패딩 없음)이라 masked sum 과 수치가 동일하다. 자연-H
    (가변 병원수) 시나리오에 쓰려면 extractor 가 valid 를 features 에 실어야 한다.
    """

    def __init__(self, n_tok: int, embed_dim: int, ctx_dim: int,
                 n_dem: int = 0, out_dim: int = 256):
        super().__init__()
        self.n_tok, self.e, self.c, self.n_dem = n_tok, embed_dim, ctx_dim, n_dem
        self.l_tok = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())
        self.l_dem = (nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LeakyReLU())
                      if n_dem else None)
        in_dim = embed_dim * (2 if n_dem else 1) + ctx_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim), nn.LeakyReLU(),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        B = features.shape[0]
        n_tok_e = self.n_tok * self.e
        tok = features[:, :n_tok_e].reshape(B, self.n_tok, self.e)
        parts = [self.l_tok(tok).sum(dim=1)]
        if self.n_dem:
            n_dem_e = self.n_dem * self.e
            dem = features[:, n_tok_e:n_tok_e + n_dem_e].reshape(B, self.n_dem, self.e)
            parts.append(self.l_dem(dem).sum(dim=1))
            ctx = features[:, n_tok_e + n_dem_e:]
        else:
            ctx = features[:, n_tok_e:]
        parts.append(ctx)
        return self.mlp(th.cat(parts, dim=1))


def _init_head(head: nn.Module, small_layers) -> None:
    """trunk gain √2 · 최종 스코어 경로 gain 0.01 (프로젝트 PPO 위생 관례)."""
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    for lin in small_layers:
        nn.init.orthogonal_(lin.weight, gain=0.01)
        nn.init.constant_(lin.bias, 0.0)


class GoptMaskablePolicy(PointerMaskablePolicy):
    """부모의 net_arch(pi=[], vf=[256,256])·옵티마이저 재생성 계약을 승계하고 head 만 교체."""

    def __init__(self, *args, pooled_critic: bool = False, **kwargs):
        self._pooled_critic = bool(pooled_critic)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # 부모가 extractor 타입·action_space 차원을 검증하고 PointerActionNet 을 만든다(폐기됨).
        # JointPointerMaskablePolicy 와 동일한 교체 패턴.
        super()._build(lr_schedule)
        fx = self.features_extractor
        assert isinstance(fx, GoptTokenExtractor), \
            "features_extractor_class=GoptTokenExtractor 필요"
        self.action_net = GoptBilinearActionNet(
            fx.H, fx.embed_dim, fx.ctx_dim, n_class=fx.n_class, n_mode=fx.n_mode)
        _init_head(self.action_net,
                   (self.action_net.layer_2[0], self.action_net.bias_cm))
        if self._pooled_critic:
            self.mlp_extractor.value_net = PooledCriticNet(
                n_tok=fx.n_dst, embed_dim=fx.embed_dim, ctx_dim=fx.ctx_dim, n_dem=fx.n_dem)
        # ⚠️ 부모/super()._build 의 옵티마이저는 폐기된 head 를 가리킨다 → 재생성 필수
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


class PointerPooledCriticMaskablePolicy(PointerMaskablePolicy):
    """X6 — v10 actor(PointerActionNet) 그대로 + GOPT식 순열불변 pooled critic 만 교체.

    features 레이아웃은 v10 그대로 [tokens H*e | ctx] 이므로 n_dem=0 으로 풀링한다.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        fx = self.features_extractor
        self.mlp_extractor.value_net = PooledCriticNet(
            n_tok=fx.H, embed_dim=fx.embed_dim, ctx_dim=fx.ctx_dim, n_dem=0)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)
