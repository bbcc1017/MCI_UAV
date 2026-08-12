"""부하균형 휴리스틱(발송상한 규칙) — 기존 64룰 best의 등급/모드 선택은 그대로 두고,
**목적지 선택만** '최근접 적격 병원'→'누적 발송 p_sent<T 인 가장 가까운 적격 병원'으로 교체.
차면 다음 가까운 병원으로 넘어가 한 병원 집중(=입원 포화→지연→사망)을 차단한다.

근거(시뮬로그 분석): 휴리스틱은 발송 게이트가 max_send(부하의 6~15배라 거의 안 걸림) 기준이라
최근접 1~2곳을 용량의 ~500%까지 채운다(gini 0.94, 3.7병원). RL은 분산(gini 0.79, 15병원, 점유 153%).
obs의 cap_remain 은 느슨한 max_send 점유라 진짜 입원병목을 못 비춘다 → 누적발송 p_sent(방출X=내가 만든 부하)
에 상한 T를 두는 게 직접적. T≈4 에서 17지역 occ +1.x, 어려운 농촌(강원)서 RL 능가.

2026-07-04 리팩터(플랜 v2 Phase 0):
  - select_lb_action() 코어 추출 — T-메타 래퍼 등 재사용 대상. 반환에 meta(dict) 포함.
  - get_static_eta() — ETA 를 obs 슬라이스(ro[:H*4], essential F=4 하드코딩)가 아니라
    en_properties 에서 HospitalFeatureWrapper 와 동일식(lognormal 평균→최소정규화→MCI_ETA_CLIP)으로
    재계산 → obs 레이아웃(essential+load 등) 비의존.
  - 코덱을 mask 길이로 도출(_codec_from_mask) — uav=0 auto-pin(action 96) 형상에서도 flat
    인덱스 정합(구 make_codec(H) n_mode=2 하드코딩 버그 해소).
  - 멀티지역(FeatureMultiRegionEnv) 대응: per-env 상태(rule init·eta·codec)를 en_manager
    아이덴티티로 캐시 무효화(구 1회 init 은 첫 지역에 영구 바인딩).
  - make_adaptive_cap_policy() 신설 — T_eff=f(잔여 이송수요 L, 총용량 C) 계단
    (평상 4 · surge 8~16 · 용량 조임 2). tradeoff 스윕 §3.7 최적 T 곡선의 규칙화 = 비교 기준선.

make_cap_policy(rule_name, T): 정책 인터페이스 fn(ro, mask, env_unwrapped)->action (기존과 동일,
페어드 등가 테스트로 봉인). 적격집합=action_masks(tier3/헬기장/게이트 인코딩)와 동일. T=∞ 면 휴리스틱과 동일(최근접).
"""
import os

import numpy as np
from distill_policy import parse_rule

H_DEFAULT = 47  # 2026-07-02 성남 헬기장 정정: 46→47


# ---------------------------------------------------------------- 코어 유틸
def get_static_eta(env_unwrapped, H):
    """시나리오 정적 ETA 쌍 (eta_amb, eta_uav) — HospitalFeatureWrapper 와 동일 계산.

    lognormal 평균(amb/uav_HtoS_t[0], 부재 시 거리/속도 폴백) → 최소 ETA 정규화(최근접=1)
    → MCI_ETA_CLIP(기본 10배) 클립. argmin 랭킹이 obs 의 eta 열과 완전 일치해야
    구(obs 슬라이스) 구현과 페어드 등가가 성립한다(클립 동률 tie-break 포함).
    """
    props = env_unwrapped.en_manager.en_properties
    hp = props['hospital']
    ambp = props.get('ambulance', {})
    uavp = props.get('uav', {})
    d_road = np.asarray(hp.get('d_HtoS_road', hp.get('d_HtoS_euc', np.zeros(H))), dtype=np.float32)
    d_euc = np.asarray(hp.get('d_HtoS_euc', d_road), dtype=np.float32)
    amb_t = ambp.get('amb_HtoS_t', None)
    if amb_t is not None and len(amb_t[0]) == H:
        eta_amb = np.asarray(amb_t[0], dtype=np.float32)
    else:
        eta_amb = d_road * 60.0 / (float(ambp.get('amb_v', 40)) or 40.0)
    uav_t = uavp.get('uav_HtoS_t', None)
    if uav_t is not None and len(uav_t[0]) == H:
        eta_uav = np.asarray(uav_t[0], dtype=np.float32)
    else:
        eta_uav = d_euc * 60.0 / (float(uavp.get('uav_v', 80)) or 80.0)

    clip = float(os.environ.get("MCI_ETA_CLIP", "10.0"))

    def _norm_clip(eta):
        pos = eta[eta > 0]
        denom = float(pos.min()) if pos.size else 1.0
        return np.minimum(eta / denom, clip).astype(np.float32)

    return _norm_clip(eta_amb), _norm_clip(eta_uav)


def _codec_from_mask(mask_len, H):
    """mask 길이로 flat encode 도출 — wrapper 의 mode auto-pin(uav=0→2*(H+1)=96) 자동 정합."""
    n_dest = H + 1
    if mask_len == 2 * n_dest * 2:      # AMB+UAV: 192
        return lambda c, d, m: int(c) * (n_dest * 2) + int(d) * 2 + int(m)
    if mask_len == 2 * n_dest:          # mode 고정(AMB-only/UAV-only): 96 — m 무시
        return lambda c, d, m: int(c) * n_dest + int(d)
    raise ValueError(f"알 수 없는 action mask 길이 {mask_len} (H={H})")


def _preserve_hospital_rule_eligibility(rule, patient_class, eligible):
    """공통 hard mask 위에 원 휴리스틱의 병원 선택집합을 복원한다.

    action mask의 치료가능성은 도메인 제약이라 Yellow의 Tier3 치료를 허용한다. 반면
    Universal_Rule의 ``RedOnly``는 Yellow를 Tier3에 보내지 않는 *정책 규칙*이므로
    공통 mask에 넣으면 PPO·YellowNearest까지 바뀐다. LB가 목적지만 다시 고를 때 이
    정책 축을 잃지 않도록 여기서만 Tier3를 제외한다.
    """
    eligible = list(eligible)
    if int(patient_class) != 1 or getattr(rule, "hos_select", None) != "RedOnly":
        return eligible
    tier3 = {int(i) for i in np.asarray(rule.tier3_idx).reshape(-1)}
    return [i for i in eligible if i not in tier3]


def select_lb_action(rule, encode, mask, env_unwrapped, T, H, eta_amb, eta_uav, dobs=None):
    """LB 코어 1스텝: rule 의 (c,d,m)에서 목적지만 'p_sent<T 최근접 적격'으로 교체.

    반환 (flat_action, meta). meta = {"c","d","m","fallback"} — d 는 최종 목적지,
    fallback=True 는 의도 액션이 마스크에 막혀 대체된 경우.
    encode(c,d,m)->flat 은 호출측 주입(형상 인지 코덱). dobs 는 재사용 주입 가능(None 이면 조회).
    """
    if dobs is None:
        dobs = env_unwrapped.en_manager.get_full_obs()
        dobs["time"] = env_unwrapped.ev_manager.time
    c, d, m = rule.select(dobs)
    base = encode(0, 0, 0) if c < 0 else encode(c, d, m)

    def fb(intended_ok):
        if base < len(mask) and mask[base]:
            return base, {"c": c, "d": d, "m": m, "fallback": not intended_ok}
        v = np.flatnonzero(mask)
        return (int(v[0]) if v.size else 0), {"c": c, "d": d, "m": m, "fallback": True}

    if c < 0 or d == 0:
        return fb(intended_ok=True)
    eta = eta_amb if m == 0 else eta_uav
    psent = np.asarray(dobs["p_sent"], float)
    elig = [i for i in range(H) if (lambda a: a < len(mask) and mask[a])(encode(c, i + 1, m))]
    elig = _preserve_hospital_rule_eligibility(rule, c, elig)
    if not elig:
        return fb(intended_ok=False)
    # 누적발송<T 인 적격 중 최근접(eta최소). 전부 T 이상이면 가장 덜 보낸 곳.
    under = [i for i in elig if psent[i] < T]
    if under:
        bi = under[int(np.argmin(eta[under]))]
    else:
        bi = elig[int(np.argmin(psent[elig]))]
    a = encode(c, bi + 1, m)
    if a < len(mask) and mask[a]:
        return a, {"c": c, "d": bi + 1, "m": m, "fallback": False}
    return fb(intended_ok=False)


class _EnvState:
    """per-env(en_manager 아이덴티티) 상태 — rule init·정적 eta·코덱. 멀티지역 재초기화 지원."""

    def __init__(self, rule, H):
        self._rule = rule
        self._H = H
        self._em = None
        self.encode = None
        self.eta = None

    def sync(self, env_unwrapped, mask_len):
        if self._em is not env_unwrapped.en_manager:
            self.encode = _codec_from_mask(mask_len, self._H)
            self.eta = get_static_eta(env_unwrapped, self._H)
            self._rule.set_seed(np.random.default_rng(0))
            self._rule.init_with_scenario({"EntityManager": env_unwrapped.en_manager})
            self._em = env_unwrapped.en_manager


# ---------------------------------------------------------------- 정책 팩토리
def make_cap_policy(rule_name, T, H=H_DEFAULT):
    """고정 T 발송상한 정책 fn(ro, mask, env_unwrapped)->action (기존 시그니처·행동 유지)."""
    rule = parse_rule(rule_name)
    st = _EnvState(rule, H)

    def fn(ro, mask, env):
        st.sync(env, len(mask))
        a, meta = select_lb_action(rule, st.encode, mask, env, T, H, *st.eta)
        fn.last_meta = meta
        return a

    fn.last_meta = None
    return fn


def make_adaptive_cap_policy(rule_name, H=H_DEFAULT,
                             t_norm=4.0, t_surge8=8.0, t_surge16=16.0, t_tight=2.0,
                             surge_bins=(80.0, 160.0), tight_capa=350.0):
    """적응 T 발송상한 정책 — T_eff = f(잔여 이송수요 L, 총 발송용량 C) 계단 규칙.

    tradeoff 스윕(§3.7)의 regime 별 최적 T(평상 4 · surge 8→16 · 용량 조임 2)를 런타임
    신호로 규칙화한 **비교 기준선**(학습 불요):
      - C = Σ max_send (시나리오 상수) < tight_capa   → T=t_tight   (용량 조임 regime)
      - L = 미발송 R/Y 수(p_states: class≤1 & move_start==0) ≥ surge_bins[1] → t_surge16
      - L ≥ surge_bins[0]                              → t_surge8
      - 그 외(현행 N=100 이면 L≈40 으로 전 에피소드 t_norm)  → t_norm
    L 은 결정 시점마다 재계산(에피소드 진행에 따라 자연 감쇠 — 초기 surge 에 넓게 열고
    소진되면 조이는 방향). 임계 기본값은 urgent≈0.4N 앵커(N100→40, N350→140, N500→200).
    """
    rule = parse_rule(rule_name)
    st = _EnvState(rule, H)

    def _t_eff(env_unwrapped, dobs):
        hp = env_unwrapped.en_manager.en_properties['hospital']
        C = float(np.sum(hp['hos_max_send']))
        if C < tight_capa:
            return t_tight
        p = np.asarray(dobs['p_states'], dtype=np.float32)
        L = float(np.sum((p[:, 0] <= 1) & (p[:, 2] == 0)))  # R/Y & 미발송(move_start==0)
        if L >= surge_bins[1]:
            return t_surge16
        if L >= surge_bins[0]:
            return t_surge8
        return t_norm

    def fn(ro, mask, env):
        st.sync(env, len(mask))
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        T = _t_eff(env, dobs)
        a, meta = select_lb_action(rule, st.encode, mask, env, T, H, *st.eta, dobs=dobs)
        meta["T_eff"] = T
        fn.last_meta = meta
        return a

    fn.last_meta = None
    return fn
