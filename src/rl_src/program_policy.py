"""해석가능 프로그램 정책 (플랜 v2 Phase 3-B) — RL 통찰을 명시적 규칙으로 합성.

Phase 3-A(UAV 운용규칙)에서 추출한 구조를 파라미터화 프로그램으로:
  - class : 64룰 best 우선순위(Red 우선 + Yellow 현장적체 override) — rule.select 그대로
  - dest  : LB 코어 = 적격(마스크: tier3·헬기장·occ게이트) 중 누적발송 p_sent<T (정원제)
  - mode  : ★RL 규칙(시간절감형) = "정원여유 적격 중, UAV 최속(raw분) 목적지가 AMB 최속보다
            factor 배 이상 빠르면 UAV, 아니면 AMB". Phase3-A 1차 rule("Red&먼tier3→UAV")은
            도심 tier3가 전부 원거리라 Red를 과잉 원거리 이송 → LB-T4 미달(정직 기록). RL의
            실제 용법은 "UAV가 더 빠를 때"라 raw 시간 비교로 수정.

파라미터(closed-loop 성능으로 선택 — 정태 acc 아님): T(정원), uav_time_factor(UAV 채택 문턱:
t_uav < factor·t_amb, <1일수록 보수적), uav_red_only(UAV를 Red에만), uav_tier3_pref(UAV 후보를
tier3 헬기장으로 제한).

정책 fn(ro, mask, env_unwrapped)->action. obs 비의존. uav0(action96)은 자동 UAV 미사용.
make_cap_policy 와 동일 인터페이스.
"""
import numpy as np
from distill_policy import parse_rule
from loadbalance_heuristic import _codec_from_mask, H_DEFAULT


def _raw_times(u, H):
    """raw(분) AMB/UAV 현장→병원 이송시간 — 모드 간 직접 비교용(정규화 전)."""
    props = u.en_manager.en_properties
    hp = props['hospital']
    ambp = props.get('ambulance', {}); uavp = props.get('uav', {})
    d_road = np.asarray(hp.get('d_HtoS_road', hp.get('d_HtoS_euc', np.zeros(H))), float)
    d_euc = np.asarray(hp.get('d_HtoS_euc', d_road), float)
    at = ambp.get('amb_HtoS_t', None)
    t_amb = np.asarray(at[0], float) if (at is not None and len(at[0]) == H) \
        else d_road * 60.0 / (float(ambp.get('amb_v', 40)) or 40.0)
    ut = uavp.get('uav_HtoS_t', None)
    t_uav = np.asarray(ut[0], float) if (ut is not None and len(ut[0]) == H) \
        else d_euc * 60.0 / (float(uavp.get('uav_v', 80)) or 80.0)
    return t_amb, t_uav


def make_program_policy(rule_name, T=4, uav_time_factor=0.8, uav_red_only=True,
                        uav_tier3_pref=False, H=H_DEFAULT):
    rule = parse_rule(rule_name)
    st = {"em": None, "encode": None, "t_amb": None, "t_uav": None, "tier3": None}

    def _sync(u, mask_len):
        if st["em"] is u.en_manager:
            return
        st["encode"] = _codec_from_mask(mask_len, H)
        st["t_amb"], st["t_uav"] = _raw_times(u, H)
        hp = u.en_manager.en_properties['hospital']
        st["tier3"] = (np.asarray(hp['hos_tier']).reshape(-1) == 3).astype(bool)
        rule.set_seed(np.random.default_rng(0))
        rule.init_with_scenario({"EntityManager": u.en_manager})
        st["em"] = u.en_manager

    def fn(ro, mask, env):
        _sync(env, len(mask))
        enc = st["encode"]; t_amb = st["t_amb"]; t_uav = st["t_uav"]; tier3 = st["tier3"]
        dobs = env.en_manager.get_full_obs(); dobs["time"] = env.ev_manager.time
        c, d0, m0 = rule.select(dobs)
        base = enc(0, 0, 0) if c < 0 else enc(c, d0, m0)

        def _valid(a):
            return 0 <= a < len(mask) and mask[a]

        def fb():
            if _valid(base):
                return base
            v = np.flatnonzero(mask)
            return int(v[0]) if v.size else 0

        if c < 0 or d0 == 0:
            return fb()
        psent = np.asarray(dobs["p_sent"], float)
        has_uav = (len(mask) == 2 * (H + 1) * 2)

        # AMB 정원여유 적격 → 최속(raw분)
        elig_amb = [i for i in range(H) if _valid(enc(c, i + 1, 0))]
        under_amb = [i for i in elig_amb if psent[i] < T]
        pool_amb = under_amb if under_amb else elig_amb
        if not pool_amb:
            return fb()
        best_amb = pool_amb[int(np.argmin(t_amb[pool_amb]))] if under_amb \
            else pool_amb[int(np.argmin(psent[pool_amb]))]

        # ---- 시간절감형 UAV 규칙 ----
        if has_uav and (c == 0 or not uav_red_only):
            elig_uav = [i for i in range(H) if _valid(enc(c, i + 1, 1))]
            under_uav = [i for i in elig_uav if psent[i] < T]
            if uav_tier3_pref:
                cand = [i for i in under_uav if tier3[i]] or under_uav
            else:
                cand = under_uav
            if cand:
                best_uav = cand[int(np.argmin(t_uav[cand]))]
                # UAV 최속이 AMB 최속보다 factor 배 이상 빠르면 UAV
                if t_uav[best_uav] < uav_time_factor * t_amb[best_amb]:
                    a = enc(c, best_uav + 1, 1)
                    if _valid(a):
                        return a

        a = enc(c, best_amb + 1, 0)
        return a if _valid(a) else fb()

    return fn
