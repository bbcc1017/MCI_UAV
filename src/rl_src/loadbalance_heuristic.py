"""부하균형 휴리스틱(발송상한 규칙) — 기존 64룰 best의 등급/모드 선택은 그대로 두고,
**목적지 선택만** '최근접 적격 병원'→'누적 발송 p_sent<T 인 가장 가까운 적격 병원'으로 교체.
차면 다음 가까운 병원으로 넘어가 한 병원 집중(=입원 포화→지연→사망)을 차단한다.

근거(시뮬로그 분석): 휴리스틱은 발송 게이트가 max_send(부하의 6~15배라 거의 안 걸림) 기준이라
최근접 1~2곳을 용량의 ~500%까지 채운다(gini 0.94, 3.7병원). RL은 분산(gini 0.79, 15병원, 점유 153%).
obs의 cap_remain 은 느슨한 max_send 점유라 진짜 입원병목을 못 비춘다 → 누적발송 p_sent(방출X=내가 만든 부하)
에 상한 T를 두는 게 직접적. T≈4 에서 17지역 occ +1.x, 어려운 농촌(강원)서 RL 능가.

make_cap_policy(rule_name, T): 정책 인터페이스 fn(ro, mask, env)->action. 적격집합=action_masks
(tier3/헬기장/게이트 인코딩)와 동일. T=∞ 면 휴리스틱과 동일(최근접).
"""
import numpy as np
from distill_policy import make_heuristic_policy, parse_rule, make_codec

H_DEFAULT = 47  # 2026-07-02 성남 헬기장 정정: 46→47


def make_cap_policy(rule_name, T, H=H_DEFAULT):
    ND = H + 1
    rule = parse_rule(rule_name)
    st = {"init": False, "codec": None}

    def fn(ro, mask, env):
        if not st["init"]:
            st["codec"] = make_codec(H)
            rule.set_seed(np.random.default_rng(0))
            rule.init_with_scenario({"EntityManager": env.en_manager})
            st["init"] = True
        _, encode = st["codec"]
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        c, d, m = rule.select(dobs)
        base = encode(0, 0, 0) if c < 0 else encode(c, d, m)
        def fb():
            if base < len(mask) and mask[base]:
                return base
            v = np.flatnonzero(mask)
            return int(v[0]) if v.size else 0
        if c < 0 or d == 0:
            return fb()
        HF = np.asarray(ro, np.float32)[:H * 4].reshape(H, 4)
        eta = HF[:, 2] if m == 0 else HF[:, 3]
        psent = np.asarray(dobs["p_sent"], float)
        elig = [i for i in range(H) if (lambda a: a < len(mask) and mask[a])(encode(c, i + 1, m))]
        if not elig:
            return fb()
        # 누적발송<T 인 적격 중 최근접(eta최소). 전부 T 이상이면 가장 덜 보낸 곳.
        under = [i for i in elig if psent[i] < T]
        if under:
            bi = under[int(np.argmin(eta[under]))]
        else:
            bi = elig[int(np.argmin(psent[elig]))]
        a = encode(c, bi + 1, m)
        return a if a < len(mask) and mask[a] else fb()

    return fn
