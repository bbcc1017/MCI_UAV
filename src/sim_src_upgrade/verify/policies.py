"""동치검증용 정책 팩토리 — 어느 코어의 RuleManager 를 쓸지 고를 수 있다.

`distill_policy.make_heuristic_policy` 와 **동작이 같아야** 하므로 그 본체를 그대로 옮기되
규칙 클래스 출처만 파라미터화했다(`core="old"|"new"`).

왜 필요한가
-----------
* `core="old"`  : 신 sim 코어 + 기존 rl_src 정책. 고속 드라이버가 실제로 도는 조합.
* `core="new"`  : 신 sim 코어 + 신 RuleManager. S2 에서 RuleManager 를 최적화하면 이쪽이 정본.
둘 다 구 코어 결과와 비트동일해야 한다.
"""
from __future__ import annotations

import numpy as np

from .._paths import ensure_paths

ensure_paths()

from loadbalance_heuristic import _codec_from_mask  # noqa: E402  (순수 함수, rl_src 원본)


def _rule_modules(core: str):
    if core == "new":
        from ..core import RuleManager as rm
        from ..core import ShinHeuristics as sh
        from ..core import ShinAlignedHeuristics as sa
        return rm, sh, sa
    if core == "old":
        import RuleManager as rm  # 구 코어 (src/sim_src)
        import ShinHeuristics as sh
        import ShinAlignedHeuristics as sa
        return rm, sh, sa
    raise ValueError(f"core 는 'old'|'new' — 받은 값 {core!r}")


def parse_rule(rule_name: str, core: str = "new"):
    """`distill_policy.parse_rule` 과 동일 규약, 규칙 클래스 출처만 교체."""
    rm, sh, _sa = _rule_modules(core)
    p = [x.strip() for x in rule_name.split(",")]
    if len(p) == 2 and p[0].startswith("Shin ") and p[1].startswith("Mode "):
        return sh.ShinHeuristicRule(
            p[0].replace("Shin ", "", 1).strip(),
            p[1].replace("Mode ", "", 1).strip(),
        )
    if len(p) != 4:
        raise ValueError(f"알 수 없는 휴리스틱 규칙명: {rule_name}")
    return rm.Universal_Rule(p[0], p[1], p[2].replace("Red", "", 1).strip(),
                             p[3].replace("Yellow", "", 1).strip())


def make_rule_policy(rule_name: str, policy_seed: int = 0, core: str = "new"):
    """flat hard-mask 규칙정책. `distill_policy.make_heuristic_policy` 와 동일 동작."""
    rule = parse_rule(rule_name, core=core)
    state = {"init": False, "encode": None}

    def fn(obs, mask, env):
        if not state["init"]:
            mode_free = (int(getattr(env, "amb_num", 0)) > 0
                         and int(getattr(env, "uav_num", 0)) > 0)
            H_layout = len(mask) // 4 - 1 if mode_free else len(mask) // 2 - 1
            state["encode"] = _codec_from_mask(len(mask), H_layout)
            rule.set_seed(np.random.default_rng(policy_seed))
            rule.init_with_scenario({"EntityManager": env.en_manager})
            state["init"] = True
        encode = state["encode"]
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        c, d, m = rule.select(dobs)
        a = encode(0, 0, 0) if c < 0 else encode(c, d, m)
        if a < len(mask) and mask[a]:
            return int(a)
        v = np.flatnonzero(mask)
        return int(v[0]) if v.size else 0

    return fn


def make_shin_aligned_policy(spec, policy_seed: int = 0, core: str = "new"):
    """`ShinAlignedHeuristics` 정합변형 정책 (v16 비교군).

    spec = (method, hospital_rule, mode_rule) — 예: ("Threshold", "RedOnly", "OnlyAMB")
    """
    _rm, _sh, sa = _rule_modules(core)
    rule = sa.ShinHospitalAlignedRule(*spec)
    state = {"init": False, "encode": None}

    def fn(obs, mask, env):
        if not state["init"]:
            mode_free = (int(getattr(env, "amb_num", 0)) > 0
                         and int(getattr(env, "uav_num", 0)) > 0)
            H_layout = len(mask) // 4 - 1 if mode_free else len(mask) // 2 - 1
            state["encode"] = _codec_from_mask(len(mask), H_layout)
            rule.set_seed(np.random.default_rng(policy_seed))
            rule.init_with_scenario({"EntityManager": env.en_manager})
            state["init"] = True
        encode = state["encode"]
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        c, d, m = rule.select(dobs)
        a = encode(0, 0, 0) if c < 0 else encode(c, d, m)
        if a < len(mask) and mask[a]:
            return int(a)
        v = np.flatnonzero(mask)
        return int(v[0]) if v.size else 0

    return fn
