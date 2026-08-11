"""고속 코어용 env 팩토리 — `rl_src/env_factory.py`·`viper_distill.make_feature_env` 대응.

원본과의 차이는 **어느 sim 코어를 쓰는가** 하나뿐이다. 래퍼(`HospitalFeatureWrapper`,
`_NormObs`)는 rl_src 원본을 그대로 재사용한다(수정 금지 대상). 래퍼들이 env 객체만
받으므로 신 코어 env 로도 그대로 동작한다.
"""
from __future__ import annotations

import numpy as np
import yaml

from ._paths import ensure_paths
from .core.MCIEnvironment_gymnasium import MCIEnvironment_gym
from .core.ScenarioManager import ScenarioManager


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_base_env_fast(config_path: str, seed: int = 0,
                       rule_test: bool = False, eval_mode: bool = False) -> MCIEnvironment_gym:
    """`env_factory.make_base_env` 와 동일 시퀀스 (rng 생성·소비 순서 포함)."""
    cfg = load_config(config_path)
    rng = np.random.default_rng(seed)
    sc = ScenarioManager(cfg, rng=rng)
    return MCIEnvironment_gym(scenario=sc.scenario, rng=rng,
                              rule_test=rule_test, eval_mode=eval_mode)


def make_feature_env_fast(config_path: str, norm=None, mask_only: bool = False):
    """`viper_distill.make_feature_env` 대응 — env_factory(seed)->env, env 1회 캐시.

    mask_only=True 면 특징 obs 벡터를 만들지 않는다(규칙·트리 정책 전용).
    action mask·decode 경로는 `HospitalFeatureWrapper` 그대로라 동작이 같다.
    """
    ensure_paths()
    from hospital_feature_wrapper import HospitalFeatureWrapper  # rl_src 원본

    if mask_only and norm:
        # 정규화는 관측을 실제로 쓴다는 뜻 = 신경망 경로 → mask_only 와 양립 불가
        raise ValueError("mask_only 와 vecnorm 은 함께 쓸 수 없다 — 관측을 쓰는 경로다")

    if norm:
        from viper_distill import _NormObs  # rl_src 원본
        wrap = lambda e: _NormObs(e, *norm)  # noqa: E731
    else:
        wrap = lambda e: e  # noqa: E731

    if mask_only:
        from .mask_only_wrapper import MaskOnlyFeatureWrapper
        head = MaskOnlyFeatureWrapper
    else:
        head = HospitalFeatureWrapper

    cache = {}

    def _f(seed: int = 0):
        if "e" not in cache:
            base = make_base_env_fast(config_path, seed=seed, rule_test=False, eval_mode=True)
            cache["e"] = wrap(head(base))
        return cache["e"]

    return _f


def make_feature_env_old(config_path: str, norm=None, mask_only: bool = False):
    """동일 인터페이스의 **구 코어** 팩토리 — 동치검증에서 짝으로 쓴다."""
    ensure_paths()
    from env_factory import make_base_env  # rl_src 원본 (구 sim_src 사용)
    from hospital_feature_wrapper import HospitalFeatureWrapper

    if norm:
        from viper_distill import _NormObs
        wrap = lambda e: _NormObs(e, *norm)  # noqa: E731
    else:
        wrap = lambda e: e  # noqa: E731

    if mask_only:
        from .mask_only_wrapper import MaskOnlyFeatureWrapper
        head = MaskOnlyFeatureWrapper
    else:
        head = HospitalFeatureWrapper

    cache = {}

    def _f(seed: int = 0):
        if "e" not in cache:
            base = make_base_env(config_path, seed=seed, rule_test=False, eval_mode=True)
            cache["e"] = wrap(head(base))
        return cache["e"]

    return _f
