# -*- coding: utf-8 -*-
"""valid 열(0/1) 정규화 면제 VecNormalize (v6 Track A-3).

패딩 병원 식별용 valid 열은 VecNormalize 아핀변환((x−mean)/std)이 0/1 을 비영값으로
옮겨 "패딩 행 = all-zero 파생 식별"을 붕괴시킨다. 이 서브클래스는 exempt_idx 열을
normalize_obs 에서 원값(0/1) 그대로 통과시켜 extractor(포인터 마스크드 풀링)가 패딩
행을 견고히 식별하게 한다.

exempt_idx 열은 러닝 통계(obs_rms)엔 정상 포함(정규화 파이프라인 무손상)되나 사용처는
없다 — 출력에서 원본 obs 값으로 되돌려지므로. save/load 는 순정 VecNormalize 계약을
따른다(__getstate__ 가 venv 제거·exempt_idx 는 일반 attr 라 자동 보존).

⚠️ 로드(pickle) 시 이 모듈이 import 가능해야 함 — pointer_policy import 전례와 동일
(train_ppo_feature 상단에서 명시 import). 구 pickle(순정 VecNormalize) 로드는 무영향.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize


class PadAwareVecNormalize(VecNormalize):
    """exempt_idx 열의 정규화를 면제하는 VecNormalize (flat Box obs 전용).

    Parameters
    ----------
    venv : VecEnv
        감쌀 벡터 환경.
    exempt_idx : Iterable[int]
        정규화 면제할 obs 평탄 인덱스(예: 각 병원 valid 열 i*F+(F-1)). 정렬·중복제거.
    """

    def __init__(self, venv, exempt_idx=(), **kwargs):
        super().__init__(venv, **kwargs)
        self.exempt_idx = np.asarray(sorted(set(int(i) for i in exempt_idx)), dtype=int)

    def normalize_obs(self, obs):
        # dict obs 미지원(이 프로젝트는 flat Box) — 명시 assert.
        assert not isinstance(obs, dict), \
            "PadAwareVecNormalize 는 flat Box obs 전용(dict obs 미지원)"
        out = super().normalize_obs(obs)  # 항상 fresh array(deepcopy 후 정규화) — in-place 안전
        if self.exempt_idx.size:
            src = np.asarray(obs)
            # 배치 (n, dim)·단건 (dim,) 모두 지원 — 마지막 축 인덱싱(ellipsis).
            out[..., self.exempt_idx] = src[..., self.exempt_idx]
        return out
