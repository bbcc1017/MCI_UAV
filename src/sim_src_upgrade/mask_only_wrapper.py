"""규칙·트리 정책 전용 래퍼 — action mask 는 그대로, 특징 obs 생성만 생략.

배경
----
전수평가 드라이버(`v10_full_baselines`·`v16_baseline_alignment`·`shin_full_baselines`)는
`make_feature_env` 로 `HospitalFeatureWrapper` 를 씌우지만, 규칙정책은 obs 벡터를 쓰지 않는다
(`env.unwrapped.en_manager.get_full_obs()` 를 직접 읽는다). 그런데 그 402차원 벡터 생성이
프로파일상 전체의 29% 다 — 순수 낭비.

이 래퍼는 `_flat_obs` 만 무력화한다. `action_masks()`·`_decode`/`_encode`·H 패딩·
`step`/`reset` 흐름은 부모 구현 그대로라 **정책이 보는 mask 와 sim 에 전달되는 액션이 동일**하다.

⚠️ **관측을 실제로 읽는 코드에는 쓰지 마라.** 신경망 정책, 구 VIPER 슬롯트리
(`viper_distill.make_tree_policy` — 평탄 obs 로 예측), 그리고 obs 데이터셋을 모으는
드라이버(`bc_dataset`·`exit_labels`·`ncrp_labels`·`leaf_value`·`collect_decisions`·
`distill_zoo`·`viper_distill`)가 여기 해당한다.

안전장치: obs 자리에 0 대신 **NaN** 을 돌려준다. 0 벡터는 "그럴듯한 관측"처럼 보여
조용히 잘못된 학습데이터·잘못된 예측을 만들지만, NaN 은 신경망·sklearn·저장된 데이터셋에서
곧바로 드러난다(sklearn 은 "Input contains NaN" 으로 즉시 예외). 실수를 조용히 넘기지 않는 쪽을 택했다.

⚠️ v10 Track D/E 후보랭킹 트리(`tree_distill_policy`)는 `env.unwrapped` 에서 물리특징을
직접 만들므로 **영향 없다** — mask_only 와 함께 써도 된다.
"""
from __future__ import annotations

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from hospital_feature_wrapper import HospitalFeatureWrapper  # noqa: E402  (rl_src 원본)


class MaskOnlyFeatureWrapper(HospitalFeatureWrapper):
    """`HospitalFeatureWrapper` 에서 obs 벡터 생성만 뺀 판. 그 외 동작 동일."""

    is_mask_only = True

    def __init__(self, env):
        super().__init__(env)
        # 부모가 정한 observation_space 형상은 그대로 따르되 값은 만들지 않는다.
        shape = getattr(self.observation_space, "shape", None)
        n = int(np.prod(shape)) if shape else 0
        self._sentinel_obs = np.full(n, np.nan, dtype=np.float32)
        self._sentinel_obs.flags.writeable = False  # 실수로 쓰면 즉시 예외

    def _flat_obs(self, obs: dict) -> np.ndarray:  # type: ignore[override]
        return self._sentinel_obs
