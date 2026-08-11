"""rl_src 관측 핫스팟의 **등가 교체** — 원본 파일은 건드리지 않는다(opt-in 몽키패치).

`src/rl_src/aggregate_obs.py` 는 수정 금지 대상이지만(실행 중 드라이버의 소스 해시·
동결 산출물 정합), 그 안의 `_patient_agg` 가 관측 생성 비용의 큰 몫을 차지한다.
그래서 여기서 **명시적으로 import 했을 때만** 같은 값을 더 싸게 내는 구현으로 바꾼다.

    from sim_src_upgrade import fast_obs_patch
    fast_obs_patch.apply()

되돌리려면 `fast_obs_patch.revert()`.

등가성은 `verify/verify_equivalence.py`(G1/G2)와 `verify/verify_rl_obs.py`(G5)가 확인한다.
"""
from __future__ import annotations

import numpy as np

from ._paths import ensure_paths

_ORIG: dict = {}
_AO = None  # aggregate_obs 모듈 참조 (패치 후 `_cared_visible` 조회에 사용)


def _module():
    global _AO
    if _AO is None:
        ensure_paths()
        import aggregate_obs as AO
        _AO = AO
    return _AO


def _fast_patient_agg(cls, p_states: np.ndarray) -> np.ndarray:
    """`AggregateObsWrapper._patient_agg` 와 값이 같은 구현.

    원본은 (등급 4 × 단계 5) 칸마다 `np.sum((c==ci)&(stage==si))` 를 돌려 numpy 호출이
    60회였다. 여기서는 `c*N_STAGE+stage` 를 bincount 한 번으로 센다. 등급이 [0,N_CLASS)
    밖인 환자를 제외하는 것도 원본과 같다(원본에서는 어떤 `(c==ci)` 에도 안 걸린다).
    """
    n_class, n_stage = cls.N_CLASS, cls.N_STAGE
    if p_states.shape[0] == 0:
        return np.zeros(n_class * n_stage, dtype=np.float32)

    c = p_states[:, 0].astype(int)
    rescued, move, moved, cared = (p_states[:, 1], p_states[:, 2],
                                   p_states[:, 3], p_states[:, 4])
    stage = np.zeros(p_states.shape[0], dtype=int)
    stage[(rescued == 1) & (move == 0)] = 1
    stage[(move == 1) & (moved == 0)] = 2
    if _module()._cared_visible():
        stage[(moved == 1) & (cared == 0)] = 3
        stage[cared == 1] = 4
    else:
        stage[moved == 1] = 3

    keep = (c >= 0) & (c < n_class) & (stage >= 0) & (stage < n_stage)
    flat = c[keep] * n_stage + stage[keep]
    counts = np.bincount(flat, minlength=n_class * n_stage)
    return counts[: n_class * n_stage].astype(np.float32)


def _fast_in_flight(obs, hos_num):
    """구 `EntityManager.in_flight_by_hospital` 의 등가 고속판(신 코어와 동일 구현).

    `hospital_feature_wrapper._dyn` 이 함수 안에서 구 모듈을 직접 import 하므로
    RL 관측 경로는 이 패치로만 빨라진다. 판정 순서(① float 비교 ② 절단 ③ 범위)는 원본 그대로.
    """
    inflight = np.zeros(hos_num, dtype=np.int32)
    for key in ('amb_states', 'uav_states'):
        st = obs.get(key)
        if st is None or len(st) == 0:
            continue
        st = np.asarray(st)
        carrying = (st[:, 0] >= 1) & (st[:, 2] > 0)
        if not carrying.any():
            continue
        d = st[carrying, 0].astype(int)
        keep = (d >= 1) & (d <= hos_num)
        if keep.any():
            inflight += np.bincount(d[keep] - 1, minlength=hos_num).astype(np.int32)
    return inflight


def _fast_fleet_agg(states):
    """`AggregateObsWrapper._fleet_agg` 등가 고속판.

    ⚠️ `min()`/`mean()` 은 **원본 호출 그대로** 둔다 — float32 누산 순서가 바뀌면
    마지막 비트가 달라질 수 있다. 여기서 줄인 것은 정수 카운트와 불필요한 복사뿐이다:
      * `astype(np.float32)` : 입력이 이미 float32 면 값이 같은 복사본을 만들 뿐이라 생략
      * `(~busy).sum()`      : `len(busy) - n_busy` 로 대체(정수 항등)
      * `np.sum(bool)`       : `np.count_nonzero` 로 대체(같은 개수)
    """
    if states.shape[0] == 0:
        return np.zeros(5, dtype=np.float32)
    if states.dtype == np.float32:
        tr = states[:, 1]
        sev = states[:, 2]
    else:
        tr = states[:, 1].astype(np.float32)
        sev = states[:, 2].astype(np.float32)
    busy = tr > 1e-6
    n_busy_i = int(busy.sum())
    if n_busy_i > 0 and _module()._comms_available():
        tb = tr[busy]
        min_t = float(tb.min())
        mean_t = float(tb.mean())
    else:
        min_t = mean_t = 0.0
    n_crit = float(np.count_nonzero((sev == 1) | (sev == 2)))
    return np.array([float(states.shape[0] - n_busy_i), float(n_busy_i), min_t, mean_t, n_crit],
                    dtype=np.float32)


def apply() -> None:
    """등가 고속 구현으로 교체(멱등)."""
    if _ORIG:
        return
    AO = _module()
    import EntityManager as EM  # 구 코어 (src/sim_src) — 파일은 그대로, 런타임 속성만 교체

    _ORIG["patient_agg"] = AO.AggregateObsWrapper.__dict__["_patient_agg"]
    _ORIG["fleet_agg"] = AO.AggregateObsWrapper.__dict__["_fleet_agg"]
    _ORIG["in_flight"] = EM.EntityManager.__dict__["in_flight_by_hospital"]
    AO.AggregateObsWrapper._patient_agg = classmethod(_fast_patient_agg)
    AO.AggregateObsWrapper._fleet_agg = staticmethod(_fast_fleet_agg)
    EM.EntityManager.in_flight_by_hospital = staticmethod(_fast_in_flight)


def revert() -> None:
    """원본 구현으로 복구."""
    if not _ORIG:
        return
    import EntityManager as EM

    AO = _module()
    AO.AggregateObsWrapper._patient_agg = _ORIG.pop("patient_agg")
    AO.AggregateObsWrapper._fleet_agg = _ORIG.pop("fleet_agg")
    EM.EntityManager.in_flight_by_hospital = _ORIG.pop("in_flight")
    _ORIG.clear()


def is_applied() -> bool:
    return bool(_ORIG)
