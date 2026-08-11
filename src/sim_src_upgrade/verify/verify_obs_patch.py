"""`fast_obs_patch` 등가 단위검사 — 규칙정책 게이트(G1/G2)가 못 잡는 구멍을 메운다.

규칙정책은 특징 obs 를 읽지 않으므로 `_patient_agg` 를 바꿔도 G1/G2 는 전부 PASS 다.
그래서 여기서 **원본 구현 vs 고속 구현을 직접 대조**한다. 무작위 + 경계 케이스 전수:

* 빈 배열 / 전부 0 / 전 단계 혼재
* 등급이 [0,N_CLASS) 밖인 오염 행 (원본은 어느 칸에도 안 세는 것이 정답)
* `MCI_CARED_OBS` 1(가시) / 0(단절 — 단계4를 단계3으로 접음) 양쪽

    python src/sim_src_upgrade/verify/verify_obs_patch.py
"""
from __future__ import annotations

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import numpy as np  # noqa: E402

from sim_src_upgrade._paths import ensure_paths  # noqa: E402


def _cases(rng, n_class, n_stage):
    """(라벨, p_states) 목록."""
    out = [("empty", np.zeros((0, 5), dtype=np.int32)),
           ("all_zero", np.zeros((100, 5), dtype=np.int32))]

    # 전 생애단계가 골고루 섞인 현실형
    for i, n in enumerate((1, 7, 100, 350)):
        p = np.zeros((n, 5), dtype=np.int32)
        p[:, 0] = rng.integers(0, n_class, size=n)
        prog = rng.integers(0, 5, size=n)           # 0..4 진행도
        p[:, 1] = (prog >= 1).astype(np.int32)      # rescued
        p[:, 2] = (prog >= 2).astype(np.int32)      # move
        p[:, 3] = (prog >= 3).astype(np.int32)      # moved
        p[:, 4] = (prog >= 4).astype(np.int32)      # cared
        out.append((f"mixed_n{n}_{i}", p))

    # 비단조 조합(원본 stage 배정의 덮어쓰기 순서를 그대로 타는지)
    p = np.zeros((60, 5), dtype=np.int32)
    p[:, 0] = rng.integers(0, n_class, size=60)
    p[:, 1:5] = rng.integers(0, 2, size=(60, 4))
    out.append(("nonmonotone", p))

    # 등급 오염 (음수·범위 초과)
    p = np.zeros((40, 5), dtype=np.int32)
    p[:, 0] = rng.integers(-2, n_class + 3, size=40)
    p[:, 1:5] = rng.integers(0, 2, size=(40, 4))
    out.append(("class_out_of_range", p))
    return out


def _inflight_cases(rng):
    """(라벨, obs, hos_num) — 이송중 카운트 경계 케이스."""
    H = 47
    out = [("empty_fleet", {"amb_states": np.zeros((0, 3), np.float32),
                            "uav_states": np.zeros((0, 3), np.float32)}, H),
           ("all_idle", {"amb_states": np.zeros((30, 3), np.float32),
                         "uav_states": np.zeros((26, 3), np.float32)}, H)]
    for i in range(6):
        def veh(n):
            st = np.zeros((n, 3), np.float32)
            st[:, 0] = rng.integers(0, H + 3, size=n)      # 범위 밖 dest 포함
            st[:, 1] = rng.random(n) * 30
            st[:, 2] = rng.integers(0, 5, size=n)          # severity 0 = 복귀 leg
            return st
        out.append((f"random_{i}", {"amb_states": veh(30), "uav_states": veh(26)}, H))
    # 키 자체가 없는 경우(UAV-only / AMB-only 시나리오)
    out.append(("amb_only", {"amb_states": np.array([[3, 5, 1], [0, 0, 0]], np.float32)}, H))
    return out


def main() -> int:
    ensure_paths()
    from sim_src_upgrade import fast_obs_patch

    n_fail = 0
    n_case = 0

    # ---- in_flight_by_hospital 등가 (구 코어 몽키패치 대상) ----
    import EntityManager as EM
    orig_if = EM.EntityManager.__dict__["in_flight_by_hospital"].__func__
    rng_if = np.random.default_rng(11)
    for label, obs, H in _inflight_cases(rng_if):
        a = orig_if(obs, H)
        b = fast_obs_patch._fast_in_flight(obs, H)
        n_case += 1
        if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
            n_fail += 1
            print(f"  FAIL in_flight case={label}\n    orig={a}\n    fast={b}")

    # ---- _fleet_agg 등가 (min/mean 은 원본 호출 유지 — 비트동일 요구) ----
    import importlib as _il

    import aggregate_obs as _AO0
    for comms in ("occ", "psent"):
        os.environ["MCI_CAP_GATE"] = comms
        _il.reload(_AO0)
        fast_obs_patch._AO = None
        orig_fa = _AO0.AggregateObsWrapper.__dict__["_fleet_agg"].__func__
        rng_fa = np.random.default_rng(77)
        fleet_cases = [("empty", np.zeros((0, 3), np.float32)),
                       ("all_idle", np.zeros((30, 3), np.float32))]
        for i in range(6):
            st = np.zeros((rng_fa.integers(1, 40), 3), np.float32)
            st[:, 0] = rng_fa.integers(0, 48, size=st.shape[0])
            st[:, 1] = rng_fa.random(st.shape[0]) * 40 * (rng_fa.random(st.shape[0]) > 0.3)
            st[:, 2] = rng_fa.integers(0, 5, size=st.shape[0])
            fleet_cases.append((f"rand_{i}", st))
        for label, st in fleet_cases:
            a = orig_fa(st)
            b = fast_obs_patch._fast_fleet_agg(st)
            n_case += 1
            if not (a.shape == b.shape and a.dtype == b.dtype
                    and np.array_equal(a.view(np.uint32), b.view(np.uint32))):  # 비트 단위 비교
                n_fail += 1
                print(f"  FAIL fleet_agg comms={comms} case={label}\n    orig={a}\n    fast={b}")
    os.environ["MCI_CAP_GATE"] = "occ"
    for cared in ("1", "0"):
        os.environ["MCI_CARED_OBS"] = cared
        import aggregate_obs as AO
        importlib.reload(AO)          # `_cared_visible` 이 캐시를 갖지 않도록 재로드
        fast_obs_patch._AO = None
        fast_obs_patch._ORIG.clear()

        orig = AO.AggregateObsWrapper.__dict__["_patient_agg"].__func__
        rng = np.random.default_rng(20260811)
        cases = _cases(rng, AO.AggregateObsWrapper.N_CLASS, AO.AggregateObsWrapper.N_STAGE)

        fast_obs_patch.apply()
        fast = AO.AggregateObsWrapper.__dict__["_patient_agg"].__func__

        for label, p in cases:
            a = orig(AO.AggregateObsWrapper, p)
            b = fast(AO.AggregateObsWrapper, p)
            n_case += 1
            same = (a.shape == b.shape and a.dtype == b.dtype
                    and np.array_equal(a, b))
            if not same:
                n_fail += 1
                print(f"  FAIL cared={cared} case={label}\n    orig={a}\n    fast={b}")
        fast_obs_patch.revert()

    ok = n_fail == 0
    print(f"[obs-patch] {'PASS' if ok else 'FAIL'} — {n_case}케이스 중 실패 {n_fail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
