# -*- coding: utf-8 -*-
"""v10 후보랭킹 의사결정나무의 특징·정책 공통 모듈.

구 VIPER의 ``평탄 병원슬롯 관측 → 192-class`` 트리는 병원 인덱스에 종속된다. 여기서는
현재 마스크가 허용한 각 ``[class, destination, mode]`` 후보를 동일한 물리 특징으로
표현하고, 결정나무가 후보 점수를 예측한 뒤 최고점을 고른다. 따라서 병원 순서와 실병원
개수에 독립적이며 세 행동축의 상호작용을 하나의 트리 안에서 유지한다.

정보 단계는 누적식이다.

* I0_MINIMAL: 정적 지리·의료 인프라 + 현장 R/Y·가용차량·시간
* I1_FIELD: I0 + 현장 구조/이송 기록 + 내가 보낸 병원별 발송·in-flight
* I2_TELEMETRY: I1 + 차량 운행/복귀 ETA 텔레메트리
* I3_CONNECTED: I2 + 병원 실시간 점유·잔여용량·치료진행 확인

모든 단계가 ``action_masks``를 안전 제약층으로 공통 사용한다. 따라서 I0는 완전 무통신이
아니라 "중앙 안전마스크가 제공된 현장 최소정보 랭커"로 해석해야 한다.
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from aggregate_obs import AggregateObsWrapper
from score_features import build_ctx, compute_static


FEATURE_NAMES = [
    # 후보 행동·정적 인프라
    "is_red", "is_yellow", "is_uav", "is_stay",
    "eta_norm", "eta_rank", "eta_raw_min", "uav_advantage_min",
    "is_tier3", "has_helipad", "max_send", "n_valid_actions",
    # 현장 최소정보
    "red_at_site", "yellow_at_site", "amb_available", "uav_available", "time_min",
    # 현장 구조·발송 기록
    "red_unrescued", "yellow_unrescued", "red_in_transport", "yellow_in_transport",
    "cand_p_sent", "cand_p_sent_rel", "cand_in_flight",
    "total_p_sent", "total_in_flight",
    # 차량 텔레메트리
    "amb_busy", "amb_min_return", "amb_mean_return",
    "uav_busy", "uav_min_return", "uav_mean_return",
    "fleet_critical", "cand_arrive_min",
    # 병원 연계정보
    "cand_cap_remain", "cand_occ", "cand_occ_ratio", "rho",
    "red_at_hospital", "yellow_at_hospital", "red_done", "yellow_done",
    "total_cap_remain",
]

INFO_LEVELS = {
    "I0_MINIMAL": list(range(0, 17)),
    "I1_FIELD": list(range(0, 26)),
    "I2_TELEMETRY": list(range(0, 34)),
    "I3_CONNECTED": list(range(0, len(FEATURE_NAMES))),
}

INFO_LABELS = {
    "I0_MINIMAL": "현장최소",
    "I1_FIELD": "현장기록",
    "I2_TELEMETRY": "차량텔레메트리",
    "I3_CONNECTED": "병원연계",
}

COMPLEXITY_SPECS = {
    "C1": {"max_depth": 3, "max_leaf_nodes": 8, "min_samples_leaf": 300},
    "C2": {"max_depth": 5, "max_leaf_nodes": 32, "min_samples_leaf": 150},
    "C3": {"max_depth": 7, "max_leaf_nodes": 128, "min_samples_leaf": 75},
    "C4": {"max_depth": 10, "max_leaf_nodes": 512, "min_samples_leaf": 40},
}


def _layout(mask_len: int, h_pad: int = 47) -> tuple[int, int]:
    """mask 길이로 레이아웃의 (H, n_mode)를 복원한다."""
    if mask_len == 2 * (h_pad + 1) * 2:
        return h_pad, 2
    if mask_len == 2 * (h_pad + 1):
        return h_pad, 1
    if mask_len % 4 == 0:
        return mask_len // 4 - 1, 2
    if mask_len % 2 == 0:
        return mask_len // 2 - 1, 1
    raise ValueError(f"지원하지 않는 action mask 길이: {mask_len}")


def decode_action(action: int, mask_len: int, h_pad: int = 47) -> tuple[int, int, int]:
    """평탄 action을 (class, dest, mode)로 디코드."""
    H, n_mode = _layout(mask_len, h_pad)
    n_dest = H + 1
    a = int(action)
    if n_mode == 1:
        return a // n_dest, a % n_dest, 0
    c = a // (n_dest * 2)
    rem = a % (n_dest * 2)
    return c, rem // 2, rem % 2


@dataclass
class _Static:
    manager_id: int
    H: int
    eta_amb: np.ndarray
    eta_uav: np.ndarray
    t_amb: np.ndarray
    t_uav: np.ndarray
    is_tier3: np.ndarray
    helipad: np.ndarray
    max_send: np.ndarray


class ActionFeatureBuilder:
    """unwrapped simulator 상태에서 유효 후보별 물리 특징을 생성."""

    def __init__(self, h_pad: int = 47):
        self.h_pad = int(h_pad)
        self._static: _Static | None = None

    def _get_static(self, env) -> _Static:
        manager_id = id(env.en_manager)
        if self._static is not None and self._static.manager_id == manager_id:
            return self._static
        base = compute_static(env)
        hp = env.en_manager.en_properties["hospital"]
        H = int(base["H"])
        helipad = np.zeros(H, dtype=np.float32)
        idx = np.asarray(hp.get("hos_helipad_idx", []), dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < H)]
        helipad[idx] = 1.0
        self._static = _Static(
            manager_id=manager_id,
            H=H,
            eta_amb=np.asarray(base["eta_amb"], dtype=np.float32),
            eta_uav=np.asarray(base["eta_uav"], dtype=np.float32),
            t_amb=np.asarray(base["t_amb"], dtype=np.float32),
            t_uav=np.asarray(base["t_uav"], dtype=np.float32),
            is_tier3=np.asarray(base["is_tier3"], dtype=np.float32),
            helipad=helipad,
            max_send=np.asarray(base["max_send"], dtype=np.float32),
        )
        return self._static

    @staticmethod
    def _candidate_arrival(dobs: dict, H: int) -> np.ndarray:
        """그 병원행 차량의 최소 잔여시간(분), 없으면 240분."""
        out = np.full(H, 240.0, dtype=np.float32)
        for key in ("amb_states", "uav_states"):
            states = np.asarray(dobs.get(key, ()), dtype=np.float32)
            if states.size == 0:
                continue
            valid = (states[:, 0] >= 1) & (states[:, 2] > 0)
            dest = states[valid, 0].astype(int) - 1
            remain = states[valid, 1]
            keep = (dest >= 0) & (dest < H)
            np.minimum.at(out, dest[keep], remain[keep])
        return out

    def build(self, env, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(유효 action ids, full 특징행렬)을 반환."""
        mask = np.asarray(mask, dtype=bool)
        actions = np.flatnonzero(mask).astype(np.int16)
        if actions.size == 0:
            raise RuntimeError("유효 action이 0개")
        st = self._get_static(env)
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        ctx = build_ctx(env, static={
            "H": st.H, "eta_amb": st.eta_amb, "eta_uav": st.eta_uav,
            "t_amb": st.t_amb, "t_uav": st.t_uav, "is_tier3": st.is_tier3,
            "max_send": st.max_send,
        }, dobs=dobs)
        pa = AggregateObsWrapper._patient_agg(np.asarray(dobs["p_states"]))[:10]
        amb = AggregateObsWrapper._fleet_agg(np.asarray(dobs["amb_states"]))
        uav = AggregateObsWrapper._fleet_agg(np.asarray(dobs["uav_states"]))
        arrival = self._candidate_arrival(dobs, st.H)

        decoded = np.asarray(
            [decode_action(int(a), len(mask), self.h_pad) for a in actions], dtype=int
        )
        cls, dest, mode = decoded[:, 0], decoded[:, 1], decoded[:, 2]
        hospital = dest - 1
        nonstay = (dest > 0) & (hospital < st.H)
        n = len(actions)

        eta_norm = np.zeros(n, dtype=np.float32)
        eta_raw = np.zeros(n, dtype=np.float32)
        eta_rank = np.zeros(n, dtype=np.float32)
        uav_adv = np.zeros(n, dtype=np.float32)
        is_tier3 = np.zeros(n, dtype=np.float32)
        helipad = np.zeros(n, dtype=np.float32)
        max_send = np.zeros(n, dtype=np.float32)
        p_sent = np.zeros(n, dtype=np.float32)
        p_sent_rel = np.zeros(n, dtype=np.float32)
        in_flight = np.zeros(n, dtype=np.float32)
        arrive = np.zeros(n, dtype=np.float32)
        cap_remain = np.zeros(n, dtype=np.float32)
        occ = np.zeros(n, dtype=np.float32)
        occ_ratio = np.zeros(n, dtype=np.float32)

        h_states = np.asarray(dobs["h_states"], dtype=np.float32)
        for i in np.flatnonzero(nonstay):
            h, m = hospital[i], mode[i]
            eta_norm[i] = st.eta_uav[h] if m == 1 else st.eta_amb[h]
            eta_raw[i] = st.t_uav[h] if m == 1 else st.t_amb[h]
            uav_adv[i] = st.t_amb[h] - st.t_uav[h]
            is_tier3[i] = st.is_tier3[h]
            helipad[i] = st.helipad[h]
            max_send[i] = st.max_send[h]
            p_sent[i] = ctx["p_sent"][h]
            in_flight[i] = ctx["in_flight"][h]
            arrive[i] = arrival[h]
            cap_remain[i] = ctx["cap_remain"][h]
            occ[i] = ctx["occ"][h]
            occ_ratio[i] = (
                (ctx["occ"][h] + ctx["in_flight"][h]) / max(st.max_send[h], 1.0)
            )

        # 같은 class·mode의 적격 목적지 집합 안에서 ETA 순위와 상대 발송량을 계산한다.
        for c in (0, 1):
            for m in np.unique(mode):
                ii = np.flatnonzero(nonstay & (cls == c) & (mode == m))
                if not len(ii):
                    continue
                ev = eta_norm[ii]
                eta_rank[ii] = (ev[:, None] > ev[None, :]).sum(axis=1) / len(ii)
                pv = p_sent[ii]
                p_sent_rel[ii] = pv - float(pv.mean())

        X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        X[:, 0] = cls == 0
        X[:, 1] = cls == 1
        X[:, 2] = mode == 1
        X[:, 3] = dest == 0
        X[:, 4] = eta_norm
        X[:, 5] = eta_rank
        X[:, 6] = eta_raw
        X[:, 7] = uav_adv
        X[:, 8] = is_tier3
        X[:, 9] = helipad
        X[:, 10] = max_send
        X[:, 11] = len(actions)
        X[:, 12] = pa[1]          # Red 현장대기
        X[:, 13] = pa[6]          # Yellow 현장대기
        X[:, 14] = amb[0]
        X[:, 15] = uav[0]
        X[:, 16] = float(env.ev_manager.time)
        X[:, 17] = pa[0]
        X[:, 18] = pa[5]
        X[:, 19] = pa[2]
        X[:, 20] = pa[7]
        X[:, 21] = p_sent
        X[:, 22] = p_sent_rel
        X[:, 23] = in_flight
        X[:, 24] = float(np.sum(ctx["p_sent"]))
        X[:, 25] = float(np.sum(ctx["in_flight"]))
        X[:, 26] = amb[1]
        X[:, 27] = amb[2]
        X[:, 28] = amb[3]
        X[:, 29] = uav[1]
        X[:, 30] = uav[2]
        X[:, 31] = uav[3]
        X[:, 32] = amb[4] + uav[4]
        X[:, 33] = arrive
        X[:, 34] = cap_remain
        X[:, 35] = occ
        X[:, 36] = np.clip(occ_ratio, 0.0, 4.0)
        X[:, 37] = float(ctx["rho"])
        X[:, 38] = pa[3]
        X[:, 39] = pa[8]
        X[:, 40] = pa[4]
        X[:, 41] = pa[9]
        X[:, 42] = float(np.sum(ctx["cap_remain"]))
        return actions, X


def load_tree_package(path: str):
    with open(path, "rb") as f:
        package = pickle.load(f)
    required = {"tree", "info_level", "feature_indices", "estimator_kind"}
    missing = required - set(package)
    if missing:
        raise ValueError(f"트리 패키지 필드 누락 {sorted(missing)}: {path}")
    # LightGBM 적합 시 사용한 n_jobs가 pickle에 남으면 다중 평가 worker마다 내부
    # 스레드를 다시 만들어 서버를 과구독한다. 추론은 worker당 단일 스레드로 고정한다.
    estimator = package["tree"]
    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        try:
            if "n_jobs" in estimator.get_params(deep=False):
                estimator.set_params(n_jobs=1)
        except Exception:
            pass
    return package


def tree_scores(package: dict, X_full: np.ndarray) -> np.ndarray:
    X = X_full[:, np.asarray(package["feature_indices"], dtype=int)]
    tree = package["tree"]
    if package["estimator_kind"] == "regressor":
        return np.asarray(tree.predict(X), dtype=float)
    proba = tree.predict_proba(X)
    classes = list(tree.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def make_rank_tree_policy(package: dict, h_pad: int = 47):
    """evaluate 정책 규약 ``fn(obs, mask, env_unwrapped)->action``."""
    # v17 증강 스키마는 68열 특징을 쓴다. 키가 없는 기존 패키지는 아래 원 경로로
    # 내려가므로 구 트리의 동작은 비트동일하다.
    if package.get("feature_schema") == "v17_aug68":
        from v17_tree_features import make_aug_rank_tree_policy

        return make_aug_rank_tree_policy(package, h_pad=h_pad)
    builder = ActionFeatureBuilder(h_pad=h_pad)

    def fn(obs, mask, env_unwrapped):
        actions, X = builder.build(env_unwrapped, mask)
        score = tree_scores(package, X)
        best = np.flatnonzero(np.isclose(score, score.max(), rtol=0.0, atol=1e-12))
        if len(best) > 1:
            # 같은 잎 점수 동률은 "비대기 우선 → ETA순위 → action id"의 공개 규칙으로 해소.
            stay = X[best, FEATURE_NAMES.index("is_stay")]
            eta = X[best, FEATURE_NAMES.index("eta_rank")]
            order = np.lexsort((actions[best], eta, stay))
            j = int(best[order[0]])
        else:
            j = int(best[0])
        return int(actions[j])

    return fn
