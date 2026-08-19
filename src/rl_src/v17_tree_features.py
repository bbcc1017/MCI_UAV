# -*- coding: utf-8 -*-
"""v17 논문형 특징 증강: 후보 간 순위·구간·상대량·상태요약.

Yan et al. (2026, TRE 213:104981) §4.5 / Table 1 은 대리 결정나무에 원 상태특징
외에 네 유형을 덧붙인다. "These features summarize relative ranks and coarse
distance categories that trees exploit effectively."

* Rank        : 위치(우리는 후보) 사이의 상대순위
* Categorical : 거리를 굵은 구간으로 범주화
* Numerical   : 원 수치
* 범위        : 위치별 / DC별 / 글로벌

우리 트리는 논문과 달리 후보랭킹 구조이므로 "위치 간 순위"를 "state 안의 유효
``[class, dest, mode]`` 후보 간 순위"로 옮긴다. 25개 전부가 기존 43개 특징의
**같은 state 안에서의 변환**이라, 이미 수집한 교사 데이터셋(npz)에서 재계산할 수
있고 시뮬레이터 재수집이 필요 없다. 추론 경로와 데이터셋 경로는 동일한
``augment_state`` 를 호출하므로 정의가 갈라질 수 없다.

대기(``dest == 0``) 후보는 병원이 없어 순위·구간·상대량이 정의되지 않는다. 전부
sentinel ``-1`` 로 두고, 기존 ``is_stay`` 특징이 분기를 담당한다.

정보단계는 기존 I0~I3 규율을 승계한다. 새 특징도 계산에 쓰인 원 특징이 속한
최저 단계에만 배정한다(예: 점유순위는 I3, 발송순위는 I1).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tree_distill_policy import (  # noqa: E402
    FEATURE_NAMES,
    ActionFeatureBuilder,
)

_IX = {name: i for i, name in enumerate(FEATURE_NAMES)}

STAY_SENTINEL = -1.0

# (이름, 유형, 최저 정보단계)
AUG_SPEC: list[tuple[str, str, str]] = [
    # --- I0: 정적 지리·인프라 + 현장 최소정보만으로 계산 ---
    ("rank_eta_all",      "RANK",   "I0_MINIMAL"),
    ("is_eta_min_cm",     "RANK",   "I0_MINIMAL"),
    ("is_eta_top3_cm",    "RANK",   "I0_MINIMAL"),
    ("eta_ratio_best_cm", "REL",    "I0_MINIMAL"),
    ("eta_gap_best_cm",   "REL",    "I0_MINIMAL"),
    ("eta_bin",           "CAT",    "I0_MINIMAL"),
    ("eta_gap_bin",       "CAT",    "I0_MINIMAL"),
    ("rank_uav_adv",      "RANK",   "I0_MINIMAL"),
    ("uav_adv_bin",       "CAT",    "I0_MINIMAL"),
    ("eta_spread_cm",     "GLOBAL", "I0_MINIMAL"),
    ("n_tier3_cand",      "GLOBAL", "I0_MINIMAL"),
    ("n_helipad_cand",    "GLOBAL", "I0_MINIMAL"),
    ("site_load_ratio",   "GLOBAL", "I0_MINIMAL"),
    ("red_share_site",    "GLOBAL", "I0_MINIMAL"),
    ("is_class_majority", "GLOBAL", "I0_MINIMAL"),
    # --- I1: 현장 구조·발송 기록 ---
    ("rank_p_sent",       "RANK",   "I1_FIELD"),
    ("p_sent_bin",        "CAT",    "I1_FIELD"),
    ("is_p_sent_min",     "RANK",   "I1_FIELD"),
    ("total_sent_ratio",  "REL",    "I1_FIELD"),
    ("rank_load_i1",      "RANK",   "I1_FIELD"),
    # --- I2: 차량 텔레메트리 ---
    ("rank_arrive",       "RANK",   "I2_TELEMETRY"),
    ("arrive_bin",        "CAT",    "I2_TELEMETRY"),
    # --- I3: 병원 실시간 연계 ---
    ("rank_occ_ratio",    "RANK",   "I3_CONNECTED"),
    ("occ_bin",           "CAT",    "I3_CONNECTED"),
    ("rank_cap_remain",   "RANK",   "I3_CONNECTED"),
]

AUG_NAMES = [name for name, _, _ in AUG_SPEC]
AUG_FAMILY = {name: fam for name, fam, _ in AUG_SPEC}
AUG_LEVEL = {name: lvl for name, _, lvl in AUG_SPEC}
ALL_FEATURE_NAMES = list(FEATURE_NAMES) + AUG_NAMES
FEATURE_SCHEMA = "v17_aug68"

_AIX = {name: len(FEATURE_NAMES) + i for i, name in enumerate(AUG_NAMES)}
# 증강행렬 A 안에서의 국소 열 위치(_AIX 는 68열 전체행렬 기준).
_ALOC = {name: i for i, name in enumerate(AUG_NAMES)}

# 굵은 구간 경계(논문의 categorical distance 대응). 단위는 분.
ETA_EDGES = np.asarray([5.0, 10.0, 20.0, 40.0], dtype=np.float32)
ETA_GAP_EDGES = np.asarray([1.0, 3.0, 6.0, 12.0], dtype=np.float32)
UAV_ADV_EDGES = np.asarray([0.0, 5.0, 15.0, 40.0], dtype=np.float32)
ARRIVE_EDGES = np.asarray([10.0, 30.0, 60.0, 120.0], dtype=np.float32)
P_SENT_EDGES = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
OCC_EDGES = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)

_CANONICAL_LEVELS = ["I0_MINIMAL", "I1_FIELD", "I2_TELEMETRY", "I3_CONNECTED"]
_BASE_LEVEL_END = {"I0_MINIMAL": 17, "I1_FIELD": 26, "I2_TELEMETRY": 34,
                   "I3_CONNECTED": len(FEATURE_NAMES)}


def _rank_asc(v: np.ndarray) -> np.ndarray:
    """0-기반 오름차순 순위(작을수록 0). 동률은 같은 순위."""
    return (v[:, None] > v[None, :]).sum(axis=1).astype(np.float32)


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """0-기반 내림차순 순위(클수록 0)."""
    return (v[:, None] < v[None, :]).sum(axis=1).astype(np.float32)


def _bin(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, v, side="right").astype(np.float32)


def augment_state(X: np.ndarray) -> np.ndarray:
    """한 state의 (n_cand, 43) 특징행렬 → (n_cand, 25) 증강행렬."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    A = np.full((n, len(AUG_NAMES)), STAY_SENTINEL, dtype=np.float32)

    is_stay = X[:, _IX["is_stay"]] > 0.5
    live = ~is_stay
    is_red = X[:, _IX["is_red"]] > 0.5
    is_uav = X[:, _IX["is_uav"]] > 0.5
    eta_norm = X[:, _IX["eta_norm"]]
    eta_raw = X[:, _IX["eta_raw_min"]]
    uav_adv = X[:, _IX["uav_advantage_min"]]
    p_sent = X[:, _IX["cand_p_sent"]]
    in_flight = X[:, _IX["cand_in_flight"]]
    arrive = X[:, _IX["cand_arrive_min"]]
    occ_ratio = X[:, _IX["cand_occ_ratio"]]
    cap_remain = X[:, _IX["cand_cap_remain"]]

    def put(name, idx, values):
        A[idx, _ALOC[name]] = values

    li = np.flatnonzero(live)
    if li.size:
        # 전 후보 공통 순위(수단 혼합). UAV 가 훨씬 빠르므로 수단 간 비교가 의미를 갖는다.
        put("rank_eta_all", li, _rank_asc(eta_norm[li]))
        put("rank_uav_adv", li, _rank_desc(uav_adv[li]))
        put("rank_p_sent", li, _rank_asc(p_sent[li]))
        put("rank_load_i1", li, _rank_asc(p_sent[li] + in_flight[li]))
        put("rank_arrive", li, _rank_asc(arrive[li]))
        put("rank_occ_ratio", li, _rank_asc(occ_ratio[li]))
        put("rank_cap_remain", li, _rank_desc(cap_remain[li]))
        put("is_p_sent_min", li, (p_sent[li] <= p_sent[li].min()).astype(np.float32))

        put("eta_bin", li, _bin(eta_raw[li], ETA_EDGES))
        put("uav_adv_bin", li, _bin(uav_adv[li], UAV_ADV_EDGES))
        put("arrive_bin", li, _bin(arrive[li], ARRIVE_EDGES))
        put("p_sent_bin", li, _bin(p_sent[li], P_SENT_EDGES))
        put("occ_bin", li, _bin(occ_ratio[li], OCC_EDGES))

        # (class, mode) 그룹 내 순위·상대량: "같은 등급·같은 수단에서 몇 번째로 가까운가"
        for c in (True, False):
            for m in (True, False):
                gi = np.flatnonzero(live & (is_red == c) & (is_uav == m))
                if not gi.size:
                    continue
                g_norm = eta_norm[gi]
                g_raw = eta_raw[gi]
                r = _rank_asc(g_norm)
                best = float(g_raw.min())
                put("is_eta_min_cm", gi, (r < 0.5).astype(np.float32))
                put("is_eta_top3_cm", gi, (r < 2.5).astype(np.float32))
                put("eta_gap_best_cm", gi, g_raw - best)
                put("eta_ratio_best_cm", gi, g_raw / max(best, 1e-3))
                put("eta_gap_bin", gi, _bin(g_raw - best, ETA_GAP_EDGES))
                put("eta_spread_cm", gi, np.full(gi.size, float(g_raw.max() - best), np.float32))

    # 상태 요약(모든 행 동일). 대기 후보도 상태 요약은 받는다.
    red_site = float(X[0, _IX["red_at_site"]])
    yellow_site = float(X[0, _IX["yellow_at_site"]])
    amb_av = float(X[0, _IX["amb_available"]])
    uav_av = float(X[0, _IX["uav_available"]])
    total_sent = float(X[0, _IX["total_p_sent"]])
    red_un = float(X[0, _IX["red_unrescued"]])
    yellow_un = float(X[0, _IX["yellow_unrescued"]])

    n_tier3 = float(np.count_nonzero(live & (X[:, _IX["is_tier3"]] > 0.5) & ~is_uav))
    n_heli = float(np.count_nonzero(live & (X[:, _IX["has_helipad"]] > 0.5) & is_uav))
    A[:, _ALOC["n_tier3_cand"]] = n_tier3
    A[:, _ALOC["n_helipad_cand"]] = n_heli
    A[:, _ALOC["site_load_ratio"]] = (red_site + yellow_site) / (amb_av + uav_av + 1.0)
    A[:, _ALOC["red_share_site"]] = red_site / (red_site + yellow_site + 1.0)
    A[:, _ALOC["total_sent_ratio"]] = total_sent / (red_un + yellow_un + total_sent + 1.0)
    major_red = red_site >= yellow_site
    A[:, _ALOC["is_class_majority"]] = (is_red == major_red).astype(np.float32)
    return A


def augment_dataset(X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """수집 npz 전체를 state 단위로 증강. 추론과 같은 함수를 재사용한다."""
    X = np.asarray(X, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    out = np.empty((X.shape[0], len(AUG_NAMES)), dtype=np.float32)
    for s in range(len(offsets) - 1):
        a, b = int(offsets[s]), int(offsets[s + 1])
        out[a:b] = augment_state(X[a:b])
    return out


def info_levels_v2() -> dict[str, list[int]]:
    """기존 누적 정보단계에 새 특징을 최저 단계 기준으로 편입."""
    order = {lvl: i for i, lvl in enumerate(_CANONICAL_LEVELS)}
    levels: dict[str, list[int]] = {}
    for lvl in _CANONICAL_LEVELS:
        idx = list(range(_BASE_LEVEL_END[lvl]))
        idx += [_AIX[n] for n in AUG_NAMES if order[AUG_LEVEL[n]] <= order[lvl]]
        levels[lvl] = sorted(idx)
    return levels


def family_indices(families: list[str], base: bool = True) -> list[int]:
    """BASE43 + 지정 유형만의 특징 인덱스(유형별 기여 분해용)."""
    idx = list(range(len(FEATURE_NAMES))) if base else []
    idx += [_AIX[n] for n in AUG_NAMES if AUG_FAMILY[n] in families]
    return sorted(idx)


class AugmentedFeatureBuilder:
    """추론 시 43개 물리특징 + 25개 증강특징을 함께 만든다."""

    def __init__(self, h_pad: int = 47):
        self.base = ActionFeatureBuilder(h_pad=h_pad)

    def build(self, env, mask) -> tuple[np.ndarray, np.ndarray]:
        actions, X = self.base.build(env, mask)
        return actions, np.hstack([X, augment_state(X)])


def make_aug_rank_tree_policy(package: dict, h_pad: int = 47):
    """증강 스키마 트리의 ``fn(obs, mask, env_unwrapped)->action`` 정책."""
    from tree_distill_policy import tree_scores

    builder = AugmentedFeatureBuilder(h_pad=h_pad)
    i_stay = ALL_FEATURE_NAMES.index("is_stay")
    i_eta = ALL_FEATURE_NAMES.index("eta_rank")

    def fn(obs, mask, env_unwrapped):
        actions, X = builder.build(env_unwrapped, mask)
        score = tree_scores(package, X)
        best = np.flatnonzero(np.isclose(score, score.max(), rtol=0.0, atol=1e-12))
        if len(best) > 1:
            # 기존 트리와 동일한 공개 동률규칙: 비대기 우선 → ETA순위 → action id
            order = np.lexsort((actions[best], X[best, i_eta], X[best, i_stay]))
            j = int(best[order[0]])
        else:
            j = int(best[0])
        return int(actions[j])

    return fn
