# -*- coding: utf-8 -*-
"""현장용 이송 배정 카드를 후보랭킹 정책으로 구현한다.

v14 ``guideline_rule_policy``는 축별 결정트리 객체를 그대로 정책에 심었기 때문에
임계값이 소수점(예: UAV 12.242119789123535)이고 사람이 카드에 옮겨 쓸 수 없었다.
이 모듈은 같은 세 축(등급·병원·수단)을 **정수 임계값만 쓰는 명시적 규칙**으로 다시
쓴다. 임계값은 SOTA(V15_BASE_G1) 결정 로그의 정수 격자 분석에서 얻는다.

카드 구조(고정):
  1단계 등급  현장에 적색·황색이 모두 있을 때만 선택 문제가 성립한다.
              ``class_mode`` 로 지정된 규칙으로 어느 등급을 먼저 보낼지 정한다.
  2단계 병원  적격(마스크 통과) 병원 중 이미 ``hospital_T`` 명을 보낸 곳은 건너뛰고
              가장 가까운 곳. 전부 찼으면 정원 조건을 무시하고 최근접.
  3단계 수단  두 수단이 모두 가능하면 UAV 시간절감이 ``uav_min_gain`` 분을 넘을 때만
              UAV. 아니면 가용한 수단.

``tree_distill_policy.make_rank_tree_policy`` 의 estimator 계약(``predict_proba`` +
``classes_``)을 구현하므로 기존 폐루프 평가기를 그대로 재사용한다. 점수는 사전식
우선순위를 **순위 정규화**로 표현한다. 시그모이드를 쓰지 않는 이유는 스케일 분리된
사전식 점수가 시그모이드에서 포화해 ``atol=1e-12`` 동률 판정에 삼켜지기 때문이다.
"""
from __future__ import annotations

import numpy as np

# 등급 규칙 종류
CLASS_MODES = ("yellow_first", "red_first", "gated", "amb_only", "yellow_low_only")


class CardRuleEstimator:
    """현장 카드 = (등급 규칙, 병원 정원, 수단 임계값) 정수 3종."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        class_mode: str = "gated",
        yellow_high: int = 8,
        yellow_low: int = 3,
        hospital_T: int = 3,
        uav_min_gain: int = 6,
    ):
        if class_mode not in CLASS_MODES:
            raise ValueError(f"class_mode 오류: {class_mode} (허용 {CLASS_MODES})")
        self.feature_names = list(feature_names)
        self.class_mode = str(class_mode)
        self.yellow_high = int(yellow_high)
        self.yellow_low = int(yellow_low)
        self.hospital_T = int(hospital_T)
        self.uav_min_gain = int(uav_min_gain)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self._i = {n: k for k, n in enumerate(self.feature_names)}

    # ------------------------------------------------------------------ 등급
    def _prefer_red(self, red_at_site: float, yellow_at_site: float,
                    amb_available: float) -> bool | None:
        """적색을 먼저 보낼지. None 이면 선택 문제가 성립하지 않음(한쪽만 존재)."""
        if red_at_site <= 0 or yellow_at_site <= 0:
            return None
        no_amb = amb_available <= 0
        if self.class_mode == "yellow_first":
            return False
        if self.class_mode == "red_first":
            return True
        if self.class_mode == "amb_only":
            # 구급차가 없으면(=UAV만 남으면) 적색. UAV 착륙 병원이 곧 적색 수용 병원이다.
            return no_amb
        if self.class_mode == "yellow_low_only":
            return yellow_at_site <= self.yellow_low
        # gated: 황색 적체가 크면 황색, 황색이 적고 구급차도 없으면 적색
        if yellow_at_site >= self.yellow_high:
            return False
        if yellow_at_site <= self.yellow_low and no_amb:
            return True
        return False

    # ------------------------------------------------------------- 점수 산출
    def decision_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        i = self._i
        n = len(X)
        if n == 0:
            return np.zeros(0, dtype=float)

        is_red = X[:, i["is_red"]] > 0.5
        is_uav = X[:, i["is_uav"]] > 0.5
        is_stay = X[:, i["is_stay"]] > 0.5
        eta_rank = X[:, i["eta_rank"]]
        uav_gain = X[:, i["uav_advantage_min"]]
        cand_p_sent = X[:, i["cand_p_sent"]]
        # 상태 특징은 한 결정 안에서 모든 후보행에 동일하다.
        red_at_site = float(X[0, i["red_at_site"]])
        yellow_at_site = float(X[0, i["yellow_at_site"]])
        amb_available = float(X[0, i["amb_available"]])
        uav_available = float(X[0, i["uav_available"]])

        # 1단계 등급
        prefer_red = self._prefer_red(red_at_site, yellow_at_site, amb_available)
        if prefer_red is None:
            class_bonus = np.zeros(n)
        else:
            class_bonus = np.where(is_red == prefer_red, 1.0, 0.0)

        # 2단계 병원 정원. 모든 적격 후보가 정원 초과면 조건을 무시한다.
        cap_ok = cand_p_sent < self.hospital_T
        movable = ~is_stay
        if movable.any() and not cap_ok[movable].any():
            cap_bonus = np.zeros(n)
        else:
            cap_bonus = np.where(cap_ok, 1.0, 0.0)

        # 3단계 수단. 두 수단이 모두 가능할 때만 임계값이 의미를 갖는다.
        if amb_available > 0 and uav_available > 0:
            want_uav = uav_gain > float(self.uav_min_gain)
            mode_bonus = np.where(is_uav == want_uav, 1.0, 0.0)
        else:
            mode_bonus = np.zeros(n)

        # 사전식: 비대기 > 등급 > 병원정원 > 수단 > ETA순위
        return (
            (~is_stay) * 1e5
            + class_bonus * 1e4
            + cap_bonus * 1e3
            + mode_bonus * 1e2
            - eta_rank
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.decision_score(X)
        n = len(raw)
        if n == 0:
            return np.zeros((0, 2), dtype=float)
        # 순위 정규화: 동률은 같은 값으로 남겨 상위 tie-break 규칙에 넘긴다.
        order = np.argsort(raw, kind="mergesort")
        rank = np.empty(n, dtype=float)
        r = 0
        k = 0
        while k < n:
            j = k
            while j + 1 < n and raw[order[j + 1]] == raw[order[k]]:
                j += 1
            rank[order[k:j + 1]] = r
            r += 1
            k = j + 1
        p = 0.01 + 0.98 * (rank / max(r - 1, 1))
        return np.column_stack([1.0 - p, p])

    def describe(self) -> str:
        """카드 문구(현장 배포용)."""
        cls = {
            "yellow_first": "황색을 먼저 보낸다",
            "red_first": "적색을 먼저 보낸다",
            "amb_only": "대기 구급차가 없으면 적색, 있으면 황색",
            "yellow_low_only": f"현장 황색이 {self.yellow_low}명 이하면 적색, 그 밖에는 황색",
            "gated": (f"현장 황색이 {self.yellow_high}명 이상이면 황색. "
                      f"황색이 {self.yellow_low}명 이하이고 대기 구급차가 없으면 적색. "
                      f"그 밖에는 황색"),
        }[self.class_mode]
        return (f"1단계 등급: {cls}\n"
                f"2단계 병원: 이미 {self.hospital_T}명 보낸 병원은 건너뛰고 가장 가까운 곳\n"
                f"3단계 수단: 두 수단이 모두 가능하면 UAV가 {self.uav_min_gain}분 이상 "
                f"빠를 때만 UAV")
