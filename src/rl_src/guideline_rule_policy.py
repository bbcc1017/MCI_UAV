# -*- coding: utf-8 -*-
"""PPO/최종교사 사후규칙을 명시적 휴리스틱으로 합성한다.

행동을 세 축으로 분리하되 후보 유효성은 기존 action mask를 그대로 따른다.

1. 환자등급: 깊이 3 class tree 또는 Red-first
2. 이송수단: 깊이 3 mode tree 또는 UAV 시간절감 임계값
3. 병원: LB-T 발송상한 또는 ETA+점유비 조건부 점수

``tree_distill_policy.make_rank_tree_policy``의 estimator 계약을 구현하므로 기존
대표점 closed-loop 평가기를 그대로 재사용할 수 있다.
"""
from __future__ import annotations

import numpy as np


class GuidelineRuleEstimator:
    """상태 내 후보행 전체를 받아 축별 규칙을 사전순위 방식으로 결합."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        ppo_class_tree=None,
        final_class_tree=None,
        class_features: list[str] | None = None,
        ppo_mode_tree=None,
        final_mode_tree=None,
        mode_features: list[str] | None = None,
        use_final: bool = False,
        correction_gate_feature: str | None = None,
        correction_gate_threshold: float | None = None,
        red_first: bool = False,
        mode_uav_threshold_min: float | None = None,
        hospital_strategy: str = "lb_t3",
        hospital_T: float = 3.0,
        hospital_coef: dict[str, float] | None = None,
    ):
        self.feature_names = list(feature_names)
        self.ppo_class_tree = ppo_class_tree
        self.final_class_tree = final_class_tree
        self.class_features = list(class_features or [])
        self.ppo_mode_tree = ppo_mode_tree
        self.final_mode_tree = final_mode_tree
        self.mode_features = list(mode_features or [])
        self.use_final = bool(use_final)
        self.correction_gate_feature = correction_gate_feature
        self.correction_gate_threshold = correction_gate_threshold
        self.red_first = bool(red_first)
        self.mode_uav_threshold_min = mode_uav_threshold_min
        self.hospital_strategy = str(hospital_strategy)
        self.hospital_T = float(hospital_T)
        self.hospital_coef = dict(hospital_coef or {})
        self.classes_ = np.asarray([0, 1], dtype=int)
        self._index = {x: i for i, x in enumerate(self.feature_names)}

    @staticmethod
    def _positive_probability(model, X: np.ndarray) -> np.ndarray:
        prob = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        return np.clip(prob[:, classes.index(1)], 1e-6, 1.0 - 1e-6)

    def _final_active(self, X: np.ndarray) -> bool:
        if self.use_final:
            return True
        if self.correction_gate_feature is None:
            return False
        value = float(X[0, self._index[self.correction_gate_feature]])
        return value > float(self.correction_gate_threshold)

    def _class_log_scores(self, X: np.ndarray, final_active: bool) -> dict[int, float]:
        if self.red_first:
            return {0: 0.0, 1: -20.0}
        model = self.final_class_tree if final_active else self.ppo_class_tree
        if model is None:
            return {0: 0.0, 1: 0.0}
        idx = [self._index[x] for x in self.class_features]
        p_red = float(self._positive_probability(model, X[:1, idx])[0])
        return {0: float(np.log(p_red)), 1: float(np.log1p(-p_red))}

    def _hospital_winner(self, X: np.ndarray, indices: np.ndarray) -> tuple[int, np.ndarray]:
        eta_rank = X[indices, self._index["eta_rank"]]
        if self.hospital_strategy.startswith("lb"):
            p_sent = X[indices, self._index["cand_p_sent"]]
            under = p_sent < self.hospital_T
            if np.any(under):
                local = np.flatnonzero(under)
                winner_local = int(local[np.argmin(eta_rank[local])])
                preference = np.where(under, -eta_rank, -100.0)
            else:
                # LB 원형과 동일: 전부 상한 이상이면 최소 발송, 동률은 가까운 병원.
                order = np.lexsort((eta_rank, p_sent))
                winner_local = int(order[0])
                preference = -(p_sent - float(p_sent.min())) - 1e-3 * eta_rank
        elif self.hospital_strategy == "eta_occ":
            preference = np.zeros(len(indices), dtype=float)
            for feature, coef in self.hospital_coef.items():
                preference += float(coef) * X[indices, self._index[feature]]
            winner_local = int(np.argmax(preference))
        else:
            raise ValueError(f"알 수 없는 병원 규칙: {self.hospital_strategy}")
        preference = preference - float(preference[winner_local])
        return int(indices[winner_local]), preference

    def _mode_log_score(
        self, X: np.ndarray, winner: int, mode: int, final_active: bool,
    ) -> float:
        if self.mode_uav_threshold_min is not None:
            advantage = float(X[winner, self._index["uav_advantage_min"]])
            choose_uav = advantage > float(self.mode_uav_threshold_min)
            return 0.0 if bool(mode) == choose_uav else -20.0
        model = self.final_mode_tree if final_active else self.ppo_mode_tree
        if model is None:
            return 0.0
        idx = [self._index[x] for x in self.mode_features]
        p_uav = float(self._positive_probability(model, X[winner:winner + 1, idx])[0])
        return float(np.log(p_uav) if mode == 1 else np.log1p(-p_uav))

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        score = np.full(len(X), -1e6, dtype=float)
        is_red = X[:, self._index["is_red"]] > 0.5
        is_uav = X[:, self._index["is_uav"]] > 0.5
        is_stay = X[:, self._index["is_stay"]] > 0.5
        nonstay = ~is_stay
        if not np.any(nonstay):
            return np.zeros(len(X), dtype=float)

        final_active = self._final_active(X)
        class_scores = self._class_log_scores(X, final_active)
        for patient_class in (0, 1):
            class_mask = is_red if patient_class == 0 else ~is_red
            for mode in (0, 1):
                group = np.flatnonzero(nonstay & class_mask & (is_uav == bool(mode)))
                if not len(group):
                    continue
                winner, hospital_pref = self._hospital_winner(X, group)
                mode_score = self._mode_log_score(X, winner, mode, final_active)
                score[group] = class_scores[patient_class] + mode_score + hospital_pref
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        score = np.clip(self.decision_score(X), -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p, p])
