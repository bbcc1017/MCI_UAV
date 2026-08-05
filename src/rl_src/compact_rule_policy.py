# -*- coding: utf-8 -*-
"""v13 교사 로그에서 추출한 축별 규칙을 후보행 점수로 합성한다."""
from __future__ import annotations

import numpy as np


class CompactRuleEstimator:
    """class tree × mode tree × 짧은 병원점수의 곱셈형 정책.

    ``tree_distill_policy.make_rank_tree_policy``가 요구하는 sklearn 분류기
    인터페이스만 구현한다. 실제 정책 순위는 log-score의 단조변환으로 정한다.
    """

    def __init__(
        self,
        *,
        feature_names: list[str],
        class_tree=None,
        class_features: list[str] | None = None,
        mode_tree=None,
        mode_features: list[str] | None = None,
        hospital_coef: dict[str, float] | None = None,
        stay_penalty: float = 25.0,
    ):
        self.feature_names = list(feature_names)
        self.class_tree = class_tree
        self.class_features = list(class_features or [])
        self.mode_tree = mode_tree
        self.mode_features = list(mode_features or [])
        self.hospital_coef = dict(hospital_coef or {})
        self.stay_penalty = float(stay_penalty)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self._index = {x: i for i, x in enumerate(self.feature_names)}

    @staticmethod
    def _positive_probability(model, X: np.ndarray) -> np.ndarray:
        prob = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(X), dtype=float)
        return np.clip(prob[:, classes.index(1)], 1e-6, 1.0 - 1e-6)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        score = np.zeros(len(X), dtype=float)
        is_red = X[:, self._index["is_red"]] > 0.5
        is_uav = X[:, self._index["is_uav"]] > 0.5
        is_stay = X[:, self._index["is_stay"]] > 0.5

        if self.class_tree is not None:
            idx = [self._index[x] for x in self.class_features]
            p_red = self._positive_probability(self.class_tree, X[:, idx])
            score += np.where(is_red, np.log(p_red), np.log1p(-p_red))
        if self.mode_tree is not None:
            idx = [self._index[x] for x in self.mode_features]
            p_uav = self._positive_probability(self.mode_tree, X[:, idx])
            score += np.where(is_uav, np.log(p_uav), np.log1p(-p_uav))
        for feature, coef in self.hospital_coef.items():
            score += float(coef) * X[:, self._index[feature]]
        score -= self.stay_penalty * is_stay.astype(float)
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        score = np.clip(self.decision_score(X), -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p, p])

