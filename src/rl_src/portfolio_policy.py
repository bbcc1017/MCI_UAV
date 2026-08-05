# -*- coding: utf-8 -*-
"""v15 정책 포트폴리오의 후보 제안기.

PPO/NCRP의 후보집합에 우수 증류 정책과 MILP가 제안한 *행동*만 추가한다.
서로 다른 정책의 점수나 확률은 척도가 달라 평균하지 않고, 추가 행동은 NCRP가
동일한 비천리안 CRN 미래에서 직접 재평가한다. LB-T는 이 모듈의 입력이 아니다.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from tree_distill_policy import (
    FEATURE_NAMES,
    ActionFeatureBuilder,
    load_tree_package,
    tree_scores,
)


def _best_action(actions: np.ndarray, X: np.ndarray, score: np.ndarray) -> int:
    best = np.flatnonzero(np.isclose(score, score.max(), rtol=0.0, atol=1e-12))
    if len(best) > 1:
        stay = X[best, FEATURE_NAMES.index("is_stay")]
        eta = X[best, FEATURE_NAMES.index("eta_rank")]
        order = np.lexsort((actions[best], eta, stay))
        j = int(best[order[0]])
    else:
        j = int(best[0])
    return int(actions[j])


class TreeCandidateProposer:
    """여러 증류 패키지를 한 번의 특징 생성으로 공동 추론한다."""

    def __init__(self, packages: dict[str, dict] | list[str], h_pad: int = 47):
        if isinstance(packages, dict):
            self.packages = dict(packages)
        else:
            self.packages = {
                Path(path).stem: load_tree_package(str(path)) for path in packages
            }
        if not self.packages:
            raise ValueError("증류 패키지가 0개")
        self.builder = ActionFeatureBuilder(h_pad=h_pad)
        self.last_actions: dict[str, int] = {}

    def propose(self, env_unwrapped, mask) -> list[int]:
        actions, X = self.builder.build(env_unwrapped, np.asarray(mask, dtype=bool))
        selected: dict[str, int] = {}
        for name, package in self.packages.items():
            selected[name] = _best_action(actions, X, tree_scores(package, X))
        self.last_actions = selected
        return list(dict.fromkeys(selected.values()))

    def action_fn(self, env_unwrapped, mask, obs=None) -> int:
        return int(self.propose(env_unwrapped, mask)[0])


class CompositeCandidateProposer:
    """이름 붙은 제안기들의 유효 행동을 순서보존 합집합으로 반환."""

    def __init__(self, sources: list[tuple[str, object]]):
        self.sources = list(sources)
        self.last_sources: dict[int, tuple[str, ...]] = {}

    def propose(self, env_unwrapped, mask) -> list[int]:
        by_action: dict[int, list[str]] = defaultdict(list)
        out: list[int] = []
        for source_name, proposer in self.sources:
            actions = proposer.propose(env_unwrapped, mask)
            detail = getattr(proposer, "last_actions", None)
            if detail:
                for child_name, action in detail.items():
                    label = f"{source_name}:{child_name}"
                    by_action[int(action)].append(label)
                    if int(action) not in out:
                        out.append(int(action))
            else:
                for j, action in enumerate(actions):
                    label = source_name if len(actions) == 1 else f"{source_name}:{j}"
                    by_action[int(action)].append(label)
                    if int(action) not in out:
                        out.append(int(action))
        self.last_sources = {a: tuple(names) for a, names in by_action.items()}
        return out
