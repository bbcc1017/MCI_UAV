"""Shin Threshold/2Step의 환자선택은 보존하고 병원선택만 HEUR64와 정렬한다.

기존 :class:`ShinHeuristicRule`과 결과 이름을 섞지 않기 위한 별도 정책군이다.
Threshold/2Step의 class 산식과 공통 mode 우선순위는 그대로 두고, 논문의 Yellow
Tier2/Tier3 0.5 선택만 다음 두 규칙으로 교체한다.

* RedOnly: Red→최근접 Tier3, Yellow→최근접 Tier2
* YellowNearest: Red→최근접 Tier3, Yellow→tier 무관 최근접

PIH/Integrated는 병원점수 자체가 방법의 핵심이므로 이 클래스에서 허용하지 않는다.
"""
from __future__ import annotations

from ShinHeuristics import SHIN_MODE_RULES, ShinHeuristicRule


SHIN_ALIGNED_METHODS = ("Threshold", "2Step")
SHIN_ALIGNED_HOSPITAL_RULES = ("RedOnly", "YellowNearest")


class ShinHospitalAlignedRule(ShinHeuristicRule):
    def __init__(self, method: str, hospital_rule: str, mode_rule: str):
        if method not in SHIN_ALIGNED_METHODS:
            raise ValueError(f"병원정렬 대상이 아닌 Shin 방법: {method}")
        if hospital_rule not in SHIN_ALIGNED_HOSPITAL_RULES:
            raise ValueError(f"지원하지 않는 병원규칙: {hospital_rule}")
        if mode_rule not in SHIN_MODE_RULES:
            raise ValueError(f"지원하지 않는 mode 규칙: {mode_rule}")
        super().__init__(method, mode_rule)
        self.hospital_rule = hospital_rule
        self.rule_name = (
            f"ShinAlignHOS {method}, {hospital_rule}, Mode {mode_rule}"
        )

    def _simple_destination(self, obs, mode: int, p_class: int):
        if p_class == 0:
            return self._nearest(obs, mode, 0, tier=3)
        if self.hospital_rule == "RedOnly":
            return self._nearest(obs, mode, 1, tier=2)
        return self._nearest(obs, mode, 1, tier=None)

