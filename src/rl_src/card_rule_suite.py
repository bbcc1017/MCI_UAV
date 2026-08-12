# -*- coding: utf-8 -*-
"""현장 카드 변형을 폐루프 평가용 패키지로 굽는다.

임계값은 SOTA(V15_BASE_G1) 결정 로그의 정수 격자 분석에서 얻은 값을 기본으로 하고,
그 주변 대조군을 함께 만들어 **폐루프 PDR로 고른다**(모방 충실도로 고르지 않는다).
v10 트리 평가기(``v10_tree_eval.py``)가 요구하는 패키지 계약을 그대로 따른다.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_rule_policy import CardRuleEstimator  # noqa: E402
from tree_distill_policy import FEATURE_NAMES  # noqa: E402

# (complexity, class_mode, yellow_high, yellow_low, hospital_T, uav_min_gain, 메모)
SPECS = [
    ("G_T3_U6", "gated", 8, 3, 3, 6, "로그 기본: 황색8/3 게이트·정원3·UAV6분"),
    ("Y_T3_U6", "yellow_first", 8, 3, 3, 6, "등급 대조: 항상 황색 먼저"),
    ("A_T3_U6", "amb_only", 8, 3, 3, 6, "등급 대조: 구급차 유무만으로 결정"),
    ("L_T3_U6", "yellow_low_only", 8, 3, 3, 6, "등급 대조: 황색 3명 이하면 적색"),
    ("R_T3_U6", "red_first", 8, 3, 3, 6, "등급 대조: 적색 먼저(구 관행)"),
    ("G_T2_U6", "gated", 8, 3, 2, 6, "정원 대조: 병원당 2명"),
    ("G_T4_U6", "gated", 8, 3, 4, 6, "정원 대조: 병원당 4명"),
    ("G_T3_U12", "gated", 8, 3, 3, 12, "수단 대조: 구 카드 임계 12분"),
    ("G_T3_U0", "gated", 8, 3, 3, 0, "수단 대조: 가능하면 항상 UAV"),
    ("G6_T3_U6", "gated", 6, 3, 3, 6, "게이트 대조: 황색 상한 6명"),
    ("G12_T3_U6", "gated", 12, 3, 3, 6, "게이트 대조: 황색 상한 12명"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"기존 카드 정책 보호: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for complexity, mode, yhi, ylo, T, gain, memo in SPECS:
        est = CardRuleEstimator(
            feature_names=FEATURE_NAMES, class_mode=mode,
            yellow_high=yhi, yellow_low=ylo, hospital_T=T, uav_min_gain=gain,
        )
        package = {
            "schema_version": 1,
            "tree": est,
            "estimator_kind": "classifier",
            "objective": "field_card_integer_rule",
            "teacher": "V15_BASE_G1_decision_log_p3_seed9200_9202",
            "info_level": "CARD",
            "info_label": "현장 카드",
            "feature_indices": list(range(len(FEATURE_NAMES))),
            "feature_names": list(FEATURE_NAMES),
            "complexity": complexity,
            "complexity_spec": dict(class_mode=mode, yellow_high=yhi, yellow_low=ylo,
                                    hospital_T=T, uav_min_gain=gain),
            "fit_protocol": "정수 임계값은 p3 개발 fold 로그에서 격자 탐색; 대표점250 미개봉",
            "memo": memo,
        }
        with (out / f"CARD_{complexity}.pkl").open("wb") as f:
            pickle.dump(package, f)
        rows.append({"policy": f"CARD_{complexity}", "info_level": "CARD",
                     "complexity": complexity, "class_mode": mode,
                     "yellow_high": yhi, "yellow_low": ylo,
                     "hospital_T": T, "uav_min_gain": gain, "memo": memo})
        print(f"[card] CARD_{complexity}  {memo}")

    with (out / "fit_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)}개 카드 → {out}")


if __name__ == "__main__":
    main()
