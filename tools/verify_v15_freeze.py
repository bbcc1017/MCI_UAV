# -*- coding: utf-8 -*-
"""v15 블라인드 평가 전에 동결된 정책·모델·데이터 해시를 검증한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FREEZE = REPO / "results/scoreboard/v15/final/selection_freeze.json"
ARTIFACTS = {
    "ppo_model_sha256": REPO / "results/rl/redesign/v10_random4_1000_pointer_s0/final_model.zip",
    "ppo_vecnormalize_sha256": REPO / "results/rl/redesign/v10_random4_1000_pointer_s0/vecnormalize.pkl",
    "gbdt_g1_full1000_sha256": REPO / "results/scoreboard/v13/sota_distill/students_full1000/I3_CONNECTED_GBDT_L63_BASE.pkl",
    "gbdt_g2_full1000_sha256": REPO / "results/scoreboard/v13/sota_distill/students_full1000/I3_CONNECTED_GBDT_L31_BASE.pkl",
    "gbdt_g3_full1000_sha256": REPO / "results/scoreboard/v13/sota_distill/students_full1000/I1_FIELD_GBDT_L63_BASE.pkl",
    "planner_policy_py_sha256": REPO / "src/rl_src/planner_policy.py",
    "portfolio_policy_py_sha256": REPO / "src/rl_src/portfolio_policy.py",
    "milp_policy_py_sha256": REPO / "src/rl_src/milp_policy.py",
    "tree_distill_policy_py_sha256": REPO / "src/rl_src/tree_distill_policy.py",
    "v15_portfolio_eval_py_sha256": REPO / "src/rl_src/v15_portfolio_eval.py",
}
ANALYSIS_ARTIFACTS = {
    "v15_portfolio_results_py_sha256": REPO / "tools/v15_portfolio_results.py",
    "v15_final_analysis_py_sha256": REPO / "tools/v15_final_analysis.py",
    "v15_region_profile_analysis_py_sha256": REPO / "tools/v15_region_profile_analysis.py",
    "v15_guideline_analysis_py_sha256": REPO / "tools/v15_guideline_analysis.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen.get("selected_policy") != "V15_BASE_G1":
        raise RuntimeError("동결 정책명이 V15_BASE_G1이 아님")
    if frozen.get("policy_definition", {}).get("lb_t_included") is not False:
        raise RuntimeError("LB-T3 배제 불변식 위반")
    expected = frozen.get("frozen_artifacts", {})
    errors = []
    for key, path in ARTIFACTS.items():
        if not path.is_file():
            errors.append(f"파일 없음: {path}")
            continue
        actual = sha256(path)
        if expected.get(key) != actual:
            errors.append(f"해시 불일치 {key}: {actual} != {expected.get(key)}")
    expected_analysis = frozen.get("analysis_artifacts", {})
    for key, path in ANALYSIS_ARTIFACTS.items():
        actual = sha256(path) if path.is_file() else None
        if expected_analysis.get(key) != actual:
            errors.append(f"분석코드 해시 불일치 {key}: {actual} != {expected_analysis.get(key)}")
    blind = frozen.get("future_blind_test", {})
    manifest = Path(blind.get("manifest", ""))
    if not manifest.is_file() or sha256(manifest) != blind.get("manifest_sha256"):
        errors.append("신규 블라인드 manifest 해시 불일치")
    analysis_plan = REPO / "results/scoreboard/v15/final/blind_analysis_plan.json"
    if not analysis_plan.is_file() or sha256(analysis_plan) != blind.get("analysis_plan_sha256"):
        errors.append("블라인드 분석계획 해시 불일치")
    if errors:
        raise RuntimeError("v15 동결 검증 실패\n" + "\n".join(errors))
    print(
        f"v15 동결 검증 PASS: {len(ARTIFACTS)}개 정책 자산 + "
        f"{len(ANALYSIS_ARTIFACTS)}개 분석코드 + blind manifest/분석계획, "
        "LB-T3 미포함"
    )


if __name__ == "__main__":
    main()
