# -*- coding: utf-8 -*-
"""PPO 기본규칙·최종교사 보정규칙의 탐색용 v14 정책 패키지.

주의: ``*_LBT3`` 결합은 UAV 운용을 위해 추가된 원거리 헬기장병원을 AMB
발송상한 풀에도 포함하는 구조라 본 연구의 가이드라인 주장에서는 철회했다.
기존 산출물 재현용으로만 보존하며 신규 규칙은 LB-T3와 결합하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src/rl_src"))

import v14_policy_rule_comparison as comparison
import v13_sota_rule_analysis as v13
from guideline_rule_policy import GuidelineRuleEstimator
from tree_distill_policy import FEATURE_NAMES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_tree(frame: pd.DataFrame, *, axis: str, seed: int):
    work = frame.copy()
    if axis == "class":
        features = v13.CLASS_FEATURES
        mask = work.both_class_available
        target = (work.selected_class == 0).astype(int)
    elif axis == "mode":
        features = v13.MODE_FEATURES
        mask = (work.selected_dest > 0) & work.both_mode_available
        target = (work.selected_mode == 1).astype(int)
    else:
        raise ValueError(axis)
    work = work.loc[mask].copy()
    y = target.loc[mask].to_numpy(int)
    count = work.groupby("region_base", observed=True).region_base.transform("size").to_numpy(float)
    weights = (1.0 / count)
    weights /= weights.mean()
    model = DecisionTreeClassifier(
        max_depth=3, max_leaf_nodes=8,
        min_samples_leaf=max(100, int(0.01 * len(work))),
        class_weight="balanced", random_state=seed,
    )
    model.fit(work[features].to_numpy(float), y, sample_weight=weights)
    return model, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=Path, default=comparison.DEFAULT_OUT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_withdrawn_lbt3", action="store_true",
        help="감사 재현 전용: 철회된 LB-T3 결합 5종도 생성",
    )
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"기존 가이드라인 정책 보호: {out}")
    out.mkdir(parents=True, exist_ok=True)

    raw, frames, quality = comparison.load_views()
    train = {
        policy: frames[(frames.policy == policy) & (frames.dataset == "train")].copy()
        for policy in ("PPO_ON_POLICY", "FINAL_TEACHER")
    }
    ppo_class, class_features = fit_tree(train["PPO_ON_POLICY"], axis="class", seed=20260803)
    ppo_mode, mode_features = fit_tree(train["PPO_ON_POLICY"], axis="mode", seed=20260804)
    final_class, _ = fit_tree(train["FINAL_TEACHER"], axis="class", seed=20260803)
    final_mode, _ = fit_tree(train["FINAL_TEACHER"], axis="mode", seed=20260804)

    coef = pd.read_csv(args.analysis_dir / "hospital_choice_coefficients.csv")
    hospital_coef = {}
    for policy in ("PPO_ON_POLICY", "FINAL_TEACHER"):
        x = coef[coef.policy == policy].set_index("feature").raw_unit_coef_train_all
        hospital_coef[policy] = {
            "eta_rank": float(x["eta_rank"]),
            "cand_occ_ratio": float(x["cand_occ_ratio"]),
        }

    common = dict(
        feature_names=FEATURE_NAMES,
        ppo_class_tree=ppo_class, final_class_tree=final_class,
        class_features=class_features,
        ppo_mode_tree=ppo_mode, final_mode_tree=final_mode,
        mode_features=mode_features,
    )
    specs = [
        ("PPO_ETA_OCC", dict(use_final=False, hospital_strategy="eta_occ", hospital_coef=hospital_coef["PPO_ON_POLICY"])),
        ("FINAL_ETA_OCC", dict(use_final=True, hospital_strategy="eta_occ", hospital_coef=hospital_coef["FINAL_TEACHER"])),
    ]
    if args.allow_withdrawn_lbt3:
        specs.extend([
            ("PPO_LBT3", dict(use_final=False, hospital_strategy="lb_t3", hospital_T=3.0)),
            ("FINAL_LBT3", dict(use_final=True, hospital_strategy="lb_t3", hospital_T=3.0)),
            ("GATED_LBT3", dict(
                correction_gate_feature="fleet_critical", correction_gate_threshold=12.5,
                hospital_strategy="lb_t3", hospital_T=3.0,
            )),
            ("EXPLICIT_LBT3", dict(
                use_final=True, mode_uav_threshold_min=12.242119789123535,
                hospital_strategy="lb_t3", hospital_T=3.0,
            )),
            ("REDFIRST_LBT3", dict(
                red_first=True, mode_uav_threshold_min=12.242119789123535,
                hospital_strategy="lb_t3", hospital_T=3.0,
            )),
        ])

    rows = []
    for complexity, knobs in specs:
        estimator = GuidelineRuleEstimator(**common, **knobs)
        package = {
            "schema_version": 1,
            "tree": estimator,
            "estimator_kind": "classifier",
            "objective": "posthoc_guideline_rule_policy",
            "teacher": "PPO_BASE_vs_PPO_NCRP_H20M16_MILPINJ",
            "info_level": "GUIDE",
            "info_label": "사후통계 가이드라인",
            "feature_indices": list(range(len(FEATURE_NAMES))),
            "feature_names": list(FEATURE_NAMES),
            "complexity": complexity,
            "complexity_spec": knobs,
            "fit_protocol": "p0-p2 only; p3 validation; eval250 untouched",
            "data_quality": quality["status"],
        }
        path = out / f"GUIDE_{complexity}.pkl"
        with path.open("wb") as f:
            pickle.dump(package, f)
        rows.append({
            "policy": f"GUIDE_{complexity}", "info_level": "GUIDE",
            "complexity": complexity, "family": "guideline_rule",
            "spec": json.dumps(knobs, ensure_ascii=False, sort_keys=True),
            "pkl": str(path), "pkl_sha256": sha256(path),
        })

    with (out / "fit_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for label, model, features in (
        ("ppo_class", ppo_class, class_features), ("final_class", final_class, class_features),
        ("ppo_mode", ppo_mode, mode_features), ("final_mode", final_mode, mode_features),
    ):
        (out / f"{label}_tree.txt").write_text(
            export_text(model, feature_names=features, decimals=2), encoding="utf-8",
        )
    (out / "meta.json").write_text(json.dumps({
        "schema_version": 1,
        "source_hashes": quality["source_hashes"],
        "analysis_dir": str(args.analysis_dir.resolve()),
        "analysis_hash": sha256(args.analysis_dir / "hospital_choice_coefficients.csv"),
        "n_policies": len(rows),
        "policies": [x["policy"] for x in rows],
        "selection_protocol": (
            "two no-LBT3 variants by default; five withdrawn LBT3 variants only with audit flag"
        ),
        "withdrawn_lbt3_included": bool(args.allow_withdrawn_lbt3),
        "evaluation_role": "representative250 untouched final closed-loop test",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v14-guideline-suite] {len(rows)}개 정책 고정 → {out}")


if __name__ == "__main__":
    main()
