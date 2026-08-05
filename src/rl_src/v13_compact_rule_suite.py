# -*- coding: utf-8 -*-
"""최종 교사의 통계 규칙을 사전고정한 compact 정책 패키지로 만든다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(__file__))

from compact_rule_policy import CompactRuleEstimator
from tree_distill_policy import FEATURE_NAMES

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import v13_sota_rule_analysis as analysis


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_axis_tree(frame, features, target, mask, seed):
    x = frame.loc[np.asarray(mask, dtype=bool)].copy()
    counts = x.groupby("region_base", observed=True).region_base.transform("size").to_numpy(float)
    weights = (1.0 / counts)
    weights /= weights.mean()
    model = DecisionTreeClassifier(
        max_depth=3,
        max_leaf_nodes=8,
        min_samples_leaf=max(100, int(0.01 * len(x))),
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x[features].to_numpy(float), x[target].to_numpy(int), sample_weight=weights)
    return model, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_log", type=Path, default=analysis.TRAIN_LOG)
    p.add_argument("--analysis_dir", type=Path, default=analysis.DEFAULT_OUT)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"기존 compact 정책 보호: {out}")
    out.mkdir(parents=True, exist_ok=True)

    z, frame, quality = analysis.load_log(args.train_log.resolve(), "train")
    frame["select_red"] = (frame.teacher_class == 0).astype(int)
    frame["select_uav"] = (frame.teacher_mode == 1).astype(int)
    class_tree, _ = fit_axis_tree(
        frame, analysis.CLASS_FEATURES, "select_red", frame.both_class_available, 20260803,
    )
    mode_mask = (frame.teacher_dest > 0) & frame.both_mode_available
    mode_tree, _ = fit_axis_tree(
        frame, analysis.MODE_FEATURES, "select_uav", mode_mask, 20260804,
    )

    coef = (
        pd.read_csv(args.analysis_dir / "hospital_conditional_choice_coefficients.csv")
        .set_index("feature").raw_unit_coef_train_all.to_dict()
    )
    eta = {"eta_rank": float(coef["eta_rank"])}
    eta_occ = {**eta, "cand_occ_ratio": float(coef["cand_occ_ratio"])}
    specs = [
        ("ETA", False, False, eta),
        ("ETA_OCC", False, False, eta_occ),
        ("CLASS_ETA_OCC", True, False, eta_occ),
        ("MODE_ETA_OCC", False, True, eta_occ),
        ("CM_ETA", True, True, eta),
        ("CM_ETA_OCC", True, True, eta_occ),
    ]
    rows = []
    for name, use_class, use_mode, hosp in specs:
        estimator = CompactRuleEstimator(
            feature_names=FEATURE_NAMES,
            class_tree=class_tree if use_class else None,
            class_features=analysis.CLASS_FEATURES if use_class else [],
            mode_tree=mode_tree if use_mode else None,
            mode_features=analysis.MODE_FEATURES if use_mode else [],
            hospital_coef=hosp,
        )
        package = {
            "schema_version": 1,
            "tree": estimator,
            "estimator_kind": "classifier",
            "objective": "compact_statistical_rule_policy",
            "teacher": "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
            "info_level": "RULE",
            "info_label": "통계규칙",
            "feature_indices": list(range(len(FEATURE_NAMES))),
            "feature_names": list(FEATURE_NAMES),
            "complexity": name,
            "complexity_spec": {
                "class_tree": use_class,
                "mode_tree": use_mode,
                "hospital_coef": hosp,
                "stay_penalty": 25.0,
            },
            "train_log_sha256": sha256(args.train_log.resolve()),
            "analysis_coef_sha256": sha256(args.analysis_dir / "hospital_conditional_choice_coefficients.csv"),
            "selection_protocol": "six nested/factorial variants fixed before eval250 simulation",
        }
        path = out / f"RULE_{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(package, f)
        rows.append({
            "policy": f"RULE_{name}",
            "info_level": "RULE",
            "complexity": name,
            "family": "compact_rule",
            "class_tree": use_class,
            "mode_tree": use_mode,
            "hospital_coef": json.dumps(hosp, ensure_ascii=False, sort_keys=True),
            "pkl": str(path),
            "pkl_sha256": sha256(path),
        })

    with (out / "fit_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "class_tree.txt").write_text(
        export_text(class_tree, feature_names=analysis.CLASS_FEATURES, decimals=2), encoding="utf-8",
    )
    (out / "mode_tree.txt").write_text(
        export_text(mode_tree, feature_names=analysis.MODE_FEATURES, decimals=2), encoding="utf-8",
    )
    (out / "meta.json").write_text(json.dumps({
        "schema_version": 1,
        "train_log": str(args.train_log.resolve()),
        "train_log_sha256": sha256(args.train_log.resolve()),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "n_states": quality["states"],
        "variants": [x["policy"] for x in rows],
        "eval_manifest_role": "representative250 final untouched by rule fitting",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[compact-rule] {len(rows)}개 사전고정 정책 생성 → {out}")


if __name__ == "__main__":
    main()
