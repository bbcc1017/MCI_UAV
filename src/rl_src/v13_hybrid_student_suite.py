# -*- coding: utf-8 -*-
"""v13 최종 하이브리드 hard-action 라벨의 CART·EBM·GBDT 병렬 적합."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

from tree_distill_policy import FEATURE_NAMES, INFO_LABELS, INFO_LEVELS
from v10_tree_distill import load_datasets, rank_metrics
from v10_student_suite import _case_specs, _sample_rows

REPO = Path(__file__).resolve().parents[2]
_TRAIN = None
_VAL = None
_ARGS = None


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _fit_worker(spec: dict) -> dict:
    try:
        assert _TRAIN is not None and _VAL is not None and _ARGS is not None
        level = spec["info_level"]
        family = spec["family"]
        feat_idx = np.asarray(INFO_LEVELS[level], dtype=int)
        case = f"{level}_{spec['complexity']}"
        max_states = _ARGS.ebm_max_states if family == "ebm" else _ARGS.max_states
        rows = _sample_rows(_TRAIN, max_states, _ARGS.random_seed + sum(map(ord, case)))
        X = _TRAIN["X"][rows][:, feat_idx]
        y = _TRAIN["chosen"][rows].astype(np.int8)
        w = _TRAIN["weight"][rows].astype(np.float64)
        t0 = time.time()

        if family == "cart":
            from sklearn.tree import DecisionTreeClassifier, export_text
            model = DecisionTreeClassifier(
                max_depth=10, max_leaf_nodes=int(spec["leaves"]),
                min_samples_leaf=40, random_state=_ARGS.random_seed,
            )
            model.fit(X, y, sample_weight=w)
            depth = int(model.get_depth())
            leaves = int(model.get_n_leaves())
            nodes = int(model.tree_.node_count)
            n_used = int(np.count_nonzero(model.feature_importances_))
            rules = export_text(
                model, feature_names=[FEATURE_NAMES[i] for i in feat_idx],
                max_depth=6, decimals=3,
            )
        elif family == "ebm":
            from interpret.glassbox import ExplainableBoostingClassifier
            model = ExplainableBoostingClassifier(
                feature_names=[FEATURE_NAMES[i] for i in feat_idx],
                max_bins=64, max_interaction_bins=32,
                interactions=int(spec["interactions"]), validation_size=0.10,
                outer_bags=4, inner_bags=0, learning_rate=0.04,
                max_rounds=2000, early_stopping_rounds=75,
                min_samples_leaf=40, max_leaves=3, n_jobs=1,
                random_state=_ARGS.random_seed,
            )
            model.fit(X, y, sample_weight=w)
            depth = leaves = -1
            nodes = int(len(model.term_names_))
            n_used = int(sum(np.any(np.asarray(v) != 0) for v in model.term_scores_))
            rules = ""
        elif family == "lgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective="binary", num_leaves=int(spec["leaves"]),
                learning_rate=0.04, n_estimators=600, min_child_samples=40,
                subsample=0.85, subsample_freq=1, colsample_bytree=0.90,
                reg_lambda=1.0, random_state=_ARGS.random_seed, n_jobs=1,
                verbosity=-1, deterministic=True, force_col_wise=True,
            )
            model.fit(X, y, sample_weight=w)
            depth = -1
            leaves = int(spec["leaves"])
            nodes = int(model.n_estimators_)
            n_used = int(np.count_nonzero(model.feature_importances_))
            rules = ""
        else:
            raise ValueError(f"미지원 family: {family}")

        package = {
            "schema_version": 3,
            "tree": model,
            "estimator_kind": "classifier",
            "objective": "hybrid_final_action_hard_label",
            "teacher": "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
            "info_level": level,
            "info_label": INFO_LABELS[level],
            "feature_indices": feat_idx.tolist(),
            "feature_names": [FEATURE_NAMES[i] for i in feat_idx],
            "complexity": spec["complexity"],
            "complexity_spec": spec,
            "actual_depth": depth,
            "actual_leaves": leaves,
            "n_train_states_total": int(len(_TRAIN["ncand"])),
            "n_train_candidate_rows_total": int(len(_TRAIN["X"])),
            "n_fit_candidate_rows": int(len(rows)),
            "weight_semantics": "per-state positive/negative totals = 0.5/0.5",
            "generalization_caveat": "planner action depends on sampled futures absent from student features",
            "git_sha": _git_sha(),
        }
        val = rank_metrics(package, _VAL, max_states=_ARGS.max_val_states)
        train = rank_metrics(package, _TRAIN, max_states=_ARGS.max_train_metric_states)
        package["validation"] = val
        package["train_metrics"] = train
        out = Path(_ARGS.out_dir).resolve() / f"{case}.pkl"
        with open(out, "wb") as f:
            pickle.dump(package, f)
        if rules:
            (out.parent / f"{case}_rules.txt").write_text(
                f"teacher={package['teacher']}\ntrain={train}\nvalidation={val}\n\n{rules}\n",
                encoding="utf-8",
            )
        return {
            "ok": True, "policy": case, "info_level": level,
            "info_label": INFO_LABELS[level], "complexity": spec["complexity"],
            "family": family, "n_features_available": len(feat_idx),
            "n_features_used": n_used, "depth": depth, "leaves": leaves,
            "nodes": nodes, "n_fit_candidate_rows": len(rows),
            "fit_seconds": time.time() - t0,
            "train_fidelity_full": train["fidelity_full"],
            "validation_gap": train["fidelity_full"] - val["fidelity_full"],
            **val, "pkl": str(out), "pkl_sha256": _sha256(out),
        }
    except Exception as exc:
        import traceback
        return {"ok": False, "policy": str(spec), "err": (str(exc) + "\n" + traceback.format_exc())[:5000]}


def main() -> None:
    global _TRAIN, _VAL, _ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True, help="쉼표구분 npz; 최종적합은 train+val 병합")
    p.add_argument("--val_data", required=True)
    p.add_argument("--families", default="cart,ebm,lgbm")
    p.add_argument("--levels", default="I1_FIELD,I3_CONNECTED")
    p.add_argument("--max_states", type=int, default=0)
    p.add_argument("--ebm_max_states", type=int, default=15000)
    p.add_argument("--max_val_states", type=int, default=100000)
    p.add_argument("--max_train_metric_states", type=int, default=30000)
    p.add_argument("--random_seed", type=int, default=20260803)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    _ARGS = args
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"기존 학생 산출물 보호: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [str(Path(x).resolve()) for x in args.train_data.split(",") if x]
    val_paths = [str(Path(x).resolve()) for x in args.val_data.split(",") if x]
    _TRAIN = load_datasets(train_paths)
    _VAL = load_datasets(val_paths)
    for name, data in (("train", _TRAIN), ("val", _VAL)):
        if not np.isfinite(data["X"]).all() or not np.isfinite(data["weight"]).all():
            raise RuntimeError(f"{name} 비유한 값")
        if data["offsets"][-1] != len(data["X"]):
            raise RuntimeError(f"{name} offsets 불일치")

    families = {x for x in args.families.split(",") if x}
    levels = {x for x in args.levels.split(",") if x}
    specs = [x for x in _case_specs() if x["family"] in families and x["info_level"] in levels]
    print(
        f"[hybrid-fit] train_states={len(_TRAIN['ncand']):,} val_states={len(_VAL['ncand']):,} "
        f"cases={len(specs)} workers={min(args.workers,len(specs))}", flush=True,
    )
    t0, results = time.time(), []
    with Pool(min(args.workers, len(specs))) as pool:
        for i, result in enumerate(pool.imap_unordered(_fit_worker, specs), 1):
            if not result["ok"]:
                raise RuntimeError(f"적합 실패: {result['err']}")
            results.append(result)
            print(
                f"  [{i}/{len(specs)}] {result['policy']} "
                f"train/val={result['train_fidelity_full']:.3f}/{result['fidelity_full']:.3f} "
                f"gap={result['validation_gap']:.3f}", flush=True,
            )
    results.sort(key=lambda x: (-x["fidelity_full"], x["policy"]))
    summary = out_dir / "fit_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    meta = {
        "schema_version": 1,
        "teacher": "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        "objective": "hybrid_final_action_hard_label",
        "train_data": {x: {"sha256": _sha256(x), "bytes": os.path.getsize(x)} for x in train_paths},
        "val_data": {x: {"sha256": _sha256(x), "bytes": os.path.getsize(x)} for x in val_paths},
        "n_train_states": int(len(_TRAIN["ncand"])),
        "n_val_states": int(len(_VAL["ncand"])),
        "cases": [x["policy"] for x in results],
        "selection_warning": "fidelity is diagnostic; final selection requires closed-loop PDR",
        "structural_warning": "future-rollout realization is not included in student features",
        "git_sha": _git_sha(), "wall_seconds": time.time() - t0,
    }
    (out_dir / "fit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[hybrid-fit] 완료 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
