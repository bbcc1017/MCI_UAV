# -*- coding: utf-8 -*-
"""v10 후보랭킹 현장정책 병렬 적합: 축소 CART·EBM·LightGBM.

기존 C4 후보랭킹 트리와 동일하게 각 의사결정에서 마스크가 허용한
``[class, destination, mode]`` 후보에 점수를 매기고 최고점만 선택한다. 차이는 점수함수다.

* CART-L: C4와 같은 깊이에서 잎 예산을 줄여 성능기반 가지치기 후보를 만든다.
* EBM: 단변량 shape function과 제한된 이원 상호작용으로 현장 설명 가능성을 높인다.
* GBDT: 해석성보다 증류 성능 상한을 확인하는 강한 머신러닝 학생 기준선이다.

``critical2``는 실제 환자 outcome 라벨이 아니라 PPO 상위 후보 간 확률차를 제곱 가중한
중요도 대리변수다. 최종 모델 선택은 별도의 closed-loop PDR 평가로 수행해야 한다.
"""
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

from tree_distill_policy import (
    FEATURE_NAMES,
    INFO_LABELS,
    INFO_LEVELS,
    decode_action,
    tree_scores,
)
from v10_tree_distill import load_datasets, rank_metrics

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
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


def _state_rows(offsets: np.ndarray, state_ids: np.ndarray) -> np.ndarray:
    """선택 state의 후보행을 원래 순서대로 반환."""
    parts = [
        np.arange(int(offsets[s]), int(offsets[s + 1]), dtype=np.int64)
        for s in state_ids
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _sample_rows(data: dict, max_states: int, seed: int) -> np.ndarray:
    n = len(data["ncand"])
    if max_states <= 0 or n <= max_states:
        return np.arange(len(data["X"]), dtype=np.int64)
    rng = np.random.default_rng(seed)
    state_ids = np.sort(rng.choice(n, size=max_states, replace=False))
    return _state_rows(data["offsets"], state_ids)


def _state_gap_rows(data: dict) -> np.ndarray:
    """각 후보행에 그 state의 PPO top1-top2 확률차를 반복."""
    gap = np.empty(len(data["ncand"]), dtype=np.float32)
    for i, (s, e) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        p = data["target"][int(s):int(e)]
        if len(p) <= 1:
            gap[i] = 1.0
        else:
            top = np.partition(p, -2)[-2:]
            gap[i] = float(top[-1] - top[-2])
    return np.repeat(gap, data["ncand"].astype(int))


def _weights(data: dict, scheme: str) -> np.ndarray:
    if scheme == "stored":
        return np.asarray(data["weight"], dtype=np.float64)
    nrow = np.repeat(data["ncand"], data["ncand"].astype(int)).astype(np.float64)
    base = (1.0 + 4.0 * np.asarray(data["target"], dtype=np.float64)) / nrow
    gap = _state_gap_rows(data).astype(np.float64)
    if scheme == "soft":
        return base
    if scheme == "critical":
        return base * (0.05 + gap)
    if scheme == "critical2":
        return base * np.square(0.05 + gap)
    raise ValueError(f"지원하지 않는 가중 방식: {scheme}")


def _case_specs() -> list[dict]:
    cases: list[dict] = []
    for level, leaves in (
        ("I1_FIELD", (64, 128, 256, 384)),
        ("I3_CONNECTED", (128, 256, 384)),
    ):
        for leaf in leaves:
            cases.append({
                "info_level": level,
                "family": "cart",
                "complexity": f"CART_L{leaf:03d}",
                "leaves": leaf,
                "weight_scheme": "stored",
            })
    for level, interactions in (
        ("I1_FIELD", (0, 4, 8)),
        ("I3_CONNECTED", (4, 8)),
    ):
        for inter in interactions:
            cases.append({
                "info_level": level,
                "family": "ebm",
                "complexity": f"EBM_I{inter:02d}",
                "interactions": inter,
                "weight_scheme": "stored",
            })
    for level, leaves, scheme in (
        ("I1_FIELD", 15, "stored"),
        ("I1_FIELD", 31, "stored"),
        ("I1_FIELD", 63, "stored"),
        ("I1_FIELD", 31, "soft"),
        ("I1_FIELD", 31, "critical2"),
        ("I3_CONNECTED", 31, "stored"),
        ("I3_CONNECTED", 63, "stored"),
    ):
        suffix = {"stored": "BASE", "soft": "SOFT", "critical2": "CRIT2"}[scheme]
        cases.append({
            "info_level": level,
            "family": "lgbm",
            "complexity": f"GBDT_L{leaves:02d}_{suffix}",
            "leaves": leaves,
            "weight_scheme": scheme,
        })
    return cases


_TRAIN: dict | None = None
_VAL: dict | None = None
_ARGS = None


def _fit_worker(spec: dict) -> dict:
    """fork 공유 데이터에서 모델 하나를 적합하고 패키지를 저장."""
    try:
        assert _TRAIN is not None and _VAL is not None and _ARGS is not None
        level = spec["info_level"]
        feat_idx = np.asarray(INFO_LEVELS[level], dtype=int)
        case = f"{level}_{spec['complexity']}"
        family = spec["family"]
        seed = int(_ARGS.random_seed)
        max_states = (
            int(_ARGS.ebm_max_states) if family == "ebm"
            else int(_ARGS.max_states)
        )
        rows = _sample_rows(_TRAIN, max_states, seed + sum(map(ord, case)))
        X = _TRAIN["X"][rows][:, feat_idx]
        y = _TRAIN["target"][rows]
        w_all = _weights(_TRAIN, spec["weight_scheme"])
        w = w_all[rows]
        t0 = time.time()

        if family == "cart":
            from sklearn.tree import DecisionTreeRegressor

            model = DecisionTreeRegressor(
                max_depth=10,
                max_leaf_nodes=int(spec["leaves"]),
                min_samples_leaf=40,
                random_state=seed,
            )
            model.fit(X, y, sample_weight=w)
            depth = int(model.get_depth())
            leaves = int(model.get_n_leaves())
            nodes = int(model.tree_.node_count)
            n_used = int(np.count_nonzero(model.feature_importances_))
        elif family == "ebm":
            from interpret.glassbox import ExplainableBoostingRegressor

            model = ExplainableBoostingRegressor(
                feature_names=[FEATURE_NAMES[i] for i in feat_idx],
                max_bins=64,
                max_interaction_bins=32,
                interactions=int(spec["interactions"]),
                validation_size=0.10,
                outer_bags=4,
                inner_bags=0,
                learning_rate=0.04,
                max_rounds=2000,
                early_stopping_rounds=75,
                min_samples_leaf=40,
                max_leaves=3,
                n_jobs=1,
                random_state=seed,
            )
            model.fit(X, y, sample_weight=w)
            depth = -1
            leaves = -1
            nodes = int(len(model.term_names_))
            n_used = int(sum(np.any(np.asarray(v) != 0) for v in model.term_scores_))
        elif family == "lgbm":
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="regression_l2",
                num_leaves=int(spec["leaves"]),
                learning_rate=0.04,
                n_estimators=600,
                min_child_samples=40,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.90,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(X, y, sample_weight=w)
            depth = int(np.max(model.booster_.dump_model()["tree_info"][0]["tree_structure"].get(
                "split_index", -1
            ))) if False else -1
            leaves = int(spec["leaves"])
            nodes = int(model.n_estimators_)
            n_used = int(np.count_nonzero(model.feature_importances_))
        else:
            raise ValueError(f"미지 모델군: {family}")

        package = {
            "schema_version": 2,
            "tree": model,
            "estimator_kind": "regressor",
            "objective": "ppo_masked_probability",
            "family": family,
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
            "weight_scheme": spec["weight_scheme"],
            "importance_semantics": (
                "PPO top1-top2 masked probability gap surrogate; not patient outcome"
            ),
            "git_sha": _git_sha(),
        }
        metrics = rank_metrics(package, _VAL, max_states=int(_ARGS.max_val_states))
        package["validation"] = metrics
        out = Path(_ARGS.out_dir).resolve() / f"{case}.pkl"
        with open(out, "wb") as f:
            pickle.dump(package, f)
        return {
            "ok": True,
            "policy": case,
            "info_level": level,
            "info_label": INFO_LABELS[level],
            "complexity": spec["complexity"],
            "family": family,
            "weight_scheme": spec["weight_scheme"],
            "n_features_available": len(feat_idx),
            "n_features_used": n_used,
            "depth": depth,
            "leaves": leaves,
            "nodes": nodes,
            "n_fit_candidate_rows": len(rows),
            "fit_seconds": time.time() - t0,
            **metrics,
            "pkl": str(out),
            "pkl_sha256": _sha256(out),
        }
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "policy": f"{spec.get('info_level')}_{spec.get('complexity')}",
            "err": (str(exc) + "\n" + traceback.format_exc())[:4000],
        }


def main() -> None:
    global _TRAIN, _VAL, _ARGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=True, help="쉼표구분 npz")
    ap.add_argument("--val_data", required=True, help="폐루프와 분리한 npz")
    ap.add_argument("--families", default="cart,ebm,lgbm")
    ap.add_argument("--levels", default="I1_FIELD,I3_CONNECTED")
    ap.add_argument("--max_states", type=int, default=0, help="0이면 CART/GBDT 전체 state")
    ap.add_argument("--ebm_max_states", type=int, default=30000)
    ap.add_argument("--max_val_states", type=int, default=100000)
    ap.add_argument("--random_seed", type=int, default=20260726)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    _ARGS = args

    train_paths = [str(Path(x).resolve()) for x in args.train_data.split(",") if x]
    val_paths = [str(Path(x).resolve()) for x in args.val_data.split(",") if x]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[student-fit] 데이터 적재 중", flush=True)
    _TRAIN = load_datasets(train_paths)
    _VAL = load_datasets(val_paths)
    if not np.isfinite(_TRAIN["X"]).all() or not np.isfinite(_TRAIN["target"]).all():
        raise RuntimeError("학습 데이터 비유한 값")
    if _TRAIN["offsets"][-1] != len(_TRAIN["X"]):
        raise RuntimeError("학습 offsets 불일치")

    families = set(x for x in args.families.split(",") if x)
    levels = set(x for x in args.levels.split(",") if x)
    specs = [
        x for x in _case_specs()
        if x["family"] in families and x["info_level"] in levels
    ]
    if not specs:
        raise ValueError("적합할 실험군이 없음")
    print(
        f"[student-fit] states={len(_TRAIN['ncand'])} rows={len(_TRAIN['X'])} "
        f"val_states={len(_VAL['ncand'])} cases={len(specs)} workers={min(args.workers,len(specs))}",
        flush=True,
    )
    t0 = time.time()
    rows = []
    with Pool(min(args.workers, len(specs))) as pool:
        for i, result in enumerate(pool.imap_unordered(_fit_worker, specs), 1):
            if not result["ok"]:
                raise RuntimeError(f"적합 실패 {result['policy']}: {result['err']}")
            rows.append(result)
            print(
                f"  [{i}/{len(specs)}] {result['policy']} "
                f"fid={result['fidelity_full']:.3f} rows={result['n_fit_candidate_rows']:,} "
                f"fit={result['fit_seconds']:.1f}s",
                flush=True,
            )

    rows.sort(key=lambda x: x["policy"])
    summary = out_dir / "fit_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "schema_version": 2,
        "purpose": "C4 pruning / EBM / GBDT / criticality weighting parallel suite",
        "train_data": {p: {"sha256": _sha256(p), "bytes": os.path.getsize(p)} for p in train_paths},
        "val_data": {p: {"sha256": _sha256(p), "bytes": os.path.getsize(p)} for p in val_paths},
        "n_train_states": int(len(_TRAIN["ncand"])),
        "n_train_candidate_rows": int(len(_TRAIN["X"])),
        "n_val_states": int(len(_VAL["ncand"])),
        "cases": [x["policy"] for x in rows],
        "random_seed": args.random_seed,
        "criticality_caveat": (
            "critical/critical2 are PPO masked probability-gap surrogates, "
            "not patient outcome labels; select by closed-loop PDR"
        ),
        "git_sha": _git_sha(),
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "fit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[student-fit] 완료 {out_dir} wall={(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
