#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shin16 전국 평가의 완전성·재계산·기존 scoreboard 비교를 감사한다."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
RL_SRC = REPO / "src/rl_src"
if str(RL_SRC) not in sys.path:
    sys.path.insert(0, str(RL_SRC))

from shin_full_baselines import (  # noqa: E402
    source_bundle_sha256,
    source_hashes,
    valid_checkpoint,
    work_path,
)


SHIN_ROOT = REPO / "results/scoreboard/v10/shin16_full1000"
BASE_ROOT = REPO / "results/scoreboard/v10/full1000"


def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    return (
        float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )


def paired_label(reference, candidate):
    delta = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    mean = float(delta.mean())
    half = ci95(delta)
    return ("W" if mean > half else "L" if mean < -half else "T"), mean, half


def read_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def strategy_summary(name: str, cube: np.ndarray) -> dict:
    region_means = np.asarray(cube, dtype=float).mean(axis=1)
    return {
        "method": name,
        "n_regions": len(region_means),
        "n_episodes_per_region": cube.shape[1],
        "pdr_wog_mean": float(region_means.mean()),
        "pdr_wog_ci95_regions": ci95(region_means),
    }


def comparison(reference_name, candidate_name, reference, candidate) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    region_delta = reference.mean(axis=1) - candidate.mean(axis=1)
    labels = [
        paired_label(reference[idx], candidate[idx])[0]
        for idx in range(len(reference))
    ]
    ref_mean = float(reference.mean(axis=1).mean())
    improvement = float(region_delta.mean())
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "mean_improvement": improvement,
        "ci95_improvement_across_regions": ci95(region_delta),
        "relative_reduction_pct": 100.0 * improvement / ref_mean,
        "W": labels.count("W"),
        "T": labels.count("T"),
        "L": labels.count("L"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shin_root", type=Path, default=SHIN_ROOT)
    parser.add_argument("--base_root", type=Path, default=BASE_ROOT)
    args = parser.parse_args()
    shin_root = args.shin_root.resolve()
    base_root = args.base_root.resolve()
    out_dir = shin_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((shin_root / "protocol_meta.json").read_text(encoding="utf-8"))
    rules = list(meta["rules"])
    n_eps = int(meta["n_episodes_per_policy_per_coordinate"])
    seed = int(meta["evaluation_seed_start"])
    source_bundle = str(meta["source_bundle_sha256"])
    full = pd.read_csv(
        shin_root / "shin_full_summary.csv",
        dtype={"sigcd": str},
    )
    best = pd.read_csv(
        shin_root / "shin_best_summary.csv",
        dtype={"sigcd": str},
    )

    checks = {
        "meta_status_complete": meta["status"] == "complete",
        "full_rows_20000": len(full) == 20_000,
        "best_rows_1250": len(best) == 1_250,
        "full_primary_key_unique": not full.duplicated(
            ["dataset", "coordinate_key", "rule"]
        ).any(),
        "best_coordinate_unique": not best.duplicated(
            ["dataset", "coordinate_key"]
        ).any(),
        "full_no_null": not full.isna().any().any(),
        "best_no_null": not best.isna().any().any(),
        "each_coordinate_has_16_rules": bool(
            (full.groupby(["dataset", "coordinate_key"]).size() == 16).all()
        ),
        "source_bundle_matches_current": source_bundle_sha256()
        == source_bundle,
        "source_file_hashes_match_current": source_hashes()
        == meta["source_hashes"],
        "failed_jobs_absent": not (shin_root / "failed_jobs.json").exists(),
    }

    summary_index = full.set_index(["dataset", "coordinate_key", "rule"])
    best_index = best.set_index(["dataset", "coordinate_key"])
    entries = (
        full[
            ["dataset", "coordinate_key", "region", "sigcd", "point", "lat", "lon"]
        ]
        .drop_duplicates(["dataset", "coordinate_key"])
        .to_dict("records")
    )
    for entry in entries:
        entry["key"] = entry["coordinate_key"]
    checkpoint_count = 0
    max_rule_mean_error = 0.0
    max_best_mean_error = 0.0
    pdr_min = np.inf
    pdr_max = -np.inf
    equality = {
        (dataset, mode): 0
        for dataset in ("train1000", "eval250")
        for mode in range(4)
    }
    dataset_counts = {"train1000": 0, "eval250": 0}
    eval_values = {}

    for entry in entries:
        path = work_path(shin_root, entry)
        if not valid_checkpoint(path, n_eps, seed, rules, source_bundle):
            raise RuntimeError(f"체크포인트 검증 실패: {path}")
        data = read_npz(path)
        values = data["values"].astype(np.float64)
        checkpoint_count += 1
        dataset_counts[entry["dataset"]] += 1
        pdr = values[:, :, 3]
        pdr_min = min(pdr_min, float(pdr.min()))
        pdr_max = max(pdr_max, float(pdr.max()))

        means = pdr.mean(axis=1)
        reported = np.asarray(
            [
                summary_index.loc[
                    (entry["dataset"], entry["coordinate_key"], rule),
                    "shin_pdr_woG_mean",
                ]
                for rule in rules
            ],
            dtype=float,
        )
        max_rule_mean_error = max(
            max_rule_mean_error,
            float(np.max(np.abs(means - reported))),
        )
        best_idx = int(np.argmin(means))
        reported_best = float(
            best_index.loc[
                (entry["dataset"], entry["coordinate_key"]),
                "shin_best_pdr_woG_mean",
            ]
        )
        max_best_mean_error = max(
            max_best_mean_error,
            abs(float(means[best_idx]) - reported_best),
        )
        for mode in range(4):
            if np.array_equal(pdr[mode], pdr[4 + mode]):
                equality[(entry["dataset"], mode)] += 1
        if entry["dataset"] == "eval250":
            eval_values[entry["coordinate_key"]] = pdr

    checks.update(
        {
            "checkpoint_count_1250": checkpoint_count == 1_250,
            "dataset_checkpoint_split_1000_250": dataset_counts
            == {"train1000": 1000, "eval250": 250},
            "pdr_range_valid": pdr_min >= -1e-7 and pdr_max <= 1.0 + 1e-7,
            "summary_rule_means_recomputed": max_rule_mean_error < 1e-7,
            "summary_best_means_recomputed": max_best_mean_error < 1e-7,
        }
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"품질 게이트 실패: {failed}")

    rule_rows = []
    for (dataset, rule), group in full.groupby(["dataset", "rule"], sort=False):
        values = group["shin_pdr_woG_mean"].to_numpy(dtype=float)
        rule_rows.append(
            {
                "dataset": dataset,
                "rule": rule,
                "n_coordinates": len(values),
                "pdr_wog_mean": float(values.mean()),
                "pdr_wog_ci95_coordinates": ci95(values),
            }
        )
    rule_perf = pd.DataFrame(rule_rows)
    rule_perf["rank_in_dataset"] = rule_perf.groupby("dataset")[
        "pdr_wog_mean"
    ].rank(method="first")
    rule_perf.to_csv(
        out_dir / "rule_performance_by_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_rule = (
        rule_perf[rule_perf["dataset"] == "train1000"]
        .sort_values("pdr_wog_mean")
        .iloc[0]
    )
    global_rule = str(train_rule["rule"])
    global_idx = rules.index(global_rule)

    train = full[full["dataset"] == "train1000"]
    regional_train = (
        train.groupby(["sigcd", "rule"], as_index=False)["shin_pdr_woG_mean"]
        .mean()
        .sort_values(["sigcd", "shin_pdr_woG_mean", "rule"])
    )
    regional_rules = (
        regional_train.groupby("sigcd", as_index=False).first().set_index("sigcd")[
            "rule"
        ]
    )

    with np.load(
        base_root / "scoreboard_common30_episodes.npz",
        allow_pickle=False,
    ) as existing:
        regions = [str(value) for value in existing["regions"]]
        existing_methods = [str(value) for value in existing["methods"]]
        existing_cube = np.asarray(existing["pdr_wog"], dtype=float)
        existing_seeds = np.asarray(existing["seeds"], dtype=int)
    if regions != list(eval_values):
        raise RuntimeError("Shin eval 순서와 기존 scoreboard 지역 순서 불일치")
    if not np.array_equal(existing_seeds, np.arange(30)):
        raise RuntimeError("기존 common30 seed가 0..29가 아님")

    shin_global = np.stack([eval_values[region][global_idx] for region in regions])
    regional_idx = [
        rules.index(regional_rules.loc[region.rsplit("_", 1)[-1]])
        for region in regions
    ]
    shin_regional = np.stack(
        [eval_values[region][idx] for region, idx in zip(regions, regional_idx)]
    )
    oracle_idx = [
        int(np.argmin(eval_values[region].mean(axis=1))) for region in regions
    ]
    shin_oracle = np.stack(
        [eval_values[region][idx] for region, idx in zip(regions, oracle_idx)]
    )

    heur_best = []
    t4 = []
    baseline_max_error = 0.0
    base_summary = pd.read_csv(
        base_root / "baseline_summary.csv",
        dtype={"sigcd": str},
    ).set_index(["dataset", "coordinate_key"])
    for region in regions:
        heur_data = read_npz(
            base_root / "work/heur/eval250" / f"{region}.npz"
        )["values"].astype(np.float64)
        best_idx = int(np.argmin(heur_data[:, :, 3].mean(axis=1)))
        heur_values = heur_data[best_idx, :, 3]
        t4_values = read_npz(
            base_root / "work/t4/eval250" / f"{region}.npz"
        )["values"].astype(np.float64)[:, 3]
        heur_best.append(heur_values)
        t4.append(t4_values)
        row = base_summary.loc[("eval250", region)]
        baseline_max_error = max(
            baseline_max_error,
            abs(float(heur_values.mean()) - float(row["heur_best_pdr_woG_mean"])),
            abs(float(t4_values.mean()) - float(row["t4_pdr_woG_mean"])),
        )
    heur_best = np.stack(heur_best)
    t4 = np.stack(t4)
    if baseline_max_error >= 1e-7:
        raise RuntimeError(f"기존 baseline 재계산 불일치: {baseline_max_error}")

    cubes1000 = {
        "HEUR64_BEST": heur_best,
        "LB_T4": t4,
        "SHIN_GLOBAL_TRAIN_SELECTED": shin_global,
        "SHIN_REGIONAL_RANDOM4_SELECTED": shin_regional,
        "SHIN_EVAL_ORACLE_BEST16": shin_oracle,
    }
    strategies = pd.DataFrame(
        [strategy_summary(name, cube) for name, cube in cubes1000.items()]
    ).sort_values("pdr_wog_mean")
    strategies.to_csv(
        out_dir / "selection_strategies_eval250_seed0_999.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison_rows = []
    for candidate in (
        "SHIN_GLOBAL_TRAIN_SELECTED",
        "SHIN_REGIONAL_RANDOM4_SELECTED",
        "SHIN_EVAL_ORACLE_BEST16",
    ):
        for reference in ("HEUR64_BEST", "LB_T4"):
            comparison_rows.append(
                comparison(
                    reference,
                    candidate,
                    cubes1000[reference],
                    cubes1000[candidate],
                )
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        out_dir / "pairwise_eval250_seed0_999.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rule_counts = []
    for selection, indices in (
        ("regional_random4_selected", regional_idx),
        ("eval_oracle_best16", oracle_idx),
    ):
        counts = pd.Series([rules[idx] for idx in indices]).value_counts()
        for rule, count in counts.items():
            rule_counts.append(
                {
                    "selection": selection,
                    "rule": rule,
                    "n_sigungu": int(count),
                    "share_pct": 100.0 * int(count) / 250,
                }
            )
    rule_count_df = pd.DataFrame(rule_counts)
    rule_count_df.to_csv(
        out_dir / "selected_rule_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    common30_names = list(existing_methods) + [
        "SHIN_GLOBAL_TRAIN_SELECTED",
        "SHIN_REGIONAL_RANDOM4_SELECTED",
        "SHIN_EVAL_ORACLE_BEST16",
    ]
    common30_cube = np.concatenate(
        [
            existing_cube,
            shin_global[:, None, :30],
            shin_regional[:, None, :30],
            shin_oracle[:, None, :30],
        ],
        axis=1,
    )
    common30_rows = [
        strategy_summary(name, common30_cube[:, idx])
        for idx, name in enumerate(common30_names)
    ]
    common30 = pd.DataFrame(common30_rows).sort_values("pdr_wog_mean")
    common30.to_csv(
        out_dir / "common30_scoreboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    common30_comparisons = []
    for candidate in common30_names[-3:]:
        candidate_idx = common30_names.index(candidate)
        for reference in (
            "HEUR64_BEST",
            "LB_T4",
            "PPO_POINTER_V10",
            "PPO_POINTER_V10_NCRP_M16",
        ):
            reference_idx = common30_names.index(reference)
            common30_comparisons.append(
                comparison(
                    reference,
                    candidate,
                    common30_cube[:, reference_idx],
                    common30_cube[:, candidate_idx],
                )
            )
    pd.DataFrame(common30_comparisons).to_csv(
        out_dir / "common30_pairwise.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equality_rows = []
    mode_names = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
    for (dataset, mode), count in equality.items():
        equality_rows.append(
            {
                "dataset": dataset,
                "mode": mode_names[mode],
                "threshold_2step_exact_equal_coordinates": count,
                "n_coordinates": dataset_counts[dataset],
                "share_pct": 100.0 * count / dataset_counts[dataset],
            }
        )
    equality_df = pd.DataFrame(equality_rows)
    equality_df.to_csv(
        out_dir / "threshold_2step_equality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    runtime_hours = (
        float(meta["completed_at_unix"]) - float(meta["created_at_unix"])
    ) / 3600.0
    report = {
        "assessment": "ready_to_share_with_caveats",
        "checks": checks,
        "runtime_hours": runtime_hours,
        "checkpoint_count": checkpoint_count,
        "pdr_wog_min": pdr_min,
        "pdr_wog_max": pdr_max,
        "max_rule_summary_abs_error": max_rule_mean_error,
        "max_best_summary_abs_error": max_best_mean_error,
        "baseline_summary_max_abs_error": baseline_max_error,
        "global_rule_selected_on_train1000": global_rule,
        "global_rule_train1000_pdr_wog": float(train_rule["pdr_wog_mean"]),
        "regional_selection_uses": (
            "시군구별 random4 네 좌표 평균으로 규칙 선택 후 대표점 평가"
        ),
        "eval_oracle_caveat": (
            "대표점 seed0..999 결과를 보고 같은 대표점의 최적 규칙을 고른 사후 oracle"
        ),
        "threshold_2step_equality": equality_rows,
        "artifacts": {
            "rule_performance": str(
                (out_dir / "rule_performance_by_dataset.csv").relative_to(REPO)
            ),
            "selection_strategies": str(
                (
                    out_dir / "selection_strategies_eval250_seed0_999.csv"
                ).relative_to(REPO)
            ),
            "pairwise": str(
                (out_dir / "pairwise_eval250_seed0_999.csv").relative_to(REPO)
            ),
            "common30_scoreboard": str(
                (out_dir / "common30_scoreboard.csv").relative_to(REPO)
            ),
        },
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("VALIDATION_OK")
    print(f"runtime_hours={runtime_hours:.3f}")
    print(f"global_train_selected={global_rule}")
    print("\nRULE PERFORMANCE")
    print(
        rule_perf.sort_values(["dataset", "pdr_wog_mean"]).to_string(
            index=False
        )
    )
    print("\nEVAL STRATEGIES seed0..999")
    print(strategies.to_string(index=False))
    print("\nPAIRWISE seed0..999")
    print(comparisons.to_string(index=False))
    print("\nCOMMON30")
    print(common30.to_string(index=False))
    print("\nSELECTED RULE COUNTS")
    print(rule_count_df.to_string(index=False))
    print("\nTHRESHOLD vs 2STEP EXACT EQUALITY")
    print(equality_df.to_string(index=False))


if __name__ == "__main__":
    main()
