#!/usr/bin/env python3
"""
v10 현장형 증류정책 병렬실험의 통계 검증, scoreboard, 그림, 보고서 입력 생성.

핵심 원칙
----------
1. 대표점 250은 모델 선택용 개발셋으로만 사용한다.
2. 최종 수치는 새 외부 좌표 250 × seed 10의 공통 난수 평가만 사용한다.
3. PDR_woG는 낮을수록 좋다.
4. 승/무/패는 지역별 paired episode 차이의 95% 신뢰구간으로 판정한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DISTILL = ROOT / "results/scoreboard/v10/distill"
DEFAULT_OUT = DISTILL / "student_experiments"
Z95 = 1.959963984540054

METHOD_LABELS = {
    "HEUR64_GLOBAL_TRAIN_BEST": "Global heuristic",
    "GLOBAL_TRAIN_BEST_T4": "Global heuristic + T4",
    "I3_CONNECTED_EBM_I08": "EBM (I3, 8 interactions)",
    "I3_CONNECTED_C4": "CART C4 (512 leaves)",
    "I3_CONNECTED_CART_L384": "CART pruned (384 leaves)",
    "PPO": "PPO teacher",
    "STUDENT": "GBDT student (I1)",
    "STUDENT_NCRP_C75": "GBDT student + selective NCRP",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ci95(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(Z95 * np.std(x, ddof=1) / math.sqrt(x.size))


def normalized_eval(path: Path, method: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pdr_woG" not in df.columns:
        raise ValueError(f"{path}: pdr_woG 열이 없습니다.")
    if method is not None:
        df = df.copy()
        df["method"] = method
    elif "method" not in df.columns:
        if "policy" not in df.columns:
            raise ValueError(f"{path}: method/policy 열이 없습니다.")
        df = df.rename(columns={"policy": "method"})
    required = {"region", "method", "seed", "pdr_woG", "ms_per_decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 필수 열 누락 {sorted(missing)}")
    return df


def load_external() -> tuple[pd.DataFrame, dict[str, str]]:
    paths = {
        "baseline": DISTILL / "external250_baselines_seed10000_10009.csv",
        "student_ppo": DISTILL / "external250_student_ppo_seed10000_10009.csv",
        "c4": DISTILL / "external250_c4_seed10000_10009.csv",
        "cart384": DISTILL / "external250_cart_l384_seed10000_10009.csv",
        "ebm": DISTILL / "external250_ebm_i08_seed10000_10009.csv",
        "selective": DISTILL / "external250_selective_ncrp_c75_seed10000_10009.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = normalized_eval(paths["baseline"])
    student_ppo = normalized_eval(paths["student_ppo"])
    c4 = normalized_eval(paths["c4"])
    cart384 = normalized_eval(paths["cart384"])
    ebm = normalized_eval(paths["ebm"])
    selective = normalized_eval(paths["selective"])

    # 선택적 평가에 반복 포함된 STUDENT는 동일 궤적 검증에만 쓰고 scoreboard에서는 제거한다.
    left = student_ppo[student_ppo["method"] == "STUDENT"][
        ["region", "seed", "pdr_woG"]
    ].sort_values(["region", "seed"])
    right = selective[selective["method"] == "STUDENT"][
        ["region", "seed", "pdr_woG"]
    ].sort_values(["region", "seed"])
    if not np.array_equal(left[["region", "seed"]].to_numpy(), right[["region", "seed"]].to_numpy()):
        raise ValueError("선택적 평가와 기본 평가의 STUDENT key가 다릅니다.")
    if not np.allclose(left["pdr_woG"], right["pdr_woG"], atol=1e-12, rtol=0):
        raise ValueError("선택적 평가와 기본 평가의 STUDENT 궤적이 재현되지 않았습니다.")

    ext = pd.concat(
        [
            baseline,
            ebm,
            c4,
            cart384,
            student_ppo,
            selective[selective["method"] == "STUDENT_NCRP_C75"],
        ],
        ignore_index=True,
        sort=False,
    )
    return ext, {name: sha256(path) for name, path in paths.items()}


def validate_external(df: pd.DataFrame) -> dict[str, Any]:
    methods = sorted(df["method"].unique())
    expected_methods = sorted(METHOD_LABELS)
    if methods != expected_methods:
        raise ValueError(f"외부평가 method 불일치: {methods} != {expected_methods}")
    if not df["pdr_woG"].between(0, 1).all():
        raise ValueError("외부평가 PDR_woG가 [0,1] 범위를 벗어났습니다.")
    if df[["region", "method", "seed"]].duplicated().any():
        raise ValueError("외부평가 region-method-seed 중복이 있습니다.")

    region_sets = {
        method: set(g["region"]) for method, g in df.groupby("method", observed=True)
    }
    seed_sets = {
        method: set(map(int, g["seed"])) for method, g in df.groupby("method", observed=True)
    }
    first_regions = region_sets[methods[0]]
    first_seeds = seed_sets[methods[0]]
    if any(v != first_regions for v in region_sets.values()):
        raise ValueError("외부평가 method별 지역 집합이 다릅니다.")
    if any(v != first_seeds for v in seed_sets.values()):
        raise ValueError("외부평가 method별 seed 집합이 다릅니다.")
    if len(first_regions) != 250 or first_seeds != set(range(10000, 10010)):
        raise ValueError(
            f"외부평가 프로토콜 불일치: regions={len(first_regions)}, seeds={sorted(first_seeds)}"
        )
    counts = df.groupby("method", observed=True).size()
    if not (counts == 2500).all():
        raise ValueError(f"외부평가 method별 행 수 불일치: {counts.to_dict()}")

    return {
        "status": "pass",
        "n_methods": len(methods),
        "n_regions": len(first_regions),
        "seeds": sorted(first_seeds),
        "rows_per_method": {k: int(v) for k, v in counts.items()},
        "pdr_bounds": [float(df["pdr_woG"].min()), float(df["pdr_woG"].max())],
        "duplicate_region_method_seed": 0,
        "common_region_set": True,
        "common_seed_set": True,
        "student_duplicate_trajectory_exact": True,
    }


def external_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, g in df.groupby("method", observed=True):
        region_means = g.groupby("region", observed=True)["pdr_woG"].mean()
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "family": (
                "heuristic"
                if "HEUR" in method or "T4" in method
                else "planner"
                if "NCRP" in method
                else "RL"
                if method == "PPO"
                else "student"
            ),
            "pdr_woG_mean": float(g["pdr_woG"].mean()),
            "pdr_woG_region_ci95": ci95(region_means),
            "n_regions": int(g["region"].nunique()),
            "n_episodes": int(len(g)),
            "ms_per_decision_mean": float(g["ms_per_decision"].mean()),
            "student_coverage": (
                float(g["student_coverage"].mean())
                if "student_coverage" in g and g["student_coverage"].notna().any()
                else float("nan")
            ),
            "defer_rate": (
                float(g["defer_rate"].mean())
                if "defer_rate" in g and g["defer_rate"].notna().any()
                else float("nan")
            ),
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("pdr_woG_mean").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["latency_log10_ms"] = np.log10(out["ms_per_decision_mean"].clip(lower=1e-6))
    return out


def external_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    student = df[df["method"] == "STUDENT"][
        ["region", "seed", "pdr_woG"]
    ].rename(columns={"pdr_woG": "student_pdr"})
    rows: list[dict[str, Any]] = []
    for method, g in df.groupby("method", observed=True):
        if method == "STUDENT":
            continue
        paired = g[["region", "seed", "pdr_woG"]].merge(
            student, on=["region", "seed"], how="inner", validate="one_to_one"
        )
        # 양수면 STUDENT가 비교정책보다 낮은 PDR.
        paired["student_improvement"] = paired["pdr_woG"] - paired["student_pdr"]
        region_delta = paired.groupby("region", observed=True)["student_improvement"].mean()

        wins = ties = losses = 0
        for _, rg in paired.groupby("region", observed=True):
            delta = rg["student_improvement"]
            mean = float(delta.mean())
            half = ci95(delta)
            if mean > half:
                wins += 1
            elif mean < -half:
                losses += 1
            else:
                ties += 1

        rows.append(
            {
                "comparator": method,
                "comparator_label": METHOD_LABELS[method],
                "student_improvement_pdr": float(paired["student_improvement"].mean()),
                "episode_ci95": ci95(paired["student_improvement"]),
                "region_mean_ci95": ci95(region_delta),
                "relative_improvement_pct": float(
                    100
                    * paired["student_improvement"].mean()
                    / paired["pdr_woG"].mean()
                ),
                "student_wins": wins,
                "ties": ties,
                "student_losses": losses,
                "n_paired_episodes": int(len(paired)),
                "n_regions": int(len(region_delta)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "student_improvement_pdr", ascending=False
    ).reset_index(drop=True)


def fit_summary() -> pd.DataFrame:
    packages = sorted((DISTILL / "students_parallel").glob("*.pkl")) + sorted(
        (DISTILL / "students_ebm").glob("*.pkl")
    )
    rows: list[dict[str, Any]] = []
    for path in packages:
        pkg = joblib.load(path)
        val = pkg.get("validation", {})
        rows.append(
            {
                "policy": path.stem,
                "info_level": pkg.get("info_level"),
                "family": pkg.get("family", pkg.get("estimator_kind")),
                "complexity": pkg.get("complexity"),
                "weight_scheme": pkg.get("weight_scheme", "stored"),
                "n_features": len(pkg.get("feature_names", [])),
                "max_leaves_or_terms": pkg.get("actual_leaves", -1),
                "n_fit_candidate_rows": pkg.get(
                    "n_fit_candidate_rows", pkg.get("n_train_candidate_rows", -1)
                ),
                "fidelity_full": val.get("fidelity_full"),
                "fidelity_class": val.get("fidelity_class"),
                "fidelity_dest": val.get("fidelity_dest"),
                "fidelity_mode": val.get("fidelity_mode"),
                "package": str(path.relative_to(ROOT)),
                "package_sha256": sha256(path),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 19:
        raise ValueError(f"학생후보 패키지가 19개가 아닙니다: {len(out)}")
    return out.sort_values(["family", "info_level", "complexity"]).reset_index(drop=True)


def development_summary(fit: pd.DataFrame) -> pd.DataFrame:
    frames = [
        normalized_eval(DISTILL / "students14_closedloop40_seed8000_8009.csv"),
        normalized_eval(DISTILL / "ebm5_closedloop40_seed8000_8009.csv"),
    ]
    dev = pd.concat(frames, ignore_index=True)
    if dev[["region", "method", "seed"]].duplicated().any():
        raise ValueError("개발평가 region-method-seed 중복이 있습니다.")
    if dev["method"].nunique() != 19:
        raise ValueError(f"개발평가 후보 수가 19가 아닙니다: {dev['method'].nunique()}")
    if not (dev.groupby("method").size() == 400).all():
        raise ValueError("개발평가 후보별 행 수가 400이 아닙니다.")
    if set(dev["seed"]) != set(range(8000, 8010)) or dev["region"].nunique() != 40:
        raise ValueError("개발평가의 지역 또는 seed 프로토콜이 다릅니다.")

    rows = []
    for method, g in dev.groupby("method", observed=True):
        region_mean = g.groupby("region", observed=True)["pdr_woG"].mean()
        rows.append(
            {
                "policy": method,
                "dev_pdr_woG_mean": float(g["pdr_woG"].mean()),
                "dev_pdr_region_ci95": ci95(region_mean),
                "dev_ms_per_decision": float(g["ms_per_decision"].mean()),
                "n_regions": int(g["region"].nunique()),
                "n_episodes": int(len(g)),
            }
        )
    out = fit.merge(pd.DataFrame(rows), on="policy", how="inner", validate="one_to_one")
    if len(out) != 19:
        raise ValueError("fit summary와 개발평가의 후보가 일치하지 않습니다.")
    return out.sort_values("dev_pdr_woG_mean").reset_index(drop=True)


def selective_summary() -> pd.DataFrame:
    path = DISTILL / "selective_dev40_seed9000_9009.csv"
    df = pd.read_csv(path)
    if df[["region", "method", "seed"]].duplicated().any():
        raise ValueError("선택적 개발평가 key 중복이 있습니다.")
    student = df[df["method"] == "STUDENT"][
        ["region", "seed", "pdr_woG"]
    ].rename(columns={"pdr_woG": "student_pdr"})
    rows = []
    for method, g in df.groupby("method", observed=True):
        region_mean = g.groupby("region", observed=True)["pdr_woG"].mean()
        row = {
            "method": method,
            "pdr_woG_mean": float(g["pdr_woG"].mean()),
            "pdr_region_ci95": ci95(region_mean),
            "student_coverage": float(g["student_coverage"].mean()),
            "defer_rate": float(g["defer_rate"].mean()),
            "ms_per_decision": float(g["ms_per_decision"].mean()),
            "n_planner_switched_mean": float(g["n_planner_switched"].mean()),
            "n_regions": int(g["region"].nunique()),
            "n_episodes": int(len(g)),
        }
        if method == "STUDENT":
            row.update(
                {
                    "improvement_vs_student": 0.0,
                    "improvement_episode_ci95": 0.0,
                    "improvement_region_ci95": 0.0,
                }
            )
        else:
            paired = g[["region", "seed", "pdr_woG"]].merge(
                student, on=["region", "seed"], validate="one_to_one"
            )
            delta = paired["student_pdr"] - paired["pdr_woG"]
            region_delta = paired.assign(delta=delta).groupby("region")["delta"].mean()
            row.update(
                {
                    "improvement_vs_student": float(delta.mean()),
                    "improvement_episode_ci95": ci95(delta),
                    "improvement_region_ci95": ci95(region_delta),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pdr_woG_mean").reset_index(drop=True)


def external_scenario_validation() -> dict[str, Any]:
    meta_path = ROOT / "scenarios/manifests/distill_external_test250_meta.json"
    points_path = ROOT / "scenarios/manifests/distill_external_test250_points.json"
    manifest_path = ROOT / "scenarios/manifests/distill_external_test250_osrm_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    points = json.loads(points_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(points) != 250 or len(manifest) != 250:
        raise ValueError("외부 시나리오/좌표가 250개가 아닙니다.")
    coords = [(round(float(v["lat"]), 7), round(float(v["lon"]), 7)) for v in points.values()]
    if len(set(coords)) != 250:
        raise ValueError("외부 좌표에 중복이 있습니다.")
    return {
        "status": "pass",
        "n_points": len(points),
        "n_manifest_entries": len(manifest),
        "n_unique_coordinates": len(set(coords)),
        "known_coordinate_exact_overlap": int(
            meta.get("exact_overlap_with_exclusions", -1)
        ),
        "exclusion_pool_coordinates": int(
            meta.get("n_unique_exclusion_coordinates", -1)
        ),
        "radius_counts": meta.get("radius_counts", {}),
        "manifest_sha256": sha256(manifest_path),
        "points_sha256": sha256(points_path),
        "meta_sha256": sha256(meta_path),
    }


def plot_external(score: pd.DataFrame, out: Path) -> None:
    d = score.sort_values("pdr_woG_mean", ascending=True)
    colors = ["#2F6B9A" if m == "STUDENT" else "#A9B8C5" for m in d["method"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    y = np.arange(len(d))
    ax.barh(
        y,
        d["pdr_woG_mean"],
        xerr=d["pdr_woG_region_ci95"],
        color=colors,
        edgecolor="#334155",
        linewidth=0.7,
        error_kw={"ecolor": "#334155", "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(y, d["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.29, float(d["pdr_woG_mean"].max()) * 1.08))
    ax.set_xlabel("Mean PDR_woG (lower is better)")
    ax.set_title("External 250-coordinate policy comparison", loc="left", pad=30)
    ax.text(
        0,
        1.01,
        "250 regions × 10 paired seeds (10,000–10,009); error bars: 95% CI across region means",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.7)
    ax.set_axisbelow(True)
    for yi, val in zip(y, d["pdr_woG_mean"]):
        ax.text(val + 0.003, yi, f"{val:.4f}", va="center", fontsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_latency(score: pd.DataFrame, out: Path) -> None:
    d = score.copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=160)
    family_colors = {
        "heuristic": "#9AA5B1",
        "student": "#2F6B9A",
        "RL": "#C98B2E",
        "planner": "#7D5BA6",
    }
    for family, g in d.groupby("family", observed=True):
        ax.scatter(
            g["ms_per_decision_mean"],
            g["pdr_woG_mean"],
            s=58,
            color=family_colors[family],
            edgecolor="#273444",
            linewidth=0.7,
            label=family,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean decision latency (ms, log scale)")
    ax.set_ylabel("Mean PDR_woG (lower is better)")
    ax.set_title("External performance and decision latency", loc="left", pad=30)
    ax.text(
        0,
        1.01,
        "Same 250 × 10 external evaluation; latency is Python server time, not edge-device benchmarking",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    for _, r in d.iterrows():
        label = {
            "HEUR64_GLOBAL_TRAIN_BEST": "Heuristic",
            "GLOBAL_TRAIN_BEST_T4": "T4",
            "I3_CONNECTED_EBM_I08": "EBM",
            "I3_CONNECTED_C4": "C4",
            "I3_CONNECTED_CART_L384": "CART-384",
            "PPO": "PPO",
            "STUDENT": "Student",
            "STUDENT_NCRP_C75": "Student+NCRP",
        }[r["method"]]
        offset = {
            "Heuristic": (5, 4),
            "T4": (5, 4),
            "EBM": (5, 12),
            "C4": (5, -10),
            "CART-384": (5, 3),
            "PPO": (5, 5),
            "Student": (5, 3),
            "Student+NCRP": (5, 4),
        }[label]
        ax.annotate(
            label,
            (r["ms_per_decision_mean"], r["pdr_woG_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    ax.grid(color="#D8DEE6", linewidth=0.7)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_development(dev: pd.DataFrame, out: Path) -> None:
    d = dev.sort_values("dev_pdr_woG_mean", ascending=False).copy()
    colors = [
        "#2F6B9A" if p == "I1_FIELD_GBDT_L31_SOFT" else "#A9B8C5"
        for p in d["policy"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=160)
    y = np.arange(len(d))
    ax.scatter(
        d["dev_pdr_woG_mean"],
        y,
        s=44,
        color=colors,
        edgecolor="#334155",
        linewidth=0.6,
        zorder=3,
    )
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.hlines(
            yi,
            r["dev_pdr_woG_mean"] - r["dev_pdr_region_ci95"],
            r["dev_pdr_woG_mean"] + r["dev_pdr_region_ci95"],
            color="#64748B",
            linewidth=0.8,
            zorder=2,
        )
    ax.set_yticks(y, d["policy"].str.replace("_", " ", regex=False))
    ax.set_xlabel("Mean PDR_woG (lower is better)")
    ax.set_title(
        "Development comparison of 19 distilled policy candidates",
        loc="left",
        pad=30,
    )
    ax.text(
        0,
        1.01,
        "40 training-distribution coordinates × 10 paired seeds (8,000–8,009)",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    ax.grid(axis="x", color="#D8DEE6", linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def artifact_payload(
    score: pd.DataFrame,
    pairwise: pd.DataFrame,
    dev: pd.DataFrame,
    selective: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    score_rows = score.replace({np.nan: None}).to_dict(orient="records")
    pair_rows = pairwise.replace({np.nan: None}).to_dict(orient="records")
    dev_rows = (
        dev[
            [
                "policy",
                "info_level",
                "family",
                "complexity",
                "weight_scheme",
                "fidelity_full",
                "dev_pdr_woG_mean",
                "dev_pdr_region_ci95",
                "dev_ms_per_decision",
            ]
        ]
        .replace({np.nan: None})
        .to_dict(orient="records")
    )
    selective_rows = selective.replace({np.nan: None}).to_dict(orient="records")

    student = score.set_index("method").loc["STUDENT"]
    ppo = score.set_index("method").loc["PPO"]
    ncrp = score.set_index("method").loc["STUDENT_NCRP_C75"]
    ppo_pair = pairwise.set_index("comparator").loc["PPO"]
    ncrp_pair = pairwise.set_index("comparator").loc["STUDENT_NCRP_C75"]

    sources = [
        {
            "id": "external_scoreboard",
            "label": "외부 250좌표 통합 scoreboard",
            "path": "results/scoreboard/v10/distill/student_experiments/external_scoreboard.csv",
        },
        {
            "id": "external_pairwise",
            "label": "외부 250좌표 paired 비교",
            "path": "results/scoreboard/v10/distill/student_experiments/external_pairwise_vs_student.csv",
        },
        {
            "id": "development_models",
            "label": "19개 증류후보 개발평가",
            "path": "results/scoreboard/v10/distill/student_experiments/development_model_comparison.csv",
        },
        {
            "id": "selective_deferral",
            "label": "선택적 PPO/NCRP 위임 평가",
            "path": "results/scoreboard/v10/distill/student_experiments/selective_deferral_summary.csv",
        },
        {
            "id": "validation",
            "label": "프로토콜·데이터 정합성 검증",
            "path": "results/scoreboard/v10/distill/student_experiments/validation_report.json",
        },
    ]
    top_sources = [
        {
            "id": source["id"],
            "label": source["label"],
            "path": source["path"],
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_csv_auto('{source['path']}');"
                    if source["path"].endswith(".csv")
                    else f"SELECT * FROM read_json_auto('{source['path']}');"
                ),
                "description": source["label"],
                "executed_at": generated_at,
            },
        }
        for source in sources
    ]

    title = "v10 현장형 증류정책 병렬실험 보고서"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "CART·EBM·GBDT 증류, 선택적 위임, 새 외부 250좌표 검증 결과",
        "generatedAt": generated_at,
        "charts": [
            {
                "id": "external_pdr",
                "title": "외부 250좌표 정책별 PDR_woG",
                "subtitle": "지역 250 × 공통 seed 10; 낮을수록 좋음",
                "type": "horizontalBar",
                "dataset": "external_methods",
                "sourceId": "external_scoreboard",
                "valueFormat": "number",
                "intent": "comparison",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "정책"},
                    "y": {
                        "field": "pdr_woG_mean",
                        "type": "quantitative",
                        "label": "평균 PDR_woG",
                    },
                    "tooltip": [
                        {
                            "field": "pdr_woG_region_ci95",
                            "type": "quantitative",
                            "label": "지역평균 95% CI 반폭",
                        },
                        {
                            "field": "ms_per_decision_mean",
                            "type": "quantitative",
                            "label": "결정 지연(ms)",
                        },
                    ],
                },
                "maxRows": 8,
            },
            {
                "id": "external_latency",
                "title": "외부 성능과 의사결정 지연",
                "subtitle": "x축은 log10(ms); Python 서버 실측",
                "type": "scatter",
                "dataset": "external_methods",
                "sourceId": "external_scoreboard",
                "intent": "relationship",
                "encodings": {
                    "x": {
                        "field": "latency_log10_ms",
                        "type": "quantitative",
                        "label": "log10 결정 지연(ms)",
                    },
                    "y": {
                        "field": "pdr_woG_mean",
                        "type": "quantitative",
                        "label": "평균 PDR_woG",
                    },
                    "color": {
                        "field": "family",
                        "type": "nominal",
                        "label": "정책군",
                    },
                    "label": {"field": "label", "type": "text", "label": "정책"},
                    "tooltip": [
                        {
                            "field": "ms_per_decision_mean",
                            "type": "quantitative",
                            "label": "결정 지연(ms)",
                        }
                    ],
                },
                "maxRows": 8,
            },
            {
                "id": "dev_candidates",
                "title": "19개 증류후보 개발평가",
                "subtitle": "학습분포 40좌표 × 공통 seed 10; 최종 외부평가가 아님",
                "type": "horizontalBar",
                "dataset": "development_models",
                "sourceId": "development_models",
                "valueFormat": "number",
                "intent": "comparison",
                "encodings": {
                    "x": {
                        "field": "policy",
                        "type": "nominal",
                        "label": "후보정책",
                    },
                    "y": {
                        "field": "dev_pdr_woG_mean",
                        "type": "quantitative",
                        "label": "평균 PDR_woG",
                    },
                    "tooltip": [
                        {
                            "field": "fidelity_full",
                            "type": "quantitative",
                            "label": "PPO 행동일치율",
                        },
                        {
                            "field": "dev_ms_per_decision",
                            "type": "quantitative",
                            "label": "결정 지연(ms)",
                        },
                    ],
                },
                "maxRows": 19,
            },
        ],
        "tables": [
            {
                "id": "external_table",
                "title": "외부평가 상세",
                "subtitle": "250좌표 × 10개 seed의 평균 및 지역간 95% CI",
                "dataset": "external_methods",
                "sourceId": "external_scoreboard",
                "defaultSort": {"field": "pdr_woG_mean", "direction": "asc"},
                "columns": [
                    {"field": "rank", "label": "순위", "format": "number"},
                    {"field": "label", "label": "정책", "type": "text"},
                    {"field": "pdr_woG_mean", "label": "PDR_woG", "format": "number"},
                    {
                        "field": "pdr_woG_region_ci95",
                        "label": "지역 95% CI",
                        "format": "number",
                    },
                    {
                        "field": "ms_per_decision_mean",
                        "label": "결정 지연(ms)",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "pairwise_table",
                "title": "학생정책 기준 paired 비교",
                "subtitle": "양의 개선량은 학생정책이 더 낮은 PDR임을 뜻함",
                "dataset": "external_pairwise",
                "sourceId": "external_pairwise",
                "defaultSort": {
                    "field": "student_improvement_pdr",
                    "direction": "desc",
                },
                "columns": [
                    {
                        "field": "comparator_label",
                        "label": "비교정책",
                        "type": "text",
                    },
                    {
                        "field": "student_improvement_pdr",
                        "label": "학생 개선량",
                        "format": "number",
                    },
                    {
                        "field": "region_mean_ci95",
                        "label": "지역 95% CI",
                        "format": "number",
                    },
                    {"field": "student_wins", "label": "승", "format": "number"},
                    {"field": "ties", "label": "무", "format": "number"},
                    {"field": "student_losses", "label": "패", "format": "number"},
                ],
            },
            {
                "id": "selective_table",
                "title": "선택적 위임 개발평가",
                "subtitle": "40좌표 × 10개 seed; C75는 약 46%를 위임",
                "dataset": "selective_deferral",
                "sourceId": "selective_deferral",
                "defaultSort": {"field": "pdr_woG_mean", "direction": "asc"},
                "columns": [
                    {"field": "method", "label": "정책", "type": "text"},
                    {"field": "pdr_woG_mean", "label": "PDR_woG", "format": "number"},
                    {
                        "field": "student_coverage",
                        "label": "학생 coverage",
                        "format": "percent",
                    },
                    {
                        "field": "improvement_vs_student",
                        "label": "학생 대비 개선",
                        "format": "number",
                    },
                    {
                        "field": "ms_per_decision",
                        "label": "결정 지연(ms)",
                        "format": "number",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "summary",
                "type": "markdown",
                "sourceId": "external_scoreboard",
                "body": (
                    "## 결론\n\n"
                    f"현장정보 26개만 쓰는 **I1 GBDT 학생정책**의 새 외부좌표 PDR_woG는 "
                    f"**{student['pdr_woG_mean']:.6f}**로, PPO의 "
                    f"**{ppo['pdr_woG_mean']:.6f}**보다 낮았습니다. "
                    f"차이는 **{ppo_pair['student_improvement_pdr']:.6f}**이며 "
                    f"지역평균 95% CI 반폭은 **{ppo_pair['region_mean_ci95']:.6f}**입니다. "
                    "따라서 이번 실험의 현장 주모델은 I1 GBDT로 채택합니다."
                ),
            },
            {"id": "external_chart", "type": "chart", "chartId": "external_pdr"},
            {"id": "external_table_block", "type": "table", "tableId": "external_table"},
            {
                "id": "distillation",
                "type": "markdown",
                "sourceId": "development_models",
                "body": (
                    "## 증류 구조 비교\n\n"
                    "C4를 384 leaf CART로 줄이면 외부 PDR이 소폭 개선됐지만, "
                    "EBM은 C4보다 악화됐습니다. 가장 큰 개선은 트리 한 그루가 아니라 "
                    "**후보 행동 전체를 점수화하는 GBDT 앙상블**에서 나왔습니다. "
                    "PPO 확률을 부드러운 표본가중치로 사용한 I1 GBDT가 대표점 개발평가에서 "
                    "최종 후보로 선택됐습니다."
                ),
            },
            {"id": "dev_chart", "type": "chart", "chartId": "dev_candidates"},
            {
                "id": "pairwise",
                "type": "markdown",
                "sourceId": "external_pairwise",
                "body": (
                    "## Paired 검정\n\n"
                    f"학생정책 대 PPO의 지역별 결과는 "
                    f"**{int(ppo_pair['student_wins'])}승·{int(ppo_pair['ties'])}무·"
                    f"{int(ppo_pair['student_losses'])}패**였습니다. "
                    "승/무/패는 각 지역의 10개 공통 seed 차이에 대한 95% CI로 판정했습니다."
                ),
            },
            {"id": "pairwise_table_block", "type": "table", "tableId": "pairwise_table"},
            {
                "id": "deferral",
                "type": "markdown",
                "sourceId": "external_scoreboard",
                "body": (
                    "## 선택적 위임\n\n"
                    f"개발셋에서 유망했던 C75 NCRP 위임은 외부평가에서 "
                    f"**{ncrp['pdr_woG_mean']:.6f}**로 학생정책보다 "
                    f"**{abs(ncrp_pair['student_improvement_pdr']):.6f}** 낮았지만, "
                    f"지역평균 95% CI 반폭이 **{ncrp_pair['region_mean_ci95']:.6f}**여서 "
                    "재현 가능한 개선으로 볼 수 없습니다. 계산비용까지 고려해 순수 학생정책을 "
                    "현장 배포 후보로 유지합니다."
                ),
            },
            {"id": "selective_table_block", "type": "table", "tableId": "selective_table"},
            {"id": "latency_chart", "type": "chart", "chartId": "external_latency"},
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "validation",
                "body": (
                    "## 평가 설계\n\n"
                    "- 학습: random4 1,000좌표에서 새로 학습한 PPO seed 0\n"
                    "- 개발/선택: 대표점 250 및 학습분포 40좌표\n"
                    "- 최종시험: 알려진 2,085좌표와 정확히 겹치지 않는 새 좌표 250\n"
                    "- 공통 난수: seed 10,000–10,009, 정책당 2,500 paired episodes\n"
                    "- 제약: 모든 학습·증류·평가에서 동일 hard action mask 유지\n"
                    "- 비교 휴리스틱: 학습 1,000좌표에서 미리 고른 단일 전국 규칙과 그 T4 변형"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 한계와 해석 범위\n\n"
                    "- PPO 학습 seed는 0 하나이므로 학습 seed 불확실성은 아직 남습니다.\n"
                    "- 외부셋은 새로운 좌표지만 같은 시군구 체계·동일 자원구성의 지리적 holdout입니다.\n"
                    "- 지역당 최종 episode가 10개라 지역별 승패 검정력은 제한적입니다.\n"
                    "- EBM은 계산비용 때문에 5,000개 상태 표본으로 학습해 전체 데이터 GBDT와 자원량이 다릅니다.\n"
                    "- 의사결정 지연은 서버 Python 측정값이며 실제 엣지 장비 지연이 아닙니다.\n"
                    "- soft/critical 가중치는 환자 outcome이 아니라 PPO 확률·top1-top2 gap 대리값입니다."
                ),
            },
            {
                "id": "recommendation",
                "type": "markdown",
                "body": (
                    "## 권고\n\n"
                    "1. 논문의 현장모델은 **I1 GBDT 학생정책**으로 고정합니다.\n"
                    "2. PPO는 교사·성능 비교군, NCRP는 별도 계산집약적 상한 옵션으로 구분합니다.\n"
                    "3. 다음 실험은 트리 RL보다 PPO 학습 seed 1·2 반복과 재난규모·자원수 변화 스트레스 시험을 우선합니다.\n"
                    "4. 현장 설명용으로는 GBDT 전체를 단일 규칙으로 과도하게 축약하지 말고 SHAP/부분의존 규칙표를 별도 추출합니다."
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 추가로 답할 질문\n\n"
                    "- 학습 seed를 바꿔도 학생정책의 PPO 추월이 유지되는가?\n"
                    "- 사고규모·AMB/UAV 수·병원 수가 변할 때 I1 정책의 강건성은 유지되는가?\n"
                    "- GBDT의 지역별 이득은 농촌·도서산간 특성과 어떤 관계가 있는가?"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "external_methods": score_rows,
                "external_pairwise": pair_rows,
                "development_models": dev_rows,
                "selective_deferral": selective_rows,
            },
            "accessIssues": [],
        },
        "sources": top_sources,
        "package_info": {
            "originUrl": "artifact://v10-student-experiments",
            "controls": {"edit": False, "refresh": False},
        },
    }


def markdown_report(
    score: pd.DataFrame,
    pairwise: pd.DataFrame,
    dev: pd.DataFrame,
    selective: pd.DataFrame,
    validation: dict[str, Any],
) -> str:
    sm = score.set_index("method")
    pw = pairwise.set_index("comparator")
    student = sm.loc["STUDENT"]
    ppo = sm.loc["PPO"]
    ncrp = sm.loc["STUDENT_NCRP_C75"]
    c4 = sm.loc["I3_CONNECTED_C4"]
    cart = sm.loc["I3_CONNECTED_CART_L384"]
    ebm = sm.loc["I3_CONNECTED_EBM_I08"]
    t4 = sm.loc["GLOBAL_TRAIN_BEST_T4"]
    heur = sm.loc["HEUR64_GLOBAL_TRAIN_BEST"]

    table = score[
        ["rank", "label", "pdr_woG_mean", "pdr_woG_region_ci95", "ms_per_decision_mean"]
    ].copy()
    table.columns = ["순위", "정책", "PDR_woG", "지역 95% CI", "결정 ms"]
    table_md = dataframe_markdown(
        table,
        {
            "순위": ".0f",
            "PDR_woG": ".6f",
            "지역 95% CI": ".6f",
            "결정 ms": ".3f",
        },
    )

    select_cols = [
        "method",
        "pdr_woG_mean",
        "student_coverage",
        "improvement_vs_student",
        "improvement_region_ci95",
        "ms_per_decision",
    ]
    selective_md = dataframe_markdown(
        selective[select_cols],
        {
            "pdr_woG_mean": ".6f",
            "student_coverage": ".6f",
            "improvement_vs_student": ".6f",
            "improvement_region_ci95": ".6f",
            "ms_per_decision": ".3f",
        },
    )

    return f"""# v10 현장형 증류정책 병렬실험 보고서

작성일: 2026-07-26

## 한 줄 결론

트리 자체를 RL로 최적화하기 전에 실시한 병렬 비교에서는 **I1_FIELD GBDT L31 soft 학생정책**이 최종 채택안이다. 새 외부 250좌표에서 PDR_woG **{student.pdr_woG_mean:.6f}**로 PPO **{ppo.pdr_woG_mean:.6f}**를 **{pw.loc['PPO'].student_improvement_pdr:.6f}** 낮췄고, 지역평균 95% CI 반폭은 **{pw.loc['PPO'].region_mean_ci95:.6f}**였다.

## 무엇을 동시에 비교했나

- CART 복잡도: 64·128·256·384 leaf와 기존 C4 512 leaf
- EBM 상호작용 수: 0·4·8
- GBDT 표현력: 15·31·63 leaf
- 정보 수준: 현장형 I1(26개 특징)과 병원연계 I3(43개 특징)
- 표본 가중: 기본·PPO soft probability·top1-top2 gap 대리값
- 선택적 위임: 불확실 결정만 PPO 또는 NCRP m16으로 보냄
- 최종시험: 모델 선택에 쓰지 않은 새 외부 좌표 250 × seed 10

## 최종 외부 scoreboard

{table_md}

학생정책은 단일 전국 휴리스틱({heur.pdr_woG_mean:.6f}), 그 T4 변형({t4.pdr_woG_mean:.6f}), EBM({ebm.pdr_woG_mean:.6f}), 기존 C4({c4.pdr_woG_mean:.6f}), 384-leaf CART({cart.pdr_woG_mean:.6f}), PPO({ppo.pdr_woG_mean:.6f})를 모두 넘었다. PPO 대비 지역별 paired 판정은 **{int(pw.loc['PPO'].student_wins)}승·{int(pw.loc['PPO'].ties)}무·{int(pw.loc['PPO'].student_losses)}패**다.

## 왜 GBDT가 이겼나

단일 CART의 가지를 늘리는 것보다, 각 의사결정에서 유효한 `[class, destination, mode]` 후보를 같은 hard mask 아래 반복적으로 점수화하는 GBDT가 교사의 비선형 경계를 더 잘 근사했다. 최종 I1 모델은 병원 실시간 점유·대기열 없이 현장과 정적 이동정보 중심 26개 특징만 사용한다. PPO 행동일치율은 {float(dev.loc[dev.policy == 'I1_FIELD_GBDT_L31_SOFT', 'fidelity_full'].iloc[0]):.3f}에 불과하지만, 폐루프 PDR은 PPO보다 좋았다. 따라서 단순 모방정확도보다 폐루프 의사결정 품질이 중요하다는 결과다.

기존 C4를 512 leaf에서 384 leaf로 줄인 경우 외부 PDR이 {c4.pdr_woG_mean:.6f}에서 {cart.pdr_woG_mean:.6f}로 소폭 개선됐다. 반면 EBM은 {ebm.pdr_woG_mean:.6f}로 C4보다 나빠, 이번 설정에서는 해석가능 상호작용 모형이 성능을 회복하지 못했다.

## 가중 증류의 해석

I1 GBDT L31 soft는 PPO가 각 후보 행동에 부여한 확률을 부드러운 가중치로 사용했다. `critical2`는 PPO top1-top2 확률차를 중요도 대리값으로 사용했지만, 개발 40좌표에서 soft가 {float(dev.loc[dev.policy == 'I1_FIELD_GBDT_L31_SOFT', 'dev_pdr_woG_mean'].iloc[0]):.6f}, critical2가 {float(dev.loc[dev.policy == 'I1_FIELD_GBDT_L31_CRIT2', 'dev_pdr_woG_mean'].iloc[0]):.6f}로 critical2가 더 나빴다. 이 대리값은 환자 outcome 중요도가 아니므로 논문에서 outcome-weighted distillation로 부르면 안 된다.

## 선택적 PPO/NCRP 위임

{selective_md}

개발셋에서는 C75 NCRP가 학생정책보다 {float(selective.loc[selective.method == 'STUDENT_NCRP_C75', 'improvement_vs_student'].iloc[0]):.6f} 개선됐지만, 외부셋에서는 학생 {student.pdr_woG_mean:.6f} 대 학생+NCRP {ncrp.pdr_woG_mean:.6f}로 차이가 **{abs(pw.loc['STUDENT_NCRP_C75'].student_improvement_pdr):.6f}**, 지역 95% CI 반폭 **{pw.loc['STUDENT_NCRP_C75'].region_mean_ci95:.6f}**였다. 지역 결과도 학생 기준 {int(pw.loc['STUDENT_NCRP_C75'].student_wins)}승·{int(pw.loc['STUDENT_NCRP_C75'].ties)}무·{int(pw.loc['STUDENT_NCRP_C75'].student_losses)}패로 대칭적이다. C75 위임은 결정당 {ncrp.ms_per_decision_mean:.1f}ms가 걸리므로 현장 기본안으로 채택하지 않는다.

## 데이터 누수와 정합성

- 학습: random4 1,000좌표의 새 PPO seed 0
- 개발선택: 대표점 250 및 학습분포 40좌표
- 최종시험: 알려진 {validation['external_scenarios']['exclusion_pool_coordinates']:,}좌표와 정확히 겹치지 않는 250좌표
- 외부평가: seed 10,000–10,009, 정책당 2,500 paired episodes
- 모든 외부 method의 지역·seed 집합 일치, 중복 key 0, PDR 범위 검증 통과
- 학습 1,000좌표에서 미리 선택한 단일 전국 휴리스틱을 외부셋에 고정 적용해 per-coordinate best-rule 누수를 피함

## 학술적으로 무엇을 주장할 수 있나

1. **현장정보 제약이 있는 정책증류**: 병원연계 I3보다 정보가 적은 I1 GBDT가 PPO를 추월했다.
2. **구조 비교의 음성·양성 결과**: 단일 CART 가지 증가와 EBM은 한계가 있었고, masked candidate-ranking GBDT는 유효했다.
3. **배포비용을 포함한 검증**: 선택적 NCRP는 평균 이득이 외부에서 재현되지 않아 계산비용 대비 기각됐다.
4. **좌표 수준 외부검증**: 모델 선택 후 새 250좌표를 한 번만 사용한 clean final test를 확보했다.

다만 “전국 모든 재난에 일반화”라고 말하기에는 이르다. 같은 시군구 체계와 동일 자원구성의 좌표 holdout이므로, 논문 표현은 **unseen-coordinate geographic generalization**이 적절하다.

## 남은 한계와 우선순위

- PPO 학습 seed가 0 하나라 학습 seed 1·2 반복이 필요하다.
- 지역당 외부 episode가 10개라 지역별 승패 검정력은 제한적이다.
- EBM은 계산비용 때문에 5,000개 상태 표본만 사용해 GBDT와 계산예산이 다르다.
- 서버 Python 지연은 실제 현장 엣지 장비 지연을 대신하지 못한다.
- 다음은 트리 RL보다 재난규모·AMB/UAV 수·병원 수 변화 스트레스 시험과 농촌/도서산간 이득 분석을 우선한다.
"""


def chart_map() -> dict[str, Any]:
    return {
        "external_policy_comparison.png": {
            "question": "새 외부 250좌표에서 어떤 정책의 PDR_woG가 가장 낮은가?",
            "family": "comparison/ranking",
            "grain": "method; 250 region means from 10 paired seeds",
            "metric": "mean PDR_woG, lower is better",
            "uncertainty": "95% CI across 250 region means",
            "scale": "absolute zero-based",
        },
        "external_performance_latency.png": {
            "question": "성능과 의사결정 계산비용 사이의 관계는 무엇인가?",
            "family": "relationship",
            "grain": "method",
            "x": "mean decision latency in ms, log scale",
            "y": "mean PDR_woG",
            "caveat": "Python server timing; not edge-device benchmark",
        },
        "development_19_candidates.png": {
            "question": "19개 후보 중 개발셋 폐루프 PDR이 낮은 구조는 무엇인가?",
            "family": "comparison with uncertainty",
            "grain": "candidate policy",
            "metric": "mean PDR_woG over 40 regions × 10 paired seeds",
            "uncertainty": "95% CI across 40 region means",
            "scope": "development only, not final evidence",
        },
    }


def dataframe_markdown(df: pd.DataFrame, formats: dict[str, str]) -> str:
    """tabulate 선택 의존성 없이 작은 결과표를 Markdown으로 변환한다."""
    headers = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rendered = []
        for column in df.columns:
            value = row[column]
            if pd.isna(value):
                rendered.append("")
            elif column in formats:
                rendered.append(format(value, formats[column]))
            else:
                rendered.append(str(value))
        rows.append(rendered)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    return "\n".join(
        [
            line(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(line(row) for row in rows),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    external, source_hashes = load_external()
    external_checks = validate_external(external)
    score = external_scoreboard(external)
    pairwise = external_pairwise(external)
    fit = fit_summary()
    dev = development_summary(fit)
    selective = selective_summary()
    scenario_checks = external_scenario_validation()

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validation = {
        "schema_version": 1,
        "generated_at": generated_at,
        "status": "pass",
        "external_evaluation": external_checks,
        "external_scenarios": scenario_checks,
        "student_candidates": {
            "n_packages": int(len(fit)),
            "n_development_policies": int(len(dev)),
            "n_development_regions": 40,
            "development_seeds": list(range(8000, 8010)),
        },
        "source_sha256": source_hashes,
        "metric_definition": "PDR_woG: Green을 제외한 preventable death rate; 낮을수록 좋음",
        "paired_rule": "지역별 10개 공통 seed 차이의 평균이 ±95% CI를 넘으면 승/패, 아니면 무",
    }

    score.to_csv(out / "external_scoreboard.csv", index=False)
    pairwise.to_csv(out / "external_pairwise_vs_student.csv", index=False)
    fit.to_csv(out / "student_fit_summary.csv", index=False)
    dev.to_csv(out / "development_model_comparison.csv", index=False)
    selective.to_csv(out / "selective_deferral_summary.csv", index=False)
    (out / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "chart_map.json").write_text(
        json.dumps(chart_map(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_external(score, out / "external_policy_comparison.png")
    plot_latency(score, out / "external_performance_latency.png")
    plot_development(dev, out / "development_19_candidates.png")

    artifact = artifact_payload(score, pairwise, dev, selective, generated_at)
    (out / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (out / "v10_student_experiment_report.md").write_text(
        markdown_report(score, pairwise, dev, selective, validation),
        encoding="utf-8",
    )

    print(f"[완료] {out}")
    print(
        score[
            ["rank", "method", "pdr_woG_mean", "pdr_woG_region_ci95", "ms_per_decision_mean"]
        ].to_string(index=False)
    )
    print("\n[학생정책 paired 비교]")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
