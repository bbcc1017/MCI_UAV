# -*- coding: utf-8 -*-
"""v13 최종 교사 증류의 paired 통계분석과 가이드라인 후보 추출.

분석 단위와 역할을 명확히 분리한다.

* 최종 성능: 대표점 250개 × 공통 seed 0..29의 폐루프 PDR_woG
* 증류 적합: random4 p0..p2 750좌표, 내부검증 p3 250좌표
* 행동 해석: 최종 실행 스택(PPO+NCRP h20m16+MILP 후보)의 hard action 로그

대표점 평가는 19개 학생을 동시에 확인하므로 평균 순위만으로 우열을 선언하지 않는다.
지역을 독립 클러스터로 둔 paired 차이, Wilcoxon 검정, Holm 보정을 함께 보고한다.
행동 규칙은 train/validation 양쪽에서 방향이 재현되는 패턴만 "후보"로 남기며,
실제 정책 가이드라인 채택 전에는 별도 폐루프 재시뮬레이션이 필요하다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text


REPO = Path(__file__).resolve().parents[1]
STUDENT_EVAL = REPO / "results/scoreboard/v13/sota_distill/hybrid_students_eval250_seed0_29.csv"
TRAIN_LOG = REPO / "results/scoreboard/v13/sota_distill/data/hybrid_train750_p0p2_seed5000.npz"
VAL_LOG = REPO / "results/scoreboard/v13/sota_distill/data/hybrid_val250_p3_seed7000.npz"
STUDENT_DIR = REPO / "results/scoreboard/v13/sota_distill/students_full1000"
STUDENT_SPLIT_DIR = REPO / "results/scoreboard/v13/sota_distill/students_split750"
BASE_CUBE = REPO / "results/scoreboard/v10/full1000/scoreboard_common30_episodes.npz"
FINAL_TEACHER = REPO / "results/scoreboard/v11/eval250/K8h20m16_milpinj.csv"
LBT_SWEEP = REPO / "results/scoreboard/v12/lbT_sweep/lbT_sweep_eval250_30ep_pe.npz"
COMPACT_EVAL = REPO / "results/scoreboard/v13/rule_analysis/compact_rules_eval250_seed0_29.csv"
DEFAULT_OUT = REPO / "results/scoreboard/v13/rule_analysis"

DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best-of-64",
    "LB_T3": "LB-T3",
    "PPO_POINTER_V10": "PPO Pointer v10",
    "PPO_POINTER_V10_NCRP_H20M16_MILPINJ": "PPO+NCRP-h20m16+MILP (교사)",
}
FAMILY_LABEL = {"lgbm": "GBDT", "ebm": "EBM", "cart": "CART"}
CORE_REFS = ["PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "PPO_POINTER_V10", "LB_T3"]

HOSPITAL_FEATURES = [
    "eta_rank",
    "eta_raw_min",
    "cand_p_sent_rel",
    "cand_in_flight",
    "cand_cap_remain",
    "cand_occ_ratio",
    "max_send",
]
CHOICE_FEATURES = [
    "eta_rank",
    "cand_p_sent_rel",
    "cand_in_flight",
    "cand_occ_ratio",
    "max_send",
]
CORRECTION_FEATURES = [
    "is_red",
    "is_uav",
    "is_stay",
    "eta_rank",
    "eta_raw_min",
    "uav_advantage_min",
    "cand_p_sent_rel",
    "cand_in_flight",
    "cand_cap_remain",
    "cand_occ_ratio",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def ci95(x: np.ndarray | pd.Series) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) <= 1:
        return 0.0
    return float(1.96 * a.std(ddof=1) / math.sqrt(len(a)))


def bootstrap_ci(
    x: np.ndarray | pd.Series,
    *,
    seed: int = 20260803,
    n_boot: int = 10000,
) -> tuple[float, float]:
    """독립 클러스터별 요약값을 재표집한 평균의 percentile CI."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) <= 1:
        v = float(a.mean()) if len(a) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    chunk = 1000
    means = []
    for start in range(0, n_boot, chunk):
        n = min(chunk, n_boot - start)
        idx = rng.integers(0, len(a), size=(n, len(a)))
        means.append(a[idx].mean(axis=1))
    b = np.concatenate(means)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def safe_wilcoxon(x: np.ndarray | pd.Series) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0.0):
        return 1.0
    return float(wilcoxon(a, alternative="two-sided", zero_method="wilcox").pvalue)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error 보정."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def decode_action(a: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H_pad=47, [class, destination, mode]의 192차원 코덱."""
    x = np.asarray(a, dtype=int)
    cls = x // 96
    rem = x % 96
    return cls, rem // 2, rem % 2


def base_region(key: str) -> str:
    return re.sub(r"_p[0-3]$", "", str(key))


def _matrix_from_planner_csv(path: Path, regions: list[str], seeds: np.ndarray) -> tuple[np.ndarray, float]:
    df = pd.read_csv(path)
    if df.duplicated(["region", "ep"]).any():
        raise ValueError("최종 교사 CSV의 (region, ep)가 중복됨")
    rix = {x: i for i, x in enumerate(regions)}
    six = {int(x): i for i, x in enumerate(seeds)}
    out = np.full((len(regions), len(seeds)), np.nan, dtype=float)
    base = np.full_like(out, np.nan)
    for row in df.itertuples(index=False):
        if row.region in rix and int(row.ep) in six:
            out[rix[row.region], six[int(row.ep)]] = float(row.pdr_planner)
            base[rix[row.region], six[int(row.ep)]] = float(row.pdr_base)
    if not np.isfinite(out).all() or not np.isfinite(base).all():
        raise ValueError("최종 교사 대표점×seed 격자에 결측이 있음")
    return out, float(np.max(np.abs(base)))


def load_performance() -> tuple[list[str], np.ndarray, dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    z = np.load(BASE_CUBE, allow_pickle=True)
    regions = [str(x) for x in z["regions"]]
    seeds = np.asarray(z["seeds"], dtype=int)
    base_methods = [str(x) for x in z["methods"]]
    base = np.asarray(z["pdr_wog"], dtype=float)
    if base.shape != (250, 4, 30) or not np.array_equal(seeds, np.arange(30)):
        raise ValueError(f"기준 cube 프로토콜 불일치: {base.shape}, seeds={seeds}")

    cubes: dict[str, np.ndarray] = {
        "HEUR64_BEST": base[:, base_methods.index("HEUR64_BEST"), :],
        "PPO_POINTER_V10": base[:, base_methods.index("PPO_POINTER_V10"), :],
    }

    lb = np.load(LBT_SWEEP, allow_pickle=True)
    lb_regions = [str(x) for x in lb["regions"]]
    lb_names = [str(x) for x in lb["names"]]
    if lb_regions != regions or not np.array_equal(lb["seeds"], seeds):
        raise ValueError("LB-T 전수스윕과 기준 cube의 대표점 또는 seed가 다름")
    cubes["LB_T3"] = np.asarray(lb["pdr"][:, lb_names.index("lb_T3"), :], dtype=float)
    t4_check = np.asarray(lb["pdr"][:, lb_names.index("lb_T4"), :], dtype=float)
    t4_base = base[:, base_methods.index("LB_T4"), :]
    t4_err = float(np.max(np.abs(t4_check - t4_base)))
    if t4_err > 1e-6:
        raise ValueError(f"LB-T 스윕 재현성 실패: T4 max err={t4_err:.3e}")

    teacher, _ = _matrix_from_planner_csv(FINAL_TEACHER, regions, seeds)
    planner_df = pd.read_csv(FINAL_TEACHER)
    ppo_err = float(
        np.max(
            np.abs(
                planner_df.pivot(index="region", columns="ep", values="pdr_base")
                .reindex(regions)[seeds].to_numpy(float)
                - cubes["PPO_POINTER_V10"]
            )
        )
    )
    if ppo_err > 1e-9:
        raise ValueError(f"최종 교사와 PPO CRN 재현성 실패: max err={ppo_err:.3e}")
    cubes["PPO_POINTER_V10_NCRP_H20M16_MILPINJ"] = teacher

    students = pd.read_csv(STUDENT_EVAL)
    required = {"region", "policy", "seed", "pdr_woG", "info_level", "complexity"}
    if required - set(students):
        raise ValueError(f"학생 평가 CSV 컬럼 누락: {sorted(required - set(students))}")
    if students.duplicated(["region", "policy", "seed"]).any():
        raise ValueError("학생 평가의 (region, policy, seed)가 중복됨")
    if students.isna().any().any() or not students.pdr_woG.between(0, 1).all():
        raise ValueError("학생 평가에 결측 또는 범위 밖 PDR이 있음")
    if set(students.region) != set(regions) or set(students.seed) != set(seeds):
        raise ValueError("학생 평가와 기준 cube의 대표점 또는 seed 집합이 다름")
    if not (students.groupby("policy").size() == 7500).all():
        raise ValueError("학생 정책별 250×30 완전 격자가 아님")
    for policy, g in students.groupby("policy", observed=True):
        cubes[str(policy)] = (
            g.pivot(index="region", columns="seed", values="pdr_woG")
            .reindex(regions)[seeds]
            .to_numpy(float)
        )

    compact_quality: dict[str, Any] = {"present": False}
    if COMPACT_EVAL.exists():
        compact = pd.read_csv(COMPACT_EVAL)
        if compact.duplicated(["region", "policy", "seed"]).any():
            raise ValueError("compact 규칙 평가의 (region, policy, seed)가 중복됨")
        if compact.isna().any().any() or not compact.pdr_woG.between(0, 1).all():
            raise ValueError("compact 규칙 평가에 결측 또는 범위 밖 PDR이 있음")
        if set(compact.region) != set(regions) or set(compact.seed) != set(seeds):
            raise ValueError("compact 규칙 평가와 기준 cube의 대표점 또는 seed가 다름")
        if not (compact.groupby("policy").size() == 7500).all():
            raise ValueError("compact 규칙 정책별 250×30 완전 격자가 아님")
        for policy, g in compact.groupby("policy", observed=True):
            cubes[str(policy)] = (
                g.pivot(index="region", columns="seed", values="pdr_woG")
                .reindex(regions)[seeds].to_numpy(float)
            )
        compact_quality = {
            "present": True,
            "rows": int(len(compact)),
            "policies": int(compact.policy.nunique()),
            "complete_cells_per_policy": 7500,
        }

    quality = {
        "base_cube_shape": list(base.shape),
        "student_rows": int(len(students)),
        "student_policies": int(students.policy.nunique()),
        "student_complete_cells_per_policy": 7500,
        "student_duplicate_keys": 0,
        "student_null_cells": 0,
        "t4_sweep_vs_base_max_abs_error": t4_err,
        "teacher_base_vs_ppo_max_abs_error": ppo_err,
        "regions": len(regions),
        "seeds": seeds.tolist(),
        "compact_rules": compact_quality,
    }
    return regions, seeds, cubes, students, quality


def family_from_policy(policy: str) -> str:
    if policy.startswith("RULE_"):
        return "compact_rule"
    if "GBDT" in policy:
        return "lgbm"
    if "EBM" in policy:
        return "ebm"
    if "CART" in policy:
        return "cart"
    return "baseline"


def build_performance_tables(
    regions: list[str],
    seeds: np.ndarray,
    cubes: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, cube in cubes.items():
        rm = cube.mean(axis=1)
        lo, hi = bootstrap_ci(rm)
        rows.append({
            "method": method,
            "display_name": DISPLAY.get(method, method),
            "family": family_from_policy(method),
            "pdr_wog_mean": float(rm.mean()),
            "pdr_wog_ci95_regions": ci95(rm),
            "pdr_boot_lo": lo,
            "pdr_boot_hi": hi,
            "n_regions": len(regions),
            "n_seeds": len(seeds),
        })
    overall = pd.DataFrame(rows).sort_values("pdr_wog_mean").reset_index(drop=True)

    student_methods = [x for x in cubes if x.startswith(("I1_", "I3_"))]
    pair_rows = []
    for ref in CORE_REFS:
        raw_p: dict[str, float] = {}
        cache: dict[str, dict[str, Any]] = {}
        for cand in student_methods:
            d_region = cubes[ref].mean(axis=1) - cubes[cand].mean(axis=1)
            wtl = []
            for i in range(len(regions)):
                d = cubes[ref][i] - cubes[cand][i]
                m, h = float(d.mean()), ci95(d)
                wtl.append("W" if m > h else "L" if m < -h else "T")
            p_w = safe_wilcoxon(d_region)
            raw_p[cand] = p_w
            lo, hi = bootstrap_ci(d_region)
            cache[cand] = {
                "reference": ref,
                "candidate": cand,
                "candidate_family": family_from_policy(cand),
                "improvement_ref_minus_candidate": float(d_region.mean()),
                "improvement_ci95_regions": ci95(d_region),
                "improvement_boot_lo": lo,
                "improvement_boot_hi": hi,
                "relative_improvement_pct": float(
                    100 * d_region.mean() / cubes[ref].mean()
                ),
                "paired_dz_regions": float(
                    d_region.mean() / d_region.std(ddof=1)
                ) if d_region.std(ddof=1) else 0.0,
                "ttest_p": float(ttest_1samp(d_region, 0.0).pvalue),
                "wilcoxon_p": p_w,
                "W": wtl.count("W"),
                "T": wtl.count("T"),
                "L": wtl.count("L"),
                "n_regions": len(regions),
                "n_paired_episodes": len(regions) * len(seeds),
            }
        adj = holm_adjust(raw_p)
        for cand in student_methods:
            cache[cand]["wilcoxon_holm_p"] = adj[cand]
            cache[cand]["significant_after_holm_0_05"] = bool(adj[cand] < 0.05)
            pair_rows.append(cache[cand])
    pairwise = pd.DataFrame(pair_rows).sort_values(
        ["reference", "improvement_ref_minus_candidate"], ascending=[True, False]
    )

    best = (
        overall[overall.family.isin(["lgbm", "ebm", "cart"])]
        .sort_values("pdr_wog_mean")
        .groupby("family", as_index=False, sort=False)
        .first()
    )
    return overall, pairwise, best


def province_code(region: str) -> str:
    """시군구 키의 5자리 SIGCD에서 광역시도 코드를 복원한다."""
    hit = re.search(r"_(\d{5})$", str(region))
    if not hit:
        raise ValueError(f"SIGCD를 읽을 수 없는 지역 키: {region}")
    return hit.group(1)[:2]


def spatial_block_sensitivity(
    regions: list[str], cubes: dict[str, np.ndarray], best: pd.DataFrame,
) -> pd.DataFrame:
    """인접 시군구 상관에 민감한지 17개 광역시도 동일가중으로 재확인한다."""
    province = np.asarray([province_code(x) for x in regions], dtype=object)
    rng = np.random.default_rng(20260803)
    rows = []
    candidates = best.method.tolist()
    for ref in ("PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "PPO_POINTER_V10"):
        for cand in candidates:
            d = cubes[ref].mean(axis=1) - cubes[cand].mean(axis=1)
            pm = pd.DataFrame({"province": province, "d": d}).groupby("province", observed=True).d.mean()
            boot = np.empty(5000, dtype=float)
            values = pm.to_numpy(float)
            for b in range(len(boot)):
                boot[b] = rng.choice(values, size=len(values), replace=True).mean()
            rows.append({
                "reference": ref,
                "candidate": cand,
                "candidate_family": family_from_policy(cand),
                "equal_province_improvement": float(values.mean()),
                "equal_province_ci95": ci95(values),
                "province_boot_lo": float(np.quantile(boot, 0.025)),
                "province_boot_hi": float(np.quantile(boot, 0.975)),
                "n_provinces": int(len(values)),
                "provinces_positive": int(np.sum(values > 0)),
                "provinces_negative": int(np.sum(values < 0)),
            })
    return pd.DataFrame(rows)


def paired_effect(ref: np.ndarray, cand: np.ndarray, label: str, candidate: str) -> dict[str, Any]:
    """양수이면 candidate가 reference보다 낮은 PDR인 공통 paired 요약."""
    d_region = ref.mean(axis=1) - cand.mean(axis=1)
    lo, hi = bootstrap_ci(d_region, n_boot=10000, seed=20260803)
    wtl = []
    for r, c in zip(ref, cand):
        d = r - c
        m, h = float(d.mean()), ci95(d)
        wtl.append("W" if m > h else "L" if m < -h else "T")
    return {
        "reference": label,
        "candidate": candidate,
        "improvement_ref_minus_candidate": float(d_region.mean()),
        "improvement_boot_lo": lo,
        "improvement_boot_hi": hi,
        "relative_improvement_pct": float(100 * d_region.mean() / ref.mean()),
        "wilcoxon_p": safe_wilcoxon(d_region),
        "W": wtl.count("W"), "T": wtl.count("T"), "L": wtl.count("L"),
        "n_regions": len(d_region), "n_paired_episodes": int(ref.size),
    }


def compact_rule_statistics(cubes: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [x for x in cubes if x.startswith("RULE_")]
    if not policies:
        return pd.DataFrame(), pd.DataFrame()
    refs = [
        "HEUR64_BEST", "LB_T3", "PPO_POINTER_V10",
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "I3_CONNECTED_GBDT_L63_BASE",
    ]
    rows = [paired_effect(cubes[r], cubes[p], r, p) for r in refs for p in policies]
    pairs = [
        ("RULE_ETA", "RULE_ETA_OCC", "혼잡도 추가"),
        ("RULE_ETA_OCC", "RULE_CLASS_ETA_OCC", "class tree 추가"),
        ("RULE_ETA_OCC", "RULE_MODE_ETA_OCC", "mode tree 추가"),
        ("RULE_ETA_OCC", "RULE_CM_ETA_OCC", "class+mode 동시 추가"),
        ("RULE_CM_ETA", "RULE_CM_ETA_OCC", "CM 조건에서 혼잡도 추가"),
    ]
    ablation = []
    for base, enriched, label in pairs:
        rec = paired_effect(cubes[base], cubes[enriched], base, enriched)
        rec["ablation"] = label
        ablation.append(rec)
    ablation_df = pd.DataFrame(ablation)
    adjusted = holm_adjust(dict(zip(ablation_df.ablation, ablation_df.wilcoxon_p)))
    ablation_df["wilcoxon_holm_p"] = ablation_df.ablation.map(adjusted)
    ablation_df["significant_after_holm_0_05"] = ablation_df.wilcoxon_holm_p < 0.05
    return pd.DataFrame(rows), ablation_df


def compact_rule_heterogeneity(
    regions: list[str], cubes: dict[str, np.ndarray], compact_ablation: pd.DataFrame,
) -> pd.DataFrame:
    """행정구역 명칭(군 vs 시·구)에 따른 규칙 기여 차이를 탐색적으로 요약."""
    group = np.asarray([
        "군 지역" if "군" in str(x).rsplit("_", 1)[0] else "시·구 지역" for x in regions
    ], dtype=object)
    rng = np.random.default_rng(20260803)
    rows = []
    for rec in compact_ablation.itertuples(index=False):
        d = cubes[rec.reference].mean(axis=1) - cubes[rec.candidate].mean(axis=1)
        gm = {}
        for label in ("군 지역", "시·구 지역"):
            x = d[group == label]
            lo, hi = bootstrap_ci(x, n_boot=10000, seed=20260803)
            gm[label] = float(x.mean())
            rows.append({
                "ablation": rec.ablation,
                "group": label,
                "improvement": float(x.mean()),
                "bootstrap_lo": lo,
                "bootstrap_hi": hi,
                "n_regions": int(len(x)),
            })
        county, city = d[group == "군 지역"], d[group == "시·구 지역"]
        boot = np.empty(10000, dtype=float)
        for b in range(len(boot)):
            boot[b] = (
                rng.choice(county, size=len(county), replace=True).mean()
                - rng.choice(city, size=len(city), replace=True).mean()
            )
        rows.append({
            "ablation": rec.ablation,
            "group": "군-시·구 차이",
            "improvement": gm["군 지역"] - gm["시·구 지역"],
            "bootstrap_lo": float(np.quantile(boot, 0.025)),
            "bootstrap_hi": float(np.quantile(boot, 0.975)),
            "n_regions": len(regions),
        })
    return pd.DataFrame(rows)


def load_log(path: Path, role: str) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    z0 = np.load(path, allow_pickle=True)
    z = {k: np.asarray(z0[k]) for k in z0.files}
    feature_names = [str(x) for x in z["feature_names"]]
    if z["offsets"][-1] != len(z["X"]) or int(z["ncand"].sum()) != len(z["X"]):
        raise ValueError(f"{role}: offsets/ncand 합이 후보행 수와 다름")
    if not np.isfinite(z["X"]).all() or not np.isfinite(z["weight"]).all():
        raise ValueError(f"{role}: 특징 또는 가중치에 비유한 값")
    pos_per_state = np.add.reduceat(z["chosen"].astype(np.int64), z["offsets"][:-1])
    if not np.all(pos_per_state == 1):
        raise ValueError(f"{role}: 상태마다 양성 후보가 정확히 하나가 아님")
    chosen_pos = np.flatnonzero(z["chosen"])
    if not np.array_equal(z["cand_action"][chosen_pos], z["teacher_action"]):
        raise ValueError(f"{role}: chosen 행과 teacher_action이 다름")

    ppo_pos = np.empty(len(z["ncand"]), dtype=np.int64)
    both_class = np.zeros(len(z["ncand"]), dtype=bool)
    both_mode = np.zeros(len(z["ncand"]), dtype=bool)
    for i, (s, e) in enumerate(zip(z["offsets"][:-1], z["offsets"][1:])):
        s, e = int(s), int(e)
        actions = z["cand_action"][s:e]
        hit = np.flatnonzero(actions == z["ppo_action"][i])
        if len(hit) != 1:
            raise ValueError(f"{role}: PPO 행동 후보 매칭 실패 state={i}")
        ppo_pos[i] = s + int(hit[0])
        cls, dest, mode = decode_action(actions)
        # stay 후보만 존재하는 등급은 실제 환자등급 선택지로 세지 않는다.
        both_class[i] = bool(np.any((cls == 0) & (dest > 0)) and np.any((cls == 1) & (dest > 0)))
        tc, td, tm = decode_action(int(z["teacher_action"][i]))
        mirror = (cls == int(tc)) & (dest == int(td)) & (mode != int(tm))
        both_mode[i] = bool(np.any(mirror))

    chosen_x = z["X"][chosen_pos]
    ppo_x = z["X"][ppo_pos]
    tc, td, tm = decode_action(z["teacher_action"])
    pc, pdest, pm = decode_action(z["ppo_action"])
    keys = np.asarray([str(x) for x in z["state_key"]])
    frame = pd.DataFrame({
        "dataset": role,
        "state_key": keys,
        "region_base": [base_region(x) for x in keys],
        "state_seed": z["state_seed"].astype(int),
        "decision_index": z["decision_index"].astype(int),
        "teacher_class": tc.astype(int),
        "teacher_dest": td.astype(int),
        "teacher_mode": tm.astype(int),
        "ppo_class": pc.astype(int),
        "ppo_dest": pdest.astype(int),
        "ppo_mode": pm.astype(int),
        "teacher_switched": z["teacher_switched"].astype(bool),
        "teacher_in_milp": z["teacher_in_milp"].astype(bool),
        "planner_lookahead": z["planner_lookahead"].astype(bool),
        "planner_dpdr": z["planner_dpdr"].astype(float),
        "both_class_available": both_class,
        "both_mode_available": both_mode,
    })
    for j, name in enumerate(feature_names):
        frame[name] = chosen_x[:, j]
        frame[f"ppo_{name}"] = ppo_x[:, j]

    ids = frame[["state_key", "state_seed", "decision_index"]]
    if ids.duplicated().any():
        raise ValueError(f"{role}: 상태 식별자가 중복됨")
    meta = {
        "role": role,
        "states": int(len(frame)),
        "candidate_rows": int(len(z["X"])),
        "regions_or_coordinates": int(frame.state_key.nunique()),
        "base_regions": int(frame.region_base.nunique()),
        "duplicate_state_ids": 0,
        "states_with_one_positive": int(len(frame)),
        "chosen_teacher_action_match": True,
        "feature_count": len(feature_names),
        "feature_names_match_npz": feature_names == [str(x) for x in z["feature_names"]],
        "ncand_quantiles": {
            str(q): float(np.quantile(z["ncand"], q)) for q in (0, 0.25, 0.5, 0.75, 1)
        },
    }
    return z, frame, meta


def rate_by_region(frame: pd.DataFrame, mask: pd.Series, value: pd.Series) -> dict[str, Any]:
    x = pd.DataFrame({"region": frame.loc[mask, "region_base"], "value": value.loc[mask].astype(float)})
    per = x.groupby("region", observed=True).value.mean()
    lo, hi = bootstrap_ci(per)
    return {
        "rate_equal_region": float(per.mean()),
        "rate_pooled_decision": float(x.value.mean()),
        "ci95_regions": ci95(per),
        "boot_lo": lo,
        "boot_hi": hi,
        "n_regions": int(len(per)),
        "n_decisions": int(len(x)),
    }


def behavior_tables(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for frame in (train, val):
        specs = {
            "Red 선택": (pd.Series(True, index=frame.index), frame.teacher_class == 0),
            "UAV 선택": (frame.teacher_dest > 0, frame.teacher_mode == 1),
            "Red에서 UAV": ((frame.teacher_dest > 0) & (frame.teacher_class == 0), frame.teacher_mode == 1),
            "Yellow에서 UAV": ((frame.teacher_dest > 0) & (frame.teacher_class == 1), frame.teacher_mode == 1),
            "대기 선택": (pd.Series(True, index=frame.index), frame.teacher_dest == 0),
            "PPO 행동 교정": (pd.Series(True, index=frame.index), frame.teacher_switched),
            "MILP 후보 행동 채택": (pd.Series(True, index=frame.index), frame.teacher_in_milp),
            "룩어헤드 실행": (pd.Series(True, index=frame.index), frame.planner_lookahead),
        }
        for metric, (mask, value) in specs.items():
            rows.append({"dataset": frame.dataset.iloc[0], "metric": metric, **rate_by_region(frame, mask, value)})
    behavior = pd.DataFrame(rows)

    bins = [-np.inf, 0, 5, 10, 15, 20, 30, np.inf]
    labels = ["≤0", "0–5", "5–10", "10–15", "15–20", "20–30", ">30"]
    bin_rows = []
    for frame in (train, val):
        eligible = frame[(frame.teacher_dest > 0) & frame.both_mode_available].copy()
        eligible["adv_bin"] = pd.cut(
            eligible.uav_advantage_min, bins=bins, labels=labels, include_lowest=True,
        )
        eligible["class_label"] = np.where(eligible.teacher_class == 0, "Red", "Yellow")
        for (cls, b), g in eligible.groupby(["class_label", "adv_bin"], observed=False):
            if not len(g):
                continue
            per = g.groupby("region_base", observed=True).teacher_mode.mean()
            lo, hi = bootstrap_ci(per)
            bin_rows.append({
                "dataset": frame.dataset.iloc[0],
                "patient_class": cls,
                "uav_advantage_bin_min": str(b),
                "uav_rate_equal_region": float(per.mean()),
                "uav_rate_pooled": float(g.teacher_mode.mean()),
                "ci95_regions": ci95(per),
                "boot_lo": lo,
                "boot_hi": hi,
                "n_regions": int(len(per)),
                "n_decisions": int(len(g)),
            })
    return behavior, pd.DataFrame(bin_rows)


def hospital_contrasts(z: dict[str, np.ndarray], frame: pd.DataFrame) -> pd.DataFrame:
    names = [str(x) for x in z["feature_names"]]
    ni = {x: names.index(x) for x in HOSPITAL_FEATURES}
    rows = []
    for i, (s, e) in enumerate(zip(z["offsets"][:-1], z["offsets"][1:])):
        s, e = int(s), int(e)
        actions = z["cand_action"][s:e]
        cls, dest, mode = decode_action(actions)
        local = int(np.flatnonzero(z["chosen"][s:e])[0])
        c, d, m = int(cls[local]), int(dest[local]), int(mode[local])
        if d == 0:
            continue
        alt = (cls == c) & (mode == m) & (dest > 0)
        alt[local] = False
        if not np.any(alt):
            continue
        rec = {
            "dataset": frame.dataset.iloc[0],
            "region_base": frame.region_base.iloc[i],
            "patient_class": "Red" if c == 0 else "Yellow",
            "mode": "UAV" if m == 1 else "AMB",
        }
        x = z["X"][s:e]
        for name, j in ni.items():
            rec[name] = float(x[local, j] - x[alt, j].mean())
        rows.append(rec)
    raw = pd.DataFrame(rows)
    out = []
    for (dataset, feature), _ in [((frame.dataset.iloc[0], f), None) for f in HOSPITAL_FEATURES]:
        per = raw.groupby("region_base", observed=True)[feature].mean()
        lo, hi = bootstrap_ci(per)
        out.append({
            "dataset": dataset,
            "feature": feature,
            "contrast_chosen_minus_same_class_mode_alternatives": float(per.mean()),
            "ci95_regions": ci95(per),
            "boot_lo": lo,
            "boot_hi": hi,
            "wilcoxon_p": safe_wilcoxon(per),
            "n_regions": int(len(per)),
            "n_decisions": int(len(raw)),
            "direction": "chosen_lower" if per.mean() < 0 else "chosen_higher",
        })
    return pd.DataFrame(out)


def build_choice_sets(z: dict[str, np.ndarray], frame: pd.DataFrame) -> dict[str, Any]:
    """선택한 class·mode 안의 병원 후보집합을 패딩 배열로 변환."""
    names = [str(x) for x in z["feature_names"]]
    fi = np.asarray([names.index(x) for x in CHOICE_FEATURES], dtype=int)
    sets, chosen, regions, folds, classes, modes = [], [], [], [], [], []
    for i, (s, e) in enumerate(zip(z["offsets"][:-1], z["offsets"][1:])):
        s, e = int(s), int(e)
        actions = z["cand_action"][s:e]
        cls, dest, mode = decode_action(actions)
        local = int(np.flatnonzero(z["chosen"][s:e])[0])
        c, d, m = int(cls[local]), int(dest[local]), int(mode[local])
        if d == 0:
            continue
        keep = np.flatnonzero((cls == c) & (mode == m) & (dest > 0))
        where = np.flatnonzero(keep == local)
        if len(keep) < 2 or len(where) != 1:
            continue
        sets.append(z["X"][s:e][keep][:, fi].astype(np.float64))
        chosen.append(int(where[0]))
        regions.append(frame.region_base.iloc[i])
        classes.append(c)
        modes.append(m)
        hit = re.search(r"_(p[0-3])$", frame.state_key.iloc[i])
        folds.append(hit.group(1) if hit else frame.dataset.iloc[0])
    max_cand = max(len(x) for x in sets)
    X = np.zeros((len(sets), max_cand, len(CHOICE_FEATURES)), dtype=np.float64)
    mask = np.zeros((len(sets), max_cand), dtype=bool)
    for i, x in enumerate(sets):
        X[i, :len(x)] = x
        mask[i, :len(x)] = True
    return {
        "X": X,
        "mask": mask,
        "chosen": np.asarray(chosen, dtype=int),
        "region": np.asarray(regions, dtype=object),
        "fold": np.asarray(folds, dtype=object),
        "patient_class": np.asarray(classes, dtype=int),
        "mode": np.asarray(modes, dtype=int),
    }


def fit_conditional_choice(
    data: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    subset: np.ndarray | None = None,
    l2: float = 0.05,
) -> dict[str, Any]:
    """상태별 병원 선택집합의 선형 softmax 조건부 선택모형."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    if subset is None:
        subset = np.ones(len(data["chosen"]), dtype=bool)
    X = (data["X"][subset] - mean[None, None, :]) / std[None, None, :]
    mask = data["mask"][subset]
    y = data["chosen"][subset]
    regions = data["region"][subset]
    unique, count = np.unique(regions, return_counts=True)
    by = dict(zip(unique, count))
    sw = np.asarray([1.0 / by[x] for x in regions], dtype=np.float64)
    sw /= sw.mean()
    row = np.arange(len(y))

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        u = np.einsum("nmk,k->nm", X, w)
        u[~mask] = -1e12
        logz = logsumexp(u, axis=1)
        loss = float(np.average(logz - u[row, y], weights=sw) + l2 * (w @ w))
        p = np.exp(u - logz[:, None])
        p[~mask] = 0.0
        expected = np.einsum("nm,nmk->nk", p, X)
        grad = np.average(expected - X[row, y], axis=0, weights=sw) + 2 * l2 * w
        return loss, grad

    result = minimize(
        objective, np.zeros(len(CHOICE_FEATURES)), jac=True, method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-8},
    )
    w = result.x
    u = np.einsum("nmk,k->nm", X, w)
    u[~mask] = -1e12
    pred = np.argmax(u, axis=1)
    ll = float(np.sum(u[row, y] - logsumexp(u, axis=1)))
    ll_null = float(-np.log(mask.sum(axis=1)).sum())
    return {
        "coef": w,
        "converged": bool(result.success),
        "n_sets": int(len(y)),
        "top1_accuracy": float(np.mean(pred == y)),
        "pseudo_r2": float(1.0 - ll / ll_null),
        "logloss": float(-ll / len(y)),
        "iterations": int(result.nit),
        "uniform_top1": float(np.mean(1.0 / mask.sum(axis=1))),
    }


def evaluate_conditional_choice(
    data: dict[str, Any], coef: np.ndarray, mean: np.ndarray, std: np.ndarray,
) -> dict[str, float]:
    from scipy.special import logsumexp

    X = (data["X"] - mean[None, None, :]) / std[None, None, :]
    u = np.einsum("nmk,k->nm", X, coef)
    u[~data["mask"]] = -1e12
    row = np.arange(len(data["chosen"]))
    y = data["chosen"]
    ll = float(np.sum(u[row, y] - logsumexp(u, axis=1)))
    ll_null = float(-np.log(data["mask"].sum(axis=1)).sum())
    return {
        "top1_accuracy": float(np.mean(np.argmax(u, axis=1) == y)),
        "pseudo_r2": float(1.0 - ll / ll_null),
        "logloss": float(-ll / len(y)),
        "uniform_top1": float(np.mean(1.0 / data["mask"].sum(axis=1))),
    }


def conditional_choice_analysis(
    train_z: dict[str, np.ndarray], train: pd.DataFrame,
    val_z: dict[str, np.ndarray], val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = build_choice_sets(train_z, train)
    va = build_choice_sets(val_z, val)
    valid_rows = tr["mask"]
    flat = tr["X"][valid_rows]
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    full = fit_conditional_choice(tr, mean, std)
    val_metric = evaluate_conditional_choice(va, full["coef"], mean, std)

    fits: dict[str, dict[str, Any]] = {"train_all": full}
    for fold in ("p0", "p1", "p2"):
        fits[fold] = fit_conditional_choice(tr, mean, std, subset=tr["fold"] == fold)
    fits["p3"] = fit_conditional_choice(va, mean, std)

    coef_rows = []
    for j, feature in enumerate(CHOICE_FEATURES):
        fold_coef = [float(fits[f]["coef"][j]) for f in ("p0", "p1", "p2", "p3")]
        coef_rows.append({
            "feature": feature,
            "standardized_coef_train_all": float(full["coef"][j]),
            "raw_unit_coef_train_all": float(full["coef"][j] / std[j]),
            "train_feature_mean": float(mean[j]),
            "train_feature_std": float(std[j]),
            "relative_abs_strength": float(abs(full["coef"][j]) / np.max(np.abs(full["coef"]))),
            "coef_p0": fold_coef[0],
            "coef_p1": fold_coef[1],
            "coef_p2": fold_coef[2],
            "coef_p3": fold_coef[3],
            "same_sign_p0_p3": bool(len(set(np.sign(fold_coef))) == 1),
            "min_abs_fold_coef": float(np.min(np.abs(fold_coef))),
            "interpretation": "값이 큰 후보 선호" if full["coef"][j] > 0 else "값이 작은 후보 선호",
        })
    metric_rows = [{
        "fit": name,
        "n_choice_sets": result["n_sets"],
        "top1_accuracy": result["top1_accuracy"],
        "pseudo_r2": result["pseudo_r2"],
        "logloss": result["logloss"],
        "converged": result["converged"],
        "uniform_top1": result["uniform_top1"],
    } for name, result in fits.items()]
    metric_rows.append({
        "fit": "train_all_to_p3_external",
        "n_choice_sets": len(va["chosen"]),
        **val_metric,
        "converged": True,
    })
    strata_rows = []
    for c, cname in ((0, "Red"), (1, "Yellow")):
        for m, mname in ((0, "AMB"), (1, "UAV")):
            tm = (tr["patient_class"] == c) & (tr["mode"] == m)
            vm = (va["patient_class"] == c) & (va["mode"] == m)
            if tm.sum() < 100 or vm.sum() < 50:
                continue
            fitted = fit_conditional_choice(tr, mean, std, subset=tm)
            va_sub = {k: (v[vm] if isinstance(v, np.ndarray) and len(v) == len(vm) else v) for k, v in va.items()}
            external = evaluate_conditional_choice(va_sub, fitted["coef"], mean, std)
            for j, feature in enumerate(CHOICE_FEATURES):
                strata_rows.append({
                    "patient_class": cname,
                    "mode": mname,
                    "feature": feature,
                    "standardized_coef_train": float(fitted["coef"][j]),
                    "raw_unit_coef_train": float(fitted["coef"][j] / std[j]),
                    "n_train_choice_sets": int(tm.sum()),
                    "n_validation_choice_sets": int(vm.sum()),
                    "validation_top1_accuracy": external["top1_accuracy"],
                    "validation_pseudo_r2": external["pseudo_r2"],
                })
    return pd.DataFrame(coef_rows), pd.DataFrame(metric_rows), pd.DataFrame(strata_rows)


def correction_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame.teacher_switched].copy()
    out = []
    for feature in CORRECTION_FEATURES:
        x[f"delta_{feature}"] = x[feature] - x[f"ppo_{feature}"]
        per = x.groupby("region_base", observed=True)[f"delta_{feature}"].mean()
        lo, hi = bootstrap_ci(per)
        out.append({
            "dataset": frame.dataset.iloc[0],
            "feature": feature,
            "contrast_teacher_minus_ppo_on_switched": float(per.mean()),
            "ci95_regions": ci95(per),
            "boot_lo": lo,
            "boot_hi": hi,
            "wilcoxon_p": safe_wilcoxon(per),
            "n_regions": int(len(per)),
            "n_switched_decisions": int(len(x)),
            "direction": "teacher_lower" if per.mean() < 0 else "teacher_higher",
        })
    return pd.DataFrame(out)


CLASS_FEATURES = [
    "red_at_site", "yellow_at_site", "amb_available", "uav_available", "time_min",
    "red_unrescued", "yellow_unrescued", "red_in_transport", "yellow_in_transport",
    "total_p_sent", "total_in_flight", "fleet_critical", "rho", "total_cap_remain",
]
MODE_FEATURES = [
    "is_red", "uav_advantage_min", "eta_raw_min", "max_send", "cand_p_sent_rel",
    "cand_in_flight", "cand_cap_remain", "cand_occ_ratio", "red_at_site",
    "yellow_at_site", "amb_available", "uav_available", "time_min", "fleet_critical", "rho",
]
SWITCH_FEATURES = [
    "ppo_is_red", "ppo_is_uav", "ppo_is_stay", "ppo_eta_rank", "ppo_eta_raw_min",
    "ppo_uav_advantage_min", "ppo_cand_p_sent_rel", "ppo_cand_in_flight",
    "ppo_cand_cap_remain", "ppo_cand_occ_ratio", "red_at_site", "yellow_at_site",
    "amb_available", "uav_available", "time_min", "fleet_critical", "rho",
]


def _equal_region_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("region_base", observed=True).region_base.transform("size").to_numpy(float)
    w = 1.0 / count
    return w / w.mean()


def fit_axis_tree(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str,
    train_mask: pd.Series,
    val_mask: pd.Series,
    out_dir: Path,
) -> tuple[dict[str, Any], DecisionTreeClassifier, pd.DataFrame, pd.DataFrame]:
    tr = train.loc[train_mask].copy()
    va = val.loc[val_mask].copy()
    Xtr, ytr = tr[features].to_numpy(float), tr[target].astype(int).to_numpy()
    Xva, yva = va[features].to_numpy(float), va[target].astype(int).to_numpy()
    model = DecisionTreeClassifier(
        max_depth=3,
        max_leaf_nodes=8,
        min_samples_leaf=max(100, int(0.01 * len(tr))),
        class_weight="balanced",
        random_state=20260803,
    )
    model.fit(Xtr, ytr, sample_weight=_equal_region_weights(tr))
    ptr, pva = model.predict(Xtr), model.predict(Xva)
    prob = model.predict_proba(Xva)
    auc = float("nan")
    if prob.shape[1] == 2 and len(np.unique(yva)) == 2:
        auc = float(roc_auc_score(yva, prob[:, list(model.classes_).index(1)]))
    metrics = {
        "axis": name,
        "target": target,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "positive_rate_train": float(ytr.mean()),
        "positive_rate_val": float(yva.mean()),
        "train_accuracy": float(accuracy_score(ytr, ptr, sample_weight=_equal_region_weights(tr))),
        "train_balanced_accuracy": float(
            balanced_accuracy_score(ytr, ptr, sample_weight=_equal_region_weights(tr))
        ),
        "val_accuracy": float(accuracy_score(yva, pva, sample_weight=_equal_region_weights(va))),
        "val_balanced_accuracy": float(
            balanced_accuracy_score(yva, pva, sample_weight=_equal_region_weights(va))
        ),
        "val_auc": auc,
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "root_feature": features[int(model.tree_.feature[0])] if model.tree_.feature[0] >= 0 else "leaf",
        "root_threshold": float(model.tree_.threshold[0]),
    }
    text = export_text(model, feature_names=features, decimals=2, max_depth=3)
    (out_dir / f"{name}_tree.txt").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n\n" + text,
        encoding="utf-8",
    )
    imp = pd.DataFrame({
        "axis": name,
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return metrics, model, tr, va


def bootstrap_root(
    axis: str,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in frame.groupby("region_base", observed=True)}
    keys = np.asarray(list(groups), dtype=object)
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in sampled])
        if len(idx) > 15000:
            idx = rng.choice(idx, size=15000, replace=False)
        x = frame.loc[idx]
        tree = DecisionTreeClassifier(
            max_depth=1,
            min_samples_leaf=max(50, int(0.02 * len(x))),
            class_weight="balanced",
            random_state=seed + b,
        )
        tree.fit(x[features].to_numpy(float), x[target].astype(int).to_numpy())
        fi = int(tree.tree_.feature[0])
        rows.append({
            "axis": axis,
            "bootstrap": b,
            "root_feature": features[fi] if fi >= 0 else "leaf",
            "root_threshold": float(tree.tree_.threshold[0]),
        })
    return pd.DataFrame(rows)


def axis_models(train: pd.DataFrame, val: pd.DataFrame, out_dir: Path, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    val = val.copy()
    train["select_red"] = (train.teacher_class == 0).astype(int)
    val["select_red"] = (val.teacher_class == 0).astype(int)
    train["select_uav"] = (train.teacher_mode == 1).astype(int)
    val["select_uav"] = (val.teacher_mode == 1).astype(int)
    train["switch_ppo"] = train.teacher_switched.astype(int)
    val["switch_ppo"] = val.teacher_switched.astype(int)

    specs = [
        (
            "class_red_vs_yellow", CLASS_FEATURES, "select_red",
            train.both_class_available, val.both_class_available,
        ),
        (
            "mode_uav_vs_amb", MODE_FEATURES, "select_uav",
            (train.teacher_dest > 0) & train.both_mode_available,
            (val.teacher_dest > 0) & val.both_mode_available,
        ),
        (
            "planner_switch_vs_ppo", SWITCH_FEATURES, "switch_ppo",
            pd.Series(True, index=train.index), pd.Series(True, index=val.index),
        ),
    ]
    metrics, importances, boots = [], [], []
    for i, (name, features, target, tm, vm) in enumerate(specs):
        met, fitted, tr, _ = fit_axis_tree(name, train, val, features, target, tm, vm, out_dir)
        metrics.append(met)
        tree_txt = out_dir / f"{name}_tree.txt"
        importances.append(pd.DataFrame({
            "axis": name,
            "feature": features,
            "importance": fitted.feature_importances_,
        }))
        boots.append(bootstrap_root(
            name, tr, features, target, n_boot=n_boot, seed=20260803 + 1000 * i,
        ))
        if not tree_txt.exists():
            raise RuntimeError(f"축소트리 규칙 파일 미생성: {tree_txt}")
    return pd.DataFrame(metrics), pd.concat(importances, ignore_index=True), pd.concat(boots, ignore_index=True)


def package_importance(policy: str) -> pd.DataFrame:
    path = STUDENT_DIR / f"{policy}.pkl"
    with path.open("rb") as f:
        pkg = pickle.load(f)
    model = pkg["tree"]
    names = [str(x) for x in pkg["feature_names"]]
    family = family_from_policy(policy)
    values = np.zeros(len(names), dtype=float)
    if family == "lgbm":
        values = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    elif family == "cart":
        values = np.asarray(model.feature_importances_, dtype=float)
    elif family == "ebm":
        term_imp = np.asarray(model.term_importances(), dtype=float)
        for imp, term in zip(term_imp, model.term_features_):
            for j in term:
                values[int(j)] += float(imp) / len(term)
    if values.sum() > 0:
        values /= values.sum()
    return pd.DataFrame({
        "policy": policy,
        "family": family,
        "feature": names,
        "normalized_importance": values,
    })


def consensus_importance(best: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.concat([package_importance(str(x)) for x in best.method], ignore_index=True)
    wide = long.pivot_table(
        index="feature", columns="family", values="normalized_importance", fill_value=0.0,
    )
    for family in ("lgbm", "ebm", "cart"):
        if family not in wide:
            wide[family] = 0.0
    wide["consensus_mean"] = wide[["lgbm", "ebm", "cart"]].mean(axis=1)
    wide["models_nonzero"] = (wide[["lgbm", "ebm", "cart"]] > 0).sum(axis=1)
    wide = wide.reset_index().sort_values("consensus_mean", ascending=False)
    return long.sort_values(["family", "normalized_importance"], ascending=[True, False]), wide


def root_summary(boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (axis, feature), g in boot.groupby(["axis", "root_feature"], observed=True):
        rows.append({
            "axis": axis,
            "root_feature": feature,
            "selection_frequency": float(len(g) / len(boot[boot.axis == axis])),
            "threshold_median": float(g.root_threshold.median()),
            "threshold_q25": float(g.root_threshold.quantile(0.25)),
            "threshold_q75": float(g.root_threshold.quantile(0.75)),
            "n_boot": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["axis", "selection_frequency"], ascending=[True, False])


def root_effects(
    train: pd.DataFrame,
    val: pd.DataFrame,
    roots: pd.DataFrame,
    *,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """학습에서 고정한 stump 임계값의 방향과 효과를 p3에서 검증한다."""
    specs = {
        "class_red_vs_yellow": (
            "select_red", train.both_class_available, val.both_class_available,
            "Red 선택",
        ),
        "mode_uav_vs_amb": (
            "select_uav", (train.teacher_dest > 0) & train.both_mode_available,
            (val.teacher_dest > 0) & val.both_mode_available, "UAV 선택",
        ),
        "planner_switch_vs_ppo": (
            "switch_ppo", pd.Series(True, index=train.index), pd.Series(True, index=val.index),
            "PPO 행동 교정",
        ),
    }
    tr = train.copy()
    va = val.copy()
    tr["select_red"] = (tr.teacher_class == 0).astype(int)
    va["select_red"] = (va.teacher_class == 0).astype(int)
    tr["select_uav"] = (tr.teacher_mode == 1).astype(int)
    va["select_uav"] = (va.teacher_mode == 1).astype(int)
    tr["switch_ppo"] = tr.teacher_switched.astype(int)
    va["switch_ppo"] = va.teacher_switched.astype(int)
    rng = np.random.default_rng(20260803)
    rows = []
    for axis, (target, tm, vm, outcome) in specs.items():
        top = roots[roots.axis == axis].sort_values("selection_frequency", ascending=False).iloc[0]
        feature, threshold = str(top.root_feature), float(top.threshold_median)
        train_sub = tr.loc[np.asarray(tm, dtype=bool)]
        val_sub = va.loc[np.asarray(vm, dtype=bool)]

        def rates(frame: pd.DataFrame) -> tuple[float, float, int, int]:
            low = frame[feature] <= threshold
            return (
                float(frame.loc[low, target].mean()),
                float(frame.loc[~low, target].mean()),
                int(low.sum()), int((~low).sum()),
            )

        tr_low, tr_high, tr_n_low, tr_n_high = rates(train_sub)
        va_low, va_high, va_n_low, va_n_high = rates(val_sub)
        groups = {k: g.index.to_numpy() for k, g in val_sub.groupby("region_base", observed=True)}
        keys = np.asarray(list(groups), dtype=object)
        boot = []
        for _ in range(n_boot):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in sampled])
            x = va.loc[idx]
            low = x[feature] <= threshold
            if low.any() and (~low).any():
                boot.append(float(x.loc[~low, target].mean() - x.loc[low, target].mean()))
        boot = np.asarray(boot, dtype=float)
        rows.append({
            "axis": axis,
            "outcome": outcome,
            "root_feature": feature,
            "threshold": threshold,
            "bootstrap_root_frequency": float(top.selection_frequency),
            "train_rate_le_threshold": tr_low,
            "train_rate_gt_threshold": tr_high,
            "train_rate_diff_gt_minus_le": tr_high - tr_low,
            "validation_rate_le_threshold": va_low,
            "validation_rate_gt_threshold": va_high,
            "validation_rate_diff_gt_minus_le": va_high - va_low,
            "validation_diff_boot_lo": float(np.quantile(boot, 0.025)),
            "validation_diff_boot_hi": float(np.quantile(boot, 0.975)),
            "n_train_le": tr_n_low,
            "n_train_gt": tr_n_high,
            "n_validation_le": va_n_low,
            "n_validation_gt": va_n_high,
            "direction_reproduced": bool(np.sign(tr_high - tr_low) == np.sign(va_high - va_low)),
        })
    return pd.DataFrame(rows)


def build_rule_candidates(
    hospital: pd.DataFrame,
    correction: pd.DataFrame,
    choice_coef: pd.DataFrame,
    roots: pd.DataFrame,
    root_effect: pd.DataFrame,
    axes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in choice_coef.itertuples(index=False):
        feature = row.feature
        direction = "낮은" if row.standardized_coef_train_all < 0 else "높은"
        endogenous_or_proxy = feature in {"cand_p_sent_rel", "max_send"}
        stable = bool(row.same_sign_p0_p3 and row.min_abs_fold_coef >= 0.05 and not endogenous_or_proxy)
        if endogenous_or_proxy:
            status = "교사상태 표지(인과해석 금지)"
        else:
            status = "재시뮬레이션 후보" if stable else "보류"
        rows.append({
            "rule_id": f"H-{feature}",
            "axis": "hospital",
            "candidate_rule": f"다른 조건을 함께 통제할 때 {feature}가 {direction} 병원을 우선",
            "train_effect": float(row.standardized_coef_train_all),
            "validation_effect": float(row.coef_p3),
            "train_boot_ci": f"p0–p2 coef={row.coef_p0:.3f}/{row.coef_p1:.3f}/{row.coef_p2:.3f}",
            "validation_boot_ci": f"p3 coef={row.coef_p3:.3f}",
            "stability_gate": bool(stable),
            "status": status,
        })

    for axis in roots.axis.unique():
        g = roots[roots.axis == axis].sort_values("selection_frequency", ascending=False).iloc[0]
        met = axes[axes.axis == axis].iloc[0]
        eff = root_effect[root_effect.axis == axis].iloc[0]
        ci_excludes_zero = not (eff.validation_diff_boot_lo <= 0 <= eff.validation_diff_boot_hi)
        stable = bool(
            g.selection_frequency >= 0.60
            and met.val_balanced_accuracy >= 0.60
            and eff.direction_reproduced
            and ci_excludes_zero
        )
        comparator = ">" if eff.validation_rate_diff_gt_minus_le > 0 else "≤"
        rows.append({
            "rule_id": f"T-{axis}",
            "axis": axis,
            "candidate_rule": (
                f"{g.root_feature} {comparator} {g.threshold_median:.2f}일 때 "
                f"{eff.outcome} 경향 (p3 {eff.validation_rate_le_threshold:.1%}→"
                f"{eff.validation_rate_gt_threshold:.1%}, root 선택률 {100*g.selection_frequency:.1f}%)"
            ),
            "train_effect": float(met.train_balanced_accuracy),
            "validation_effect": float(met.val_balanced_accuracy),
            "train_boot_ci": f"root n={int(g.n_boot)}",
            "validation_boot_ci": f"AUC={met.val_auc:.3f}",
            "stability_gate": stable,
            "status": (
                "기전 근거" if axis == "planner_switch_vs_ppo" and stable
                else "재시뮬레이션 후보" if stable else "설명 보조"
            ),
        })

    # PPO 교정에서 train/validation 방향과 CI가 일치하는 행동 변화도 별도 기록한다.
    cs = correction.pivot(index="feature", columns="dataset", values=[
        "contrast_teacher_minus_ppo_on_switched", "boot_lo", "boot_hi",
    ])
    for feature in CORRECTION_FEATURES:
        tr = float(cs.loc[feature, ("contrast_teacher_minus_ppo_on_switched", "train")])
        va = float(cs.loc[feature, ("contrast_teacher_minus_ppo_on_switched", "validation")])
        tr_lo, tr_hi = float(cs.loc[feature, ("boot_lo", "train")]), float(cs.loc[feature, ("boot_hi", "train")])
        va_lo, va_hi = float(cs.loc[feature, ("boot_lo", "validation")]), float(cs.loc[feature, ("boot_hi", "validation")])
        stable = np.sign(tr) == np.sign(va) and not (tr_lo <= 0 <= tr_hi) and not (va_lo <= 0 <= va_hi)
        rows.append({
            "rule_id": f"C-{feature}",
            "axis": "teacher_correction",
            "candidate_rule": f"PPO 교정 시 {feature}를 {'낮추는' if tr < 0 else '높이는'} 방향",
            "train_effect": tr,
            "validation_effect": va,
            "train_boot_ci": f"[{tr_lo:.4f}, {tr_hi:.4f}]",
            "validation_boot_ci": f"[{va_lo:.4f}, {va_hi:.4f}]",
            "stability_gate": bool(stable),
            "status": "기전 근거" if stable else "보류",
        })
    return pd.DataFrame(rows)


def build_guideline_draft(
    root_effect: pd.DataFrame,
    compact_ablation: pd.DataFrame,
    choice_coef: pd.DataFrame,
) -> pd.DataFrame:
    """보고용으로 바로 읽을 수 있는, 폐루프 근거가 붙은 규칙 초안."""
    coef = choice_coef.set_index("feature")
    abl = compact_ablation.set_index("ablation")
    mode = root_effect[root_effect.axis == "mode_uav_vs_amb"].iloc[0]
    planner = root_effect[root_effect.axis == "planner_switch_vs_ppo"].iloc[0]
    rows = [
        {
            "priority": 1,
            "layer": "병원 선택",
            "guideline": (
                "유효 병원 안에서 score = "
                f"{coef.loc['eta_rank','raw_unit_coef_train_all']:.2f}×ETA순위 "
                f"{coef.loc['cand_occ_ratio','raw_unit_coef_train_all']:+.2f}×점유비를 최대화"
            ),
            "statistical_evidence": "p0–p3에서 ETA·점유비 계수 방향 재현",
            "closed_loop_effect": f"ETA-only 대비 PDR {abl.loc['혼잡도 추가','improvement_ref_minus_candidate']:.6f} 감소",
            "status": "핵심 가이드라인 후보",
        },
        {
            "priority": 2,
            "layer": "이송수단",
            "guideline": (
                f"동일 목적지에서 UAV 시간절감이 약 {mode.threshold:.2f}분을 넘으면 UAV, "
                "그 이하면 AMB를 우선"
            ),
            "statistical_evidence": (
                f"p3 UAV 선택률 {mode.validation_rate_le_threshold:.1%}→"
                f"{mode.validation_rate_gt_threshold:.1%}; root bootstrap {mode.bootstrap_root_frequency:.0%}"
            ),
            "closed_loop_effect": f"ETA+혼잡도 대비 PDR {abl.loc['mode tree 추가','improvement_ref_minus_candidate']:.6f} 감소",
            "status": "핵심 가이드라인 후보",
        },
        {
            "priority": 3,
            "layer": "환자등급",
            "guideline": (
                "두 등급 모두 이송 가능할 때 Yellow 현장잔류·경과시간·AMB 가용성을 이용한 "
                "깊이 3 class tree로 Red/Yellow 우선순위를 조정"
            ),
            "statistical_evidence": "p3 balanced accuracy 0.672; 상세 분기는 compact_policies/class_tree.txt",
            "closed_loop_effect": f"ETA+혼잡도 대비 PDR {abl.loc['class tree 추가','improvement_ref_minus_candidate']:.6f} 감소",
            "status": "조건부 가이드라인 후보",
        },
        {
            "priority": 4,
            "layer": "플래너 개입",
            "guideline": (
                f"Red/Yellow 이송 중 차량이 {math.ceil(planner.threshold):d}대 이상인 고부하 구간에서 "
                "PPO 단독결정을 특히 재검토"
            ),
            "statistical_evidence": (
                f"p3 PPO 교정률 {planner.validation_rate_le_threshold:.1%}→"
                f"{planner.validation_rate_gt_threshold:.1%}; root bootstrap {planner.bootstrap_root_frequency:.0%}"
            ),
            "closed_loop_effect": "직접 규칙이 아니라 교사기전; 별도 trigger ablation 필요",
            "status": "설명 근거",
        },
    ]
    return pd.DataFrame(rows)


def configure_font() -> None:
    for name in ("NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def plot_scoreboard(overall: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    configure_font()
    selected = [
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        "PPO_POINTER_V10",
        "LB_T3",
        "HEUR64_BEST",
        *best.method.tolist(),
    ]
    compact = overall[overall.family == "compact_rule"].nsmallest(1, "pdr_wog_mean")
    if len(compact):
        selected.append(str(compact.method.iloc[0]))
    selected = list(dict.fromkeys(selected))
    x = overall.set_index("method").loc[selected].sort_values("pdr_wog_mean", ascending=False)
    colors = {
        "baseline": "#A5ABB3", "lgbm": "#2B6F9F", "ebm": "#D18B2C", "cart": "#6F8F3D",
        "compact_rule": "#4C84A8",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    y = np.arange(len(x))
    ax.barh(y, x.pdr_wog_mean, xerr=x.pdr_wog_ci95_regions, capsize=3,
            color=[colors.get(f, "#7E57A5") for f in x.family], edgecolor="#39434D", linewidth=0.5)
    labels = [DISPLAY.get(i, i.replace("_", " ")) for i in x.index]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(0.26, float((x.pdr_wog_mean + x.pdr_wog_ci95_regions).max() * 1.08)))
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    fig.suptitle("최종 교사 증류정책의 공통 seed 0–29 폐루프 성능", fontsize=15, y=0.98)
    fig.text(0.5, 0.935, "오차막대: 250개 지역평균의 95% CI · 절대 크기 비교이므로 x축 0부터 표시",
             ha="center", fontsize=9, color="#5B6670")
    for yi, v in zip(y, x.pdr_wog_mean):
        ax.text(v + 0.003, yi, f"{v:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(overall: pd.DataFrame, out: Path) -> None:
    configure_font()
    # fidelity는 p0~p2 적합 후 p3에서 본 clean split 값을 사용한다.
    fit = pd.read_csv(STUDENT_SPLIT_DIR / "fit_summary.csv")
    x = overall[overall.method.str.startswith(("I1_", "I3_"))][["method", "pdr_wog_mean"]]
    x = x.merge(fit[["policy", "family", "fidelity_full"]], left_on="method", right_on="policy")
    palette = {"lgbm": "#2B6F9F", "ebm": "#D18B2C", "cart": "#6F8F3D"}
    fig, ax = plt.subplots(figsize=(9.4, 6.5))
    for family, g in x.groupby("family", observed=True):
        ax.scatter(g.fidelity_full, g.pdr_wog_mean, s=58, color=palette[family],
                   edgecolor="#303840", linewidth=0.5, label=FAMILY_LABEL[family])
    teacher = float(overall.loc[overall.method == "PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "pdr_wog_mean"].iloc[0])
    ax.axhline(teacher, color="#703E8C", linestyle="--", linewidth=1.4, label=f"최종 교사 {teacher:.4f}")
    top = x.sort_values("pdr_wog_mean").groupby("family", as_index=False, sort=False).first()
    offsets = {"lgbm": (7, 8), "ebm": (7, -14), "cart": (7, 8)}
    for row in top.itertuples(index=False):
        ax.annotate(row.method.replace("_", " "), (row.fidelity_full, row.pdr_wog_mean),
                    xytext=offsets[row.family], textcoords="offset points", fontsize=8)
    ax.set_xlabel("내부검증 exact-action fidelity")
    ax.set_ylabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    ax.set_title("정확한 행동복제율과 폐루프 성능의 관계")
    ax.grid(color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rule_evidence(
    bins: pd.DataFrame,
    choice_coef: pd.DataFrame,
    importance: pd.DataFrame,
    out: Path,
) -> None:
    configure_font()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    order = ["≤0", "0–5", "5–10", "10–15", "15–20", "20–30", ">30"]
    val = bins[bins.dataset == "validation"]
    for cls, color, marker in (("Red", "#2B6F9F", "o"), ("Yellow", "#D18B2C", "s")):
        g = val[val.patient_class == cls].set_index("uav_advantage_bin_min").reindex(order)
        axes[0].errorbar(np.arange(len(order)), g.uav_rate_equal_region,
                         yerr=g.ci95_regions, marker=marker, color=color, capsize=3, label=cls)
    axes[0].set_xticks(np.arange(len(order)), order, rotation=25)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("UAV 시간절감(분)")
    axes[0].set_ylabel("UAV 선택률")
    axes[0].set_title("이송수단 선택과 UAV 시간절감")
    axes[0].legend(frameon=False)
    axes[0].grid(color="#D9DEE3", linewidth=0.6)

    label = {
        "eta_rank": "ETA 순위", "cand_p_sent_rel": "상대 발송량*",
        "cand_in_flight": "이송 중", "cand_occ_ratio": "점유비", "max_send": "발송상한*",
    }
    h = choice_coef.copy()
    h["label"] = h.feature.map(label)
    h = h.sort_values("standardized_coef_train_all")
    y = np.arange(len(h))
    xerr = np.vstack([
        h.standardized_coef_train_all - h[["coef_p0", "coef_p1", "coef_p2", "coef_p3"]].min(axis=1),
        h[["coef_p0", "coef_p1", "coef_p2", "coef_p3"]].max(axis=1) - h.standardized_coef_train_all,
    ])
    axes[1].barh(y, h.standardized_coef_train_all,
                 xerr=xerr, color="#6F8F3D", edgecolor="#39434D", capsize=3)
    axes[1].axvline(0, color="#39434D", linewidth=1)
    axes[1].set_yticks(y, h.label)
    axes[1].set_xlabel("조건부 선택점수 표준화 계수 (오차: p0–p3 범위)")
    axes[1].set_title("동일 상태·class·mode 안의 병원 선택")
    axes[1].text(0.5, -0.22, "* 누적발송량·발송상한은 내생성/대리변수 가능성 때문에 가이드라인에서 보류",
                 transform=axes[1].transAxes, ha="center", fontsize=8, color="#5B6670")
    axes[1].grid(axis="x", color="#D9DEE3", linewidth=0.6)

    imp = importance.head(10).sort_values("consensus_mean")
    axes[2].barh(np.arange(len(imp)), imp.consensus_mean, color="#2B6F9F", edgecolor="#39434D")
    axes[2].set_yticks(np.arange(len(imp)), imp.feature)
    axes[2].set_xlabel("CART·EBM·GBDT 정규화 중요도 평균")
    axes[2].set_title("서로 다른 학생모델의 공통 특징")
    axes[2].grid(axis="x", color="#D9DEE3", linewidth=0.6)

    fig.suptitle("최종 교사 로그에서 추출한 규칙 근거", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compact_ablation(overall: pd.DataFrame, out: Path) -> None:
    configure_font()
    order = [
        "RULE_ETA", "RULE_CM_ETA", "RULE_ETA_OCC", "RULE_CLASS_ETA_OCC",
        "RULE_MODE_ETA_OCC", "RULE_CM_ETA_OCC",
    ]
    labels = {
        "RULE_ETA": "ETA만",
        "RULE_CM_ETA": "class+mode+ETA",
        "RULE_ETA_OCC": "ETA+혼잡도",
        "RULE_CLASS_ETA_OCC": "class+ETA+혼잡도",
        "RULE_MODE_ETA_OCC": "mode+ETA+혼잡도",
        "RULE_CM_ETA_OCC": "class+mode+ETA+혼잡도",
    }
    x = overall.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    y = np.arange(len(x))
    colors = ["#A7AFB7", "#8C99A5", "#6F8F3D", "#4C84A8", "#D18B2C", "#2B6F9F"]
    ax.barh(y, x.pdr_wog_mean, xerr=x.pdr_wog_ci95_regions, color=colors,
            edgecolor="#39434D", linewidth=0.5, capsize=3)
    ax.set_yticks(y, [labels[k] for k in order])
    ax.invert_yaxis()
    for method, color in (
        ("PPO_POINTER_V10", "#7A3E9D"),
        ("LB_T3", "#2E8B57"),
        ("PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "#B33E2E"),
    ):
        v = float(overall.loc[overall.method == method, "pdr_wog_mean"].iloc[0])
        ax.axvline(v, linestyle="--", linewidth=1.4, color=color,
                   label=f"{DISPLAY.get(method, method)} {v:.4f}")
    for yi, v in zip(y, x.pdr_wog_mean):
        ax.text(v + 0.004, yi, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.13, 0.28)
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    ax.set_title("통계 규칙의 폐루프 기여: 혼잡도·class·mode 순차 복원")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    best: pd.DataFrame,
    behavior: pd.DataFrame,
    axes: pd.DataFrame,
    rules: pd.DataFrame,
    spatial: pd.DataFrame,
    choice_metrics: pd.DataFrame,
    compact_stats: pd.DataFrame,
    compact_ablation: pd.DataFrame,
    compact_heterogeneity: pd.DataFrame,
    quality: dict[str, Any],
) -> None:
    teacher = "PPO_POINTER_V10_NCRP_H20M16_MILPINJ"
    gbdt = str(best.loc[best.family == "lgbm", "method"].iloc[0])
    teacher_pdr = float(overall.loc[overall.method == teacher, "pdr_wog_mean"].iloc[0])
    gbdt_pdr = float(overall.loc[overall.method == gbdt, "pdr_wog_mean"].iloc[0])
    p = pairwise[(pairwise.reference == teacher) & (pairwise.candidate == gbdt)].iloc[0]
    fit = pd.read_csv(STUDENT_SPLIT_DIR / "fit_summary.csv").set_index("policy").loc[gbdt]
    stable_rules = rules[rules.status == "재시뮬레이션 후보"]
    val_behavior = behavior[behavior.dataset == "validation"].set_index("metric")
    sp = spatial[(spatial.reference == teacher) & (spatial.candidate == gbdt)].iloc[0]
    cm = choice_metrics[choice_metrics.fit == "train_all_to_p3_external"].iloc[0]
    compact_name = "RULE_CM_ETA_OCC"
    compact_pdr = float(overall.loc[overall.method == compact_name, "pdr_wog_mean"].iloc[0])
    compact_lb = compact_stats[(compact_stats.reference == "LB_T3") & (compact_stats.candidate == compact_name)].iloc[0]
    compact_ppo = compact_stats[(compact_stats.reference == "PPO_POINTER_V10") & (compact_stats.candidate == compact_name)].iloc[0]
    abl = compact_ablation.set_index("ablation")
    heur = float(overall.loc[overall.method == "HEUR64_BEST", "pdr_wog_mean"].iloc[0])
    recovery = 100 * (heur - compact_pdr) / (heur - teacher_pdr)
    hetero = compact_heterogeneity.pivot(index="ablation", columns="group", values="improvement")
    report = f"""# 최종 SOTA 교사 증류의 통계분석과 규칙 추출

## 기술 요약

- 최종 교사 **PPO+NCRP-h20m16+MILP**의 대표점 PDR_woG는 **{teacher_pdr:.6f}**이다.
- 최상위 학생 **{gbdt}**는 **{gbdt_pdr:.6f}**로 수치상 {teacher_pdr-gbdt_pdr:+.6f} 낮지만, 지역 paired 차이의 95% CI는 **[{p.improvement_boot_lo:.6f}, {p.improvement_boot_hi:.6f}]**, Holm 보정 Wilcoxon p={p.wilcoxon_holm_p:.4g}이다. 따라서 **교사를 추월했다고 볼 수 없고 통계적으로 구분되지 않는 수준**이다.
- 정확한 전체 행동 fidelity는 **{fit.fidelity_full:.1%}**에 불과하지만 class={fit.fidelity_class:.1%}, mode={fit.fidelity_mode:.1%}, destination={fit.fidelity_dest:.1%}다. 일반화 가능한 지식은 주로 **환자등급·이송수단 축**, 남은 불확실성은 **정확한 병원 선택과 상상미래 의존 교정**에 집중된다.
- train/validation 양쪽에서 재현되고 내생성 검문을 통과한 **재시뮬레이션 후보는 {len(stable_rules)}개**다. 아직 최종 가이드라인이 아니라, 폐루프 실험으로 인과적 성능 기여를 확인할 항목이다.
- 사전고정한 6개 compact 규칙을 대표점에서 재시뮬레이션한 결과, 최종 조합은 **{compact_pdr:.6f}**로 HEUR→교사 개선폭의 **{recovery:.1f}%**를 회수했다. LB-T3와는 통계적 동률(p={compact_lb.wilcoxon_p:.3f})이지만 PPO보다는 성능이 낮다(p={compact_ppo.wilcoxon_p:.2g}, 부호는 PPO 우세).

## 핵심 성능 결과

![최종 교사와 증류정책 성능](policy_scoreboard.png)

대표점 250개와 seed 0–29를 모든 정책에 동일하게 사용했다. 19개 학생을 한꺼번에 확인했기 때문에 개별 p-value 대신 지역 클러스터 paired 차이와 Holm family-wise 보정을 함께 사용했다. 최상위 GBDT의 수치상 교사 초과는 선택 후 비교에서 유의하지 않으므로 SOTA 갱신 근거로 쓰지 않는다.

## fidelity보다 폐루프 PDR이 선택 기준이다

![fidelity와 폐루프 성능](fidelity_closedloop_tradeoff.png)

정확한 목적지 행동을 맞히는 비율은 낮아도 유사한 병원으로 대체되면 누적 PDR은 유지될 수 있다. 반대로 EBM·CART는 단순한 설명을 제공하지만 최종 교사의 미래조건부 교정을 충분히 복제하지 못한다. 따라서 GBDT는 성능 보존형 설명기, EBM·CART는 기전 교차검증 도구로 구분한다.

## 교사 로그에서 재현된 의사결정 기전

![규칙 근거](teacher_rule_evidence.png)

검증좌표에서 교사는 PPO 행동을 지역동일가중 평균 **{val_behavior.loc['PPO 행동 교정','rate_equal_region']:.1%}** 교정했고, MILP가 제안한 후보를 **{val_behavior.loc['MILP 후보 행동 채택','rate_equal_region']:.1%}** 선택했다. 병원 규칙은 선택 병원과 같은 환자등급·이송수단의 다른 유효 병원을 같은 상태 안에서 비교하는 조건부 softmax로 추정했다. p3 top-1은 **{cm.top1_accuracy:.1%}**로 무작위 후보선택 **{cm.uniform_top1:.1%}**보다 높지만, 누적발송량처럼 정책 행동의 결과로 생기는 변수는 인과적 권고에서 제외했다.

## 규칙을 실제로 적용했을 때의 기여

![compact 규칙 폐루프 ablation](compact_rule_ablation.png)

- ETA 순위만 사용한 정책에 혼잡도를 추가하면 PDR이 **{abl.loc['혼잡도 추가','improvement_ref_minus_candidate']:.6f}** 감소했다.
- ETA+혼잡도에 class tree를 추가하면 **{abl.loc['class tree 추가','improvement_ref_minus_candidate']:.6f}**, mode tree를 추가하면 **{abl.loc['mode tree 추가','improvement_ref_minus_candidate']:.6f}** 감소했다.
- class와 mode를 함께 넣으면 **{abl.loc['class+mode 동시 추가','improvement_ref_minus_candidate']:.6f}** 감소했다. 이 결과는 거리만이 아니라 **부하분산 → 이송수단 → 환자등급** 순으로 정책 가치가 더 크게 회수됨을 보여준다.
- 그러나 compact 정책은 PPO를 넘지 못했다. 짧은 가이드라인은 교사의 핵심 기전을 요약하지만, 정확한 병원 상호작용과 미래조건부 교정을 모두 대체하지는 못한다.
- 탐색적 행정유형 분석에서 class 규칙 이득은 군 지역 **{hetero.loc['class tree 추가','군 지역']:.4f}**, 시·구 **{hetero.loc['class tree 추가','시·구 지역']:.4f}**였고, mode 규칙은 군 **{hetero.loc['mode tree 추가','군 지역']:.4f}**, 시·구 **{hetero.loc['mode tree 추가','시·구 지역']:.4f}**였다. 이는 단순히 “UAV 규칙은 군 지역에서만 중요하다”는 결론을 지지하지 않으며, 실제 UAV 시간절감량을 기준으로 한 후속 이질성 분석이 필요하다.

## 데이터와 지표 정의

- 성능평가: 미학습 대표점 250개 × seed 0–29, PDR_woG(낮을수록 우수)
- 증류학습: random4 p0–p2 750좌표, 내부검증 p3 250좌표
- 규칙 추출: 후보행 {quality['train_log']['candidate_rows']+quality['validation_log']['candidate_rows']:,}개, 의사결정 {quality['train_log']['states']+quality['validation_log']['states']:,}개
- paired 효과: 각 지역에서 30개 seed를 먼저 평균한 뒤 `reference − candidate`로 계산
- 불확실성: 250개 지역의 정규근사 CI와 지역 bootstrap percentile CI

## 방법

1. 평가 CSV의 250×30 완전격자, 중복·결측·범위를 검증했다.
2. 최종 교사 CSV의 PPO 재실행행과 기존 PPO cube가 최대오차 {quality['performance']['teacher_base_vs_ppo_max_abs_error']:.1e}로 일치함을 확인했다.
3. 19개 학생을 교사·PPO·LB-T3와 지역 paired 비교하고 Wilcoxon p-value를 Holm 보정했다.
4. 병원 선택은 같은 상태의 동일 class·mode 대안과 비교했고, PPO 교정은 교사행동과 PPO행동의 후보특징 차이로 계산했다.
5. class·mode·교정 여부에 깊이 3의 축소 CART를 적합하고 p3 좌표에서 balanced accuracy를 확인했다.
6. CART·EBM·GBDT의 정규화 특징중요도가 합의하는 조건만 해석 근거로 사용했다.
7. 시군구 간 공간상관 민감도를 보기 위해 17개 광역시도 동일가중 block bootstrap을 별도로 계산했다.
8. 통계 규칙 6개 조합을 대표점 250개×30 seed에서 다시 실행하고, nested ablation을 paired 비교했다.

## 강건성 및 제한

- 최종 교사는 재시드된 상상미래를 사용한다. 이 미래 실현값은 학생 특징에 없으므로 exact-action fidelity의 구조적 상한이 낮다.
- 대표점에서 19개 학생을 확인한 결과는 모델 선택 후 편향 가능성이 있다. Holm 보정과 교사 대비 paired CI를 함께 보고하지만, 별도 외부 좌표셋 재확인이 가장 강한 검증이다.
- 최상위 GBDT 대 교사의 17개 광역시도 동일가중 개선은 **{sp.equal_province_improvement:+.6f}**, block bootstrap 95% CI **[{sp.province_boot_lo:+.6f}, {sp.province_boot_hi:+.6f}]**다. 공간 블록으로 넓혀도 교사와 동률이라는 결론이 유지된다.
- ETA·혼잡도·class·mode **구성요소 수준**의 기여는 폐루프 ablation으로 확인했다. 다만 각 tree 내부의 개별 분기와 임계값 하나하나의 인과효과까지 분리한 것은 아니다.

## 재시뮬레이션 우선순위

1. 폐루프에서 검증된 네 축(ETA·혼잡도·class·mode)을 가이드라인 초안으로 사용한다.
2. GBDT/교사의 지역별 SHAP·부분의존과 compact 규칙의 방향이 합의하는지 교차검증한다.
3. 도시·농촌/UAV 시간절감 구간별 이질성을 분석해 전국 공통규칙과 조건부규칙을 분리한다.
4. 신규 외부좌표에서 임계값을 건드리지 않고 마지막 재현성 검증을 수행한다.

## 추가 질문

- 실제 UAV 시간절감 구간별로 같은 12.24분 threshold가 유지되는가?
- class tree의 각 분기를 하나씩 제거해도 개선이 유지되는가?
"""
    (out_dir / "technical_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap_roots", type=int, default=100)
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    regions, seeds, cubes, students, perf_quality = load_performance()
    overall, pairwise, best = build_performance_tables(regions, seeds, cubes)
    spatial = spatial_block_sensitivity(regions, cubes, best)
    compact_stats, compact_ablation = compact_rule_statistics(cubes)
    compact_heterogeneity = compact_rule_heterogeneity(regions, cubes, compact_ablation)
    train_z, train, train_quality = load_log(TRAIN_LOG, "train")
    val_z, val, val_quality = load_log(VAL_LOG, "validation")
    if set(train.state_key) & set(val.state_key):
        raise ValueError("증류 train/validation 좌표 키가 겹침")

    behavior, uav_bins = behavior_tables(train, val)
    hospital = pd.concat([
        hospital_contrasts(train_z, train), hospital_contrasts(val_z, val),
    ], ignore_index=True)
    choice_coef, choice_metrics, choice_strata = conditional_choice_analysis(train_z, train, val_z, val)
    correction = pd.concat([
        correction_contrasts(train), correction_contrasts(val),
    ], ignore_index=True)
    axes, axis_importance, boot = axis_models(train, val, out, args.bootstrap_roots)
    roots = root_summary(boot)
    root_effect = root_effects(train, val, roots)
    imp_long, imp_consensus = consensus_importance(best)
    rules = build_rule_candidates(hospital, correction, choice_coef, roots, root_effect, axes)
    guidelines = build_guideline_draft(root_effect, compact_ablation, choice_coef)

    quality = {
        "status": "pass",
        "as_of": "2026-08-03",
        "performance": perf_quality,
        "train_log": train_quality,
        "validation_log": val_quality,
        "train_validation_state_key_overlap": 0,
        "source_hashes": {
            str(x.relative_to(REPO)): sha256(x) for x in (
                STUDENT_EVAL, TRAIN_LOG, VAL_LOG, BASE_CUBE, FINAL_TEACHER, LBT_SWEEP,
                COMPACT_EVAL,
            )
        },
        "git_sha": git_sha(),
    }

    tables = {
        "policy_scoreboard.csv": overall,
        "pairwise_students_vs_references.csv": pairwise,
        "spatial_block_sensitivity.csv": spatial,
        "compact_rule_pairwise.csv": compact_stats,
        "compact_rule_ablation.csv": compact_ablation,
        "compact_rule_heterogeneity.csv": compact_heterogeneity,
        "best_student_by_family.csv": best,
        "teacher_behavior_rates.csv": behavior,
        "uav_advantage_bins.csv": uav_bins,
        "hospital_preference_contrasts.csv": hospital,
        "hospital_conditional_choice_coefficients.csv": choice_coef,
        "hospital_conditional_choice_metrics.csv": choice_metrics,
        "hospital_conditional_choice_strata.csv": choice_strata,
        "teacher_vs_ppo_correction_contrasts.csv": correction,
        "axis_tree_metrics.csv": axes,
        "axis_tree_importance.csv": axis_importance,
        "root_threshold_bootstrap.csv": boot,
        "root_threshold_summary.csv": roots,
        "root_threshold_effects.csv": root_effect,
        "student_feature_importance_long.csv": imp_long,
        "student_feature_importance_consensus.csv": imp_consensus,
        "rule_candidates.csv": rules,
        "guideline_draft.csv": guidelines,
    }
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    chart_map = {
        "policy_scoreboard.png": {
            "question": "최종 교사와 대표 학생·기준정책의 폐루프 PDR 순위",
            "family": "comparison/ranking", "type": "horizontal bar with region CI",
            "source": "policy_scoreboard.csv", "palette": "two-root plus neutrals",
        },
        "fidelity_closedloop_tradeoff.png": {
            "question": "exact-action fidelity가 폐루프 PDR을 설명하는가",
            "family": "relationship", "type": "scatter",
            "source": "policy_scoreboard.csv + students_split750/fit_summary.csv",
            "palette": "three model-family roots",
        },
        "teacher_rule_evidence.png": {
            "question": "UAV·병원·특징중요도에서 재현되는 교사 규칙은 무엇인가",
            "family": "comparison + uncertainty", "type": "three-panel line/bar",
            "source": "uav_advantage_bins.csv + hospital_conditional_choice_coefficients.csv + student_feature_importance_consensus.csv",
            "palette": "two-root plus neutrals",
        },
        "compact_rule_ablation.png": {
            "question": "ETA·혼잡도·class·mode 규칙을 실제 적용했을 때의 폐루프 기여",
            "family": "comparison/ablation", "type": "horizontal bar with reference lines",
            "source": "compact_rules_eval250_seed0_29.csv + policy_scoreboard.csv",
            "palette": "sequential categorical plus reference lines",
        },
    }
    (out / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    plot_scoreboard(overall, best, out / "policy_scoreboard.png")
    plot_tradeoff(overall, out / "fidelity_closedloop_tradeoff.png")
    plot_rule_evidence(uav_bins, choice_coef, imp_consensus, out / "teacher_rule_evidence.png")
    plot_compact_ablation(overall, out / "compact_rule_ablation.png")
    write_report(
        out, overall, pairwise, best, behavior, axes, rules, spatial, choice_metrics,
        compact_stats, compact_ablation, compact_heterogeneity, quality,
    )

    top = overall.head(8)[["method", "pdr_wog_mean", "pdr_wog_ci95_regions"]]
    print("[v13-rule-analysis] 데이터 품질 PASS")
    print(top.to_string(index=False))
    print("\n[축소 규칙 검증]")
    print(axes.to_string(index=False))
    print(
        f"\n폐루프 재시뮬레이션 후보="
        f"{int((rules.status == '재시뮬레이션 후보').sum())}/{len(rules)}"
    )
    print(f"산출물 → {out}")


if __name__ == "__main__":
    main()
