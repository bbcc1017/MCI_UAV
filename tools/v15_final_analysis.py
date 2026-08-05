# -*- coding: utf-8 -*-
"""v15 최종 정책의 대표점250 paired 통계·이질성 분석.

LB-T3는 비교표의 독립 기준선일 뿐 v15 정책 구성에는 사용하지 않는다.
최종 CSV가 완성된 뒤에만 실행되며, 기존 v10/v11/v13 공통 seed cube와 정확히
정합하는지 먼저 검증한다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import v13_sota_rule_analysis as v13

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_PORTFOLIO = (
    REPO / "results/scoreboard/v15/final/portfolio_base_g1_eval250_seed0_29.csv"
)
DEFAULT_OUT = REPO / "results/scoreboard/v15/final/analysis"
PORTFOLIO = "V15_GBDT_BASE_PPO_MILP_NCRP"
DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best-of-64",
    "LB_T3": "LB-T3 (독립 기준선)",
    "PPO_POINTER_V10": "PPO Pointer v10",
    "PPO_POINTER_V10_NCRP_H20M16_MILPINJ": "PPO + NCRP h20m16 + MILP",
    "I3_CONNECTED_GBDT_L63_BASE": "최종교사 증류 GBDT I3-L63",
    PORTFOLIO: "v15 정책 후보 포트폴리오",
}


def _portfolio_cube(path: Path, regions: list[str], seeds: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    d = pd.read_csv(path)
    required = {
        "region", "policy", "seed", "pdr_woG", "n_decisions", "n_switch",
        "n_tree_offered", "n_tree_exec", "n_novel_tree_exec", "n_milp_exec",
    }
    if required - set(d):
        raise ValueError(f"v15 최종 CSV 컬럼 누락: {sorted(required-set(d))}")
    if set(d.policy) != {"BASE_G1"}:
        raise ValueError(f"최종 정책 행 오류: {sorted(d.policy.unique())}")
    if d.duplicated(["region", "policy", "seed"]).any():
        raise ValueError("v15 최종 CSV 복합키 중복")
    if len(d) != len(regions) * len(seeds):
        raise ValueError(f"v15 최종 완전격자 아님: {len(d)}")
    if set(d.region) != set(regions) or set(d.seed) != set(map(int, seeds)):
        raise ValueError("v15 최종 평가와 기준 cube의 지역/seed 불일치")
    if d.isna().any().any() or not d.pdr_woG.between(0, 1).all():
        raise ValueError("v15 최종 CSV 결측 또는 PDR 범위 오류")
    cube = (
        d.pivot(index="region", columns="seed", values="pdr_woG")
        .reindex(regions)[seeds]
        .to_numpy(float)
    )
    if not np.isfinite(cube).all():
        raise ValueError("v15 최종 cube 비유한값")
    return cube, d


def _score(method: str, cube: np.ndarray) -> dict:
    x = cube.mean(axis=1)
    lo, hi = v13.bootstrap_ci(x, n_boot=20000, seed=20260804)
    return {
        "method": method,
        "display_name": DISPLAY[method],
        "pdr_woG": float(x.mean()),
        "region_ci95_halfwidth": v13.ci95(x),
        "cluster_boot_lo": lo,
        "cluster_boot_hi": hi,
        "n_regions": len(x),
        "n_seeds": cube.shape[1],
    }


def _province(region: str) -> str:
    hit = re.search(r"_(\d{5})$", str(region))
    return hit.group(1)[:2] if hit else "??"


def _group_summary(region: pd.DataFrame, column: str) -> pd.DataFrame:
    """지역을 독립 표본 단위로 둔 층화 효과와 bootstrap CI를 반환한다."""
    rows = []
    for value, g in region.groupby(column, observed=True):
        x = g.improvement.to_numpy(float)
        lo, hi = v13.bootstrap_ci(x, n_boot=20000, seed=20260804)
        rows.append({
            "stratifier": column,
            "group": str(value),
            "n_regions": len(g),
            "teacher_pdr": float(g.teacher_pdr.mean()),
            "portfolio_pdr": float(g.portfolio_pdr.mean()),
            "improvement": float(x.mean()),
            "cluster_boot_lo": lo,
            "cluster_boot_hi": hi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--portfolio_csv", default=str(DEFAULT_PORTFOLIO))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    path = Path(args.portfolio_csv).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    regions, seeds, cubes, _, base_quality = v13.load_performance()
    portfolio, raw = _portfolio_cube(path, regions, seeds)
    cubes[PORTFOLIO] = portfolio
    methods = [
        PORTFOLIO,
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ",
        "I3_CONNECTED_GBDT_L63_BASE",
        "PPO_POINTER_V10",
        "LB_T3",
        "HEUR64_BEST",
    ]
    score = pd.DataFrame([_score(x, cubes[x]) for x in methods]).sort_values("pdr_woG")
    score.to_csv(out / "final_scoreboard.csv", index=False, encoding="utf-8-sig")

    refs = [x for x in methods if x != PORTFOLIO]
    pair = pd.DataFrame([
        v13.paired_effect(cubes[ref], portfolio, DISPLAY[ref], DISPLAY[PORTFOLIO])
        for ref in refs
    ])
    pair["wilcoxon_holm_p"] = pair.reference.map(
        v13.holm_adjust(dict(zip(pair.reference, pair.wilcoxon_p)))
    )
    pair["significant_after_holm_0_05"] = pair.wilcoxon_holm_p < 0.05
    pair.to_csv(out / "final_pairwise.csv", index=False, encoding="utf-8-sig")

    # 지역 효과는 최종정책의 기존 최종교사 대비 paired 개선으로 정의한다.
    ref = cubes["PPO_POINTER_V10_NCRP_H20M16_MILPINJ"]
    delta = ref - portfolio
    region = pd.DataFrame({
        "region": regions,
        "province_code": [_province(x) for x in regions],
        "admin_type": ["군" if "군" in x.rsplit("_", 1)[0] else "시·구" for x in regions],
        "teacher_pdr": ref.mean(axis=1),
        "portfolio_pdr": portfolio.mean(axis=1),
        "improvement": delta.mean(axis=1),
        "improvement_ci95": [v13.ci95(x) for x in delta],
    })
    region["wtl"] = np.where(
        region.improvement > region.improvement_ci95, "W",
        np.where(region.improvement < -region.improvement_ci95, "L", "T"),
    )
    region.to_csv(out / "final_region_effects.csv", index=False, encoding="utf-8-sig")

    # 사전 개발가설 재검증: 기존 최종교사가 어려워하는 지역에서 통합 이득이 커지는가.
    # 난이도는 비교대상인 기존 교사의 PDR로 정의하며 v15 결과를 사용해 구간을 고르지 않는다.
    region["difficulty_quintile"] = pd.qcut(
        region.teacher_pdr.rank(method="first"), 5,
        labels=["Q1 쉬움", "Q2", "Q3", "Q4", "Q5 어려움"],
    )
    rho, rho_p = spearmanr(region.teacher_pdr, region.improvement)
    heterogeneity = pd.concat([
        _group_summary(region, "difficulty_quintile"),
        _group_summary(region, "admin_type"),
    ], ignore_index=True)
    heterogeneity.to_csv(out / "final_heterogeneity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "n_regions": len(region),
        "difficulty_definition": "existing final teacher regional mean PDR_woG",
        "spearman_r": float(rho),
        "spearman_p": float(rho_p),
        "exploratory": True,
    }]).to_csv(out / "final_difficulty_association.csv", index=False, encoding="utf-8-sig")
    province = (
        region.groupby("province_code", as_index=False)
        .agg(n_regions=("region", "size"), improvement=("improvement", "mean"),
             teacher_pdr=("teacher_pdr", "mean"), portfolio_pdr=("portfolio_pdr", "mean"))
        .sort_values("improvement", ascending=False)
    )
    province.to_csv(out / "final_province_effects.csv", index=False, encoding="utf-8-sig")

    dec = max(float(raw.n_decisions.sum()), 1.0)
    contribution = pd.DataFrame([{
        "switch_rate_from_gbdt_baseline": float(raw.n_switch.sum() / dec),
        "tree_candidate_offered_rate": float(raw.n_tree_offered.sum() / dec),
        "tree_candidate_exec_rate": float(raw.n_tree_exec.sum() / dec),
        "novel_tree_exec_rate": float(raw.n_novel_tree_exec.sum() / dec),
        "milp_exec_rate": float(raw.n_milp_exec.sum() / dec),
        "n_decisions": int(raw.n_decisions.sum()),
    }])
    contribution.to_csv(out / "final_candidate_contribution.csv", index=False, encoding="utf-8-sig")

    # 후보의 선택률과 성능 개선 간 관계는 후보의 인과적 기여가 아니라 작동 위치를 보여주는
    # 설명적 연관성이다. 구성요소의 인과 기여는 개발셋의 제거실험으로 별도 보고한다.
    by_region = raw.groupby("region", as_index=False).agg(
        n_decisions=("n_decisions", "sum"), n_switch=("n_switch", "sum"),
        n_tree_offered=("n_tree_offered", "sum"), n_tree_exec=("n_tree_exec", "sum"),
        n_novel_tree_exec=("n_novel_tree_exec", "sum"), n_milp_exec=("n_milp_exec", "sum"),
    )
    by_region = by_region.merge(region[["region", "teacher_pdr", "improvement"]], on="region")
    association_rows = []
    for count in ("n_switch", "n_tree_offered", "n_tree_exec", "n_novel_tree_exec", "n_milp_exec"):
        rate = by_region[count] / by_region.n_decisions.clip(lower=1)
        r_imp, p_imp = spearmanr(rate, by_region.improvement)
        r_diff, p_diff = spearmanr(rate, by_region.teacher_pdr)
        association_rows.append({
            "candidate_measure": count.replace("n_", "") + "_rate",
            "mean_rate": float(rate.mean()),
            "spearman_with_improvement": float(r_imp),
            "p_with_improvement": float(p_imp),
            "spearman_with_difficulty": float(r_diff),
            "p_with_difficulty": float(p_diff),
            "interpretation": "descriptive association, not causal contribution",
        })
    pd.DataFrame(association_rows).to_csv(
        out / "final_candidate_region_association.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    s = score.sort_values("pdr_woG")
    y = np.arange(len(s))
    colors = ["#a63d2d" if x == PORTFOLIO else "#7f8c8d" if x in {"LB_T3", "HEUR64_BEST"}
              else "#2d6a8e" for x in s.method]
    ax.barh(y, s.pdr_woG, xerr=s.region_ci95_halfwidth, color=colors, capsize=3)
    ax.set_yticks(y, s.display_name)
    ax.invert_yaxis()
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    ax.set_title("v15 최종 정책 후보 포트폴리오 · 공통 seed 0–29")
    ax.grid(axis="x", alpha=0.2)
    for yi, row in enumerate(s.itertuples(index=False)):
        ax.text(row.pdr_woG + row.region_ci95_halfwidth + 0.001, yi,
                f"{row.pdr_woG:.5f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "final_scoreboard.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    q = heterogeneity[heterogeneity.stratifier == "difficulty_quintile"].copy()
    order = ["Q1 쉬움", "Q2", "Q3", "Q4", "Q5 어려움"]
    q["group"] = pd.Categorical(q.group, order, ordered=True)
    q = q.sort_values("group")
    yerr = np.vstack([
        q.improvement - q.cluster_boot_lo,
        q.cluster_boot_hi - q.improvement,
    ])
    axes[0].bar(q.group.astype(str), q.improvement, yerr=yerr, color="#2d6a8e", capsize=3)
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set_ylabel("기존 최종교사 대비 PDR 감소")
    axes[0].set_title("지역 난이도 5분위별 통합정책 이득")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=.2)
    colors = np.where(region.admin_type == "군", "#d97627", "#7f8c8d")
    axes[1].scatter(region.teacher_pdr, region.improvement, c=colors, alpha=.75, s=28)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_xlabel("기존 최종교사의 지역 평균 PDR")
    axes[1].set_ylabel("통합정책의 PDR 감소")
    axes[1].set_title(f"난이도–개선 연관성: r_s={rho:.3f}, p={rho_p:.3g}")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "final_difficulty_heterogeneity.png", dpi=220)
    plt.close(fig)

    quality = {
        "portfolio_csv": str(path), "rows": len(raw), "complete_grid": True,
        "regions": len(regions), "seeds": seeds.tolist(),
        "baseline_quality": base_quality, "lb_t_included_in_policy": False,
        "lb_t_role": "independent heuristic baseline only",
        "selection_protocol": "split750(p0-p2) fit / p3 development / full1000 refit / representative250 comparability",
        "representative250_status": "comparability set previously used by v10-v13; not a pristine blind test",
        "heterogeneity_status": "prespecified directional replication of the p3 development finding; subgroup analyses remain exploratory",
        "blind_test_manifest": str(REPO / "scenarios/manifests/v15_blind250_osrm_manifest.json"),
    }
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(score.to_string(index=False))
    print("\n", pair.to_string(index=False))
    print("\n", contribution.to_string(index=False))
    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
