# -*- coding: utf-8 -*-
"""철회된 LB-T3 결합을 포함한 v14 탐색 결과의 감사용 집계.

``GUIDE_*_LBT3``는 UAV 확보용 원거리 헬기장병원을 AMB 발송상한 풀에 포함하는
도메인 부정합 때문에 최종 가이드라인 성과에서 제외한다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import v13_sota_rule_analysis as v13


GUIDE_EVAL = REPO / "results/scoreboard/v14/guideline_eval250_seed0_29.csv"
GUIDE_META = Path(str(GUIDE_EVAL) + ".meta.json")
DEFAULT_OUT = REPO / "results/scoreboard/v14/guideline_results"
COMPARISON_BINS = REPO / "results/scoreboard/v14/policy_rule_comparison/conditional_action_bins.csv"

DISPLAY = {
    "HEUR64_BEST": "HEUR64 Best-of-64",
    "LB_T3": "LB-T3",
    "PPO_POINTER_V10": "PPO Pointer v10",
    "PPO_POINTER_V10_NCRP_H20M16_MILPINJ": "최종교사 PPO+NCRP+MILP",
    "RULE_CM_ETA_OCC": "기존 compact 규칙",
    "GUIDE_PPO_ETA_OCC": "PPO 규칙+ETA·점유",
    "GUIDE_FINAL_ETA_OCC": "최종교사 규칙+ETA·점유",
    "GUIDE_PPO_LBT3": "PPO 규칙+LB-T3",
    "GUIDE_FINAL_LBT3": "최종교사 규칙+LB-T3",
    "GUIDE_GATED_LBT3": "고부하 보정게이트+LB-T3",
    "GUIDE_EXPLICIT_LBT3": "최종 class+UAV12.2+LB-T3",
    "GUIDE_REDFIRST_LBT3": "Red-first+UAV12.2+LB-T3",
}


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def load_all() -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    regions, seeds, cubes, _, base_quality = v13.load_performance()
    df = pd.read_csv(GUIDE_EVAL)
    required = {"region", "policy", "seed", "pdr_woG", "episode", "info_level", "complexity"}
    if required - set(df):
        raise ValueError(f"가이드라인 평가 컬럼 누락: {sorted(required - set(df))}")
    if df.duplicated(["region", "policy", "seed"]).any():
        raise ValueError("가이드라인 평가 (region, policy, seed) 중복")
    if df.isna().any().any() or not df.pdr_woG.between(0, 1).all():
        raise ValueError("가이드라인 평가 결측 또는 PDR 범위 오류")
    if set(df.region) != set(regions) or set(df.seed) != set(seeds):
        raise ValueError("가이드라인과 기준 cube의 지역 또는 seed 불일치")
    counts = df.groupby("policy", observed=True).size()
    if len(counts) != 7 or not (counts == 7500).all():
        raise ValueError(f"가이드라인 완전격자 실패: {counts.to_dict()}")
    for policy, group in df.groupby("policy", observed=True):
        cubes[str(policy)] = (
            group.pivot(index="region", columns="seed", values="pdr_woG")
            .reindex(regions)[seeds].to_numpy(float)
        )
    meta = json.loads(GUIDE_META.read_text())
    quality = {
        "status": "pass",
        "rows": int(len(df)), "policies": int(df.policy.nunique()),
        "regions": len(regions), "seeds": seeds.tolist(),
        "duplicate_keys": 0, "null_cells": 0,
        "complete_cells_per_policy": 7500,
        "manifest": meta["manifest"], "manifest_sha256": meta["manifest_sha256"],
        "tree_hashes": meta["tree_hashes"],
        "base_quality": base_quality,
    }
    return regions, seeds, cubes, quality


def overall_table(regions: list[str], seeds: np.ndarray, cubes: dict[str, np.ndarray]) -> pd.DataFrame:
    keep = [
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "PPO_POINTER_V10", "LB_T3",
        "RULE_CM_ETA_OCC", "HEUR64_BEST",
    ] + sorted(x for x in cubes if x.startswith("GUIDE_"))
    rows = []
    for method in keep:
        region_mean = cubes[method].mean(axis=1)
        lo, hi = v13.bootstrap_ci(region_mean)
        rows.append({
            "method": method, "display_name": DISPLAY[method],
            "pdr_wog_mean": float(region_mean.mean()),
            "pdr_wog_ci95_regions": ci95(region_mean),
            "bootstrap_lo": lo, "bootstrap_hi": hi,
            "n_regions": len(regions), "n_seeds": len(seeds),
        })
    return pd.DataFrame(rows).sort_values("pdr_wog_mean").reset_index(drop=True)


def paired_rows(cubes: dict[str, np.ndarray]) -> pd.DataFrame:
    refs = [
        "HEUR64_BEST", "LB_T3", "PPO_POINTER_V10",
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "RULE_CM_ETA_OCC",
    ]
    guides = sorted(x for x in cubes if x.startswith("GUIDE_"))
    rows = [v13.paired_effect(cubes[ref], cubes[g], ref, g) for ref in refs for g in guides]
    out = pd.DataFrame(rows)
    # 기준별 7개 가이드라인 탐색에 Holm 보정.
    adjusted = []
    for ref, group in out.groupby("reference", sort=False):
        adj = v13.holm_adjust(dict(zip(group.candidate, group.wilcoxon_p)))
        adjusted.extend((idx, adj[row.candidate]) for idx, row in group.iterrows())
    out["wilcoxon_holm_p"] = np.nan
    for idx, value in adjusted:
        out.loc[idx, "wilcoxon_holm_p"] = value
    out["significant_after_holm_0_05"] = out.wilcoxon_holm_p < 0.05
    return out.sort_values(["reference", "improvement_ref_minus_candidate"], ascending=[True, False])


def component_ablation(cubes: dict[str, np.ndarray]) -> pd.DataFrame:
    pairs = [
        ("GUIDE_PPO_LBT3", "GUIDE_FINAL_LBT3", "PPO 규칙→최종교사 규칙"),
        ("GUIDE_PPO_LBT3", "GUIDE_GATED_LBT3", "고부하에서만 최종규칙"),
        ("GUIDE_REDFIRST_LBT3", "GUIDE_EXPLICIT_LBT3", "Red-first→최종 class tree"),
        ("GUIDE_EXPLICIT_LBT3", "GUIDE_FINAL_LBT3", "UAV 12.2분 임계→mode tree"),
        ("GUIDE_FINAL_ETA_OCC", "GUIDE_FINAL_LBT3", "ETA·점유 점수→LB-T3 병원골격"),
        ("RULE_CM_ETA_OCC", "GUIDE_FINAL_LBT3", "기존 compact→LB-T3 결합"),
        ("LB_T3", "GUIDE_FINAL_LBT3", "LB-T3→교사규칙 결합"),
    ]
    out = []
    for base, enriched, label in pairs:
        rec = v13.paired_effect(cubes[base], cubes[enriched], base, enriched)
        rec["ablation"] = label
        out.append(rec)
    frame = pd.DataFrame(out)
    adj = v13.holm_adjust(dict(zip(frame.ablation, frame.wilcoxon_p)))
    frame["wilcoxon_holm_p"] = frame.ablation.map(adj)
    frame["significant_after_holm_0_05"] = frame.wilcoxon_holm_p < 0.05
    return frame


def regional_effects(regions: list[str], cubes: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = cubes["LB_T3"].mean(axis=1)
    guide = cubes["GUIDE_FINAL_LBT3"].mean(axis=1)
    rows = []
    for i, region in enumerate(regions):
        diff = cubes["LB_T3"][i] - cubes["GUIDE_FINAL_LBT3"][i]
        mean, half = float(diff.mean()), ci95(diff)
        rows.append({
            "region": region,
            "province_code": re.search(r"_(\d{5})$", region).group(1)[:2],
            "region_type": "군" if "군" in region.rsplit("_", 1)[0] else "시·구",
            "pdr_lb_t3": float(base[i]), "pdr_guide_final_lbt3": float(guide[i]),
            "improvement": mean, "episode_ci95": half,
            "wtl": "W" if mean > half else "L" if mean < -half else "T",
        })
    detail = pd.DataFrame(rows)
    summary = []
    for group, data in detail.groupby("region_type", observed=True):
        lo, hi = v13.bootstrap_ci(data.improvement)
        summary.append({
            "group": group, "n_regions": len(data),
            "mean_improvement": float(data.improvement.mean()),
            "bootstrap_lo": lo, "bootstrap_hi": hi,
            "W": int((data.wtl == "W").sum()), "T": int((data.wtl == "T").sum()),
            "L": int((data.wtl == "L").sum()),
        })
    # 광역시도 동일가중 민감도.
    province = detail.groupby("province_code", observed=True).improvement.mean()
    lo, hi = v13.bootstrap_ci(province, n_boot=10000)
    summary.append({
        "group": "17개 광역시도 동일가중", "n_regions": len(province),
        "mean_improvement": float(province.mean()), "bootstrap_lo": lo, "bootstrap_hi": hi,
        "W": int((province > 0).sum()), "T": 0, "L": int((province < 0).sum()),
    })
    return detail, pd.DataFrame(summary)


def configure_font() -> None:
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def plot_scoreboard(overall: pd.DataFrame, out: Path) -> None:
    configure_font()
    order = [
        "PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "PPO_POINTER_V10",
        "GUIDE_FINAL_LBT3", "GUIDE_GATED_LBT3", "GUIDE_EXPLICIT_LBT3",
        "GUIDE_PPO_LBT3", "GUIDE_FINAL_ETA_OCC", "RULE_CM_ETA_OCC", "LB_T3",
        "GUIDE_PPO_ETA_OCC", "GUIDE_REDFIRST_LBT3", "HEUR64_BEST",
    ]
    data = overall.set_index("method").loc[order]
    colors = [
        "#B33E2E", "#2B6F9F", "#D18B2C", "#D9A45A", "#E2B875", "#6F8F3D",
        "#8C99A5", "#4C84A8", "#A7AFB7", "#A7AFB7", "#A7AFB7", "#B9B9B9",
    ]
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    y = np.arange(len(data))
    ax.barh(y, data.pdr_wog_mean, xerr=data.pdr_wog_ci95_regions,
            color=colors, edgecolor="#39434D", linewidth=0.5, capsize=3)
    ax.set_yticks(y, data.display_name)
    ax.invert_yaxis()
    for yi, value in zip(y, data.pdr_wog_mean):
        ax.text(value + 0.003, yi, f"{value:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.12, 0.255)
    ax.set_xlabel("대표점 250개 평균 PDR_woG (낮을수록 우수)")
    ax.set_title("PPO·최종교사 사후규칙을 결합한 휴리스틱 성능")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, out: Path) -> None:
    configure_font()
    data = ablation.copy().sort_values("improvement_ref_minus_candidate")
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    y = np.arange(len(data))
    colors = np.where(data.improvement_ref_minus_candidate >= 0, "#2B6F9F", "#C7CDD3")
    ax.barh(y, data.improvement_ref_minus_candidate,
            xerr=data.improvement_ref_minus_candidate - data.improvement_boot_lo,
            color=colors, edgecolor="#39434D", capsize=3)
    ax.axvline(0, color="#39434D", linewidth=1)
    ax.set_yticks(y, data.ablation)
    ax.set_xlabel("PDR 개선량 = 기존 - 개선정책 (양수일수록 우수)")
    ax.set_title("사후규칙 구성요소의 폐루프 기여")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out: Path, overall: pd.DataFrame, pairs: pd.DataFrame, ablation: pd.DataFrame,
    regional: pd.DataFrame, quality: dict[str, Any],
) -> None:
    score = overall.set_index("method")
    lb = float(score.loc["LB_T3", "pdr_wog_mean"])
    guide = float(score.loc["GUIDE_FINAL_LBT3", "pdr_wog_mean"])
    ppo = float(score.loc["PPO_POINTER_V10", "pdr_wog_mean"])
    teacher = float(score.loc["PPO_POINTER_V10_NCRP_H20M16_MILPINJ", "pdr_wog_mean"])
    pair = pairs[(pairs.reference == "LB_T3") & (pairs.candidate == "GUIDE_FINAL_LBT3")].iloc[0]
    comp = ablation.set_index("ablation")
    reg = regional.set_index("group")
    class_bins = pd.read_csv(COMPARISON_BINS)
    class_bins = class_bins[
        (class_bins.dataset == "validation")
        & (class_bins.condition == "현장 Yellow 대기자수")
        & (class_bins.policy == "FINAL_TEACHER")
    ].set_index("stratum")
    hospital_ablation = comp.loc["ETA·점유 점수→LB-T3 병원골격"]
    report = f"""# [철회된 탐색] 사후통계 규칙과 LB-T3 결합 실험

> **사용 금지:** 아래 `GUIDE_*_LBT3` 결합은 UAV 확보용 원거리 헬기장병원을 AMB 발송상한 풀에도 포함하는 도메인 부정합이 확인되어 본 연구 주장에서는 철회했다. LB-T3는 별도 기준선으로만 유지한다. 상세 근거는 `../WITHDRAWN_LBT3_COMBINATION.md`에 기록했다.

## 기술 요약

- p0~p2 로그에서 사전고정한 7개 규칙을 미사용 대표점 250개×seed 0~29에서 재시뮬레이션했다.
- 최상위 가이드라인 **최종교사 class·mode 규칙 + LB-T3 병원골격**의 PDR_woG는 **{guide:.6f}**로 LB-T3 **{lb:.6f}**보다 **{lb-guide:.6f} ({100*(lb-guide)/lb:.2f}%) 감소**했다.
- 지역 paired bootstrap 95% CI는 **[{pair.improvement_boot_lo:.6f}, {pair.improvement_boot_hi:.6f}]**, 7개 가이드라인 탐색에 대한 Holm 보정 Wilcoxon p={pair.wilcoxon_holm_p:.3g}, 지역 W/T/L={int(pair['W'])}/{int(pair['T'])}/{int(pair['L'])}다. 따라서 개선은 평균 수치만의 현상이 아니다.
- PPO({ppo:.6f})와 최종교사({teacher:.6f})에는 아직 못 미친다. 이번 성과의 의미는 블랙박스 교사에서 학습한 작은 트리 규칙을 명시적 병원 부하분산 규칙과 결합해 **해석 가능한 규칙 정책으로 성능 개선을 폐루프 재현했다는 것**이다.

## 최종교사 규칙과 LB-T3의 결합이 가장 좋은 설명형 정책이다

![가이드라인 scoreboard](guideline_scoreboard.png)

동일한 대표점·seed·마스크를 사용했다. 최종교사의 class·mode 규칙을 ETA·점유 선형점수와 결합한 기존 compact 정책보다 LB-T3 병원 골격과 결합할 때 더 좋았다. 이는 최종교사에서 추출할 지식과 기존 OR형 휴리스틱의 강점을 역할별로 분리해야 함을 뜻한다.

## 개선의 핵심은 class·mode 규칙과 검증된 병원 부하분산의 결합이다

![구성요소 기여](guideline_component_ablation.png)

- 같은 LB-T3 병원골격에서 PPO 규칙을 최종교사 규칙으로 바꾸면 PDR이 **{comp.loc['PPO 규칙→최종교사 규칙','improvement_ref_minus_candidate']:.6f}** 감소했다.
- Red-first를 최종교사 class tree로 바꾸면 **{comp.loc['Red-first→최종 class tree','improvement_ref_minus_candidate']:.6f}** 감소했다. 환자등급 순서를 고정하지 않는 조건부 우선순위가 중요하다.
- 단일 UAV 12.2분 임계값을 mode tree로 바꾼 추가 이득은 **{comp.loc['UAV 12.2분 임계→mode tree','improvement_ref_minus_candidate']:.6f}**다. 단일 임계값만으로도 대부분을 설명하지만 주변 상태가 남은 차이를 만든다.
- ETA·점유 선형 병원점수를 LB-T3 골격으로 바꾸면 평균 PDR은 **{hospital_ablation.improvement_ref_minus_candidate:.6f}** 낮아졌지만, 지역 순위 기반 Holm 보정 p={hospital_ablation.wilcoxon_holm_p:.3f}로 일관된 우세는 확정할 수 없었다. 따라서 발송상한 기반 부하분산은 **평균값 측면의 유망한 골격**으로만 해석한다.

## 지역별로도 개선 방향이 유지된다

- 군 지역 평균 개선 **{reg.loc['군','mean_improvement']:.6f}**, 95% CI [{reg.loc['군','bootstrap_lo']:.6f}, {reg.loc['군','bootstrap_hi']:.6f}]
- 시·구 평균 개선 **{reg.loc['시·구','mean_improvement']:.6f}**, 95% CI [{reg.loc['시·구','bootstrap_lo']:.6f}, {reg.loc['시·구','bootstrap_hi']:.6f}]
- 17개 광역시도 동일가중 개선 **{reg.loc['17개 광역시도 동일가중','mean_improvement']:.6f}**, 95% CI [{reg.loc['17개 광역시도 동일가중','bootstrap_lo']:.6f}, {reg.loc['17개 광역시도 동일가중','bootstrap_hi']:.6f}]

행정구역 유형은 UAV 효용의 직접 측정치가 아니므로 도시·농촌 인과해석에는 쓰지 않는다. 다만 개선이 특정 소수 도시의 평균효과만은 아니라는 강건성 점검이다.

## 연구적으로 도출할 수 있는 가이드라인

1. **병원:** 단순 최근접보다 병원당 누적발송이 3명 미만인 적격 병원 중 가까운 곳을 우선하고, 모두 상한에 도달하면 가장 적게 보낸 병원으로 분산한다.
2. **이송수단:** UAV가 AMB보다 약 12분 이상 빠른 구간에서 UAV 선택이 급격히 증가한다. 단, 최종 정책은 거리만이 아니라 현장 부하와 병원상태를 함께 본다.
3. **환자등급:** Red·Yellow 모두 선택 가능한 때, 최종교사의 Red 선택률은 Yellow 대기 0–9명 **{class_bins.loc['0–9','rate_equal_region']:.1%}**에서 20명 이상 **{class_bins.loc['20+','rate_equal_region']:.1%}**로 낮아졌다. 즉 Yellow 적체가 커질수록 Yellow 이송 비중을 높이는 조건부 정책이며, Red-first 고정규칙보다 폐루프 성능이 좋았다.
4. **플래너 개입:** 최종교사의 보정은 주로 이송수단이 아니라 병원 목적지 변경이었다. 이송 중·복귀 중 차량이 13대 이상인 고부하 구간에서 PPO 목적지를 특히 재검토한다.

## 검증 범위와 한계

- 가이드라인 규칙 적합에는 random4 p0~p2만 사용했고, p3는 규칙 재현성 확인, 대표점250은 폐루프 최종평가에만 사용했다.
- 7개 후보 중 최상위를 보고하므로 LB-T3 비교 p-value는 Holm 보정했다.
- HEUR64 Best-of-64와 LB-T3는 공통 좌표·seed로 구성된 강한 기준선이다. GUIDE_FINAL_LBT3는 Full64의 지역별 class/mode 조합을 쓴 것이 아니라, **LB-T3의 병원당 3명 발송상한 병원선택 골격**에 교사의 class/mode 트리를 결합한 별도 규칙 정책이다.
- 사후 규칙은 행동과 성능의 재현을 보여주지만 의료현장의 인과효과나 임상적 안전성을 직접 증명하지 않는다.

## 다음 단계

1. class tree 각 분기를 하나씩 제거해 어떤 분기가 성능 개선을 만드는지 폐루프 ablation한다.
2. 대표점에서 UAV 시간절감·헬리패드 밀도·병원 접근성을 직접 계산해 지역별 이질성을 분석한다.
3. 전혀 새로운 외부좌표에서 임계값과 규칙을 고정한 최종 재현성 검증을 수행한다.
"""
    (out / "technical_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    regions, seeds, cubes, quality = load_all()
    overall = overall_table(regions, seeds, cubes)
    pairs = paired_rows(cubes)
    ablation = component_ablation(cubes)
    detail, regional = regional_effects(regions, cubes)
    for name, frame in {
        "guideline_scoreboard.csv": overall,
        "guideline_pairwise.csv": pairs,
        "guideline_component_ablation.csv": ablation,
        "guideline_region_effects.csv": detail,
        "guideline_region_summary.csv": regional,
    }.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    (out / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "chart_map.json").write_text(json.dumps({
        "guideline_scoreboard.png": {"family": "ranked horizontal bar", "source": "guideline_scoreboard.csv"},
        "guideline_component_ablation.png": {"family": "diverging effect bar", "source": "guideline_component_ablation.csv"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_scoreboard(overall, out / "guideline_scoreboard.png")
    plot_ablation(ablation, out / "guideline_component_ablation.png")
    write_report(out, overall, pairs, ablation, regional, quality)
    print("[v14-guideline-results] 데이터 품질 PASS")
    print(overall[["method", "pdr_wog_mean", "pdr_wog_ci95_regions"]].to_string(index=False))
    print("\n[LB-T3 대비]")
    print(pairs[pairs.reference == "LB_T3"][["candidate", "improvement_ref_minus_candidate", "improvement_boot_lo", "improvement_boot_hi", "wilcoxon_holm_p", "W", "T", "L"]].to_string(index=False))
    print(f"산출물 → {out}")


if __name__ == "__main__":
    main()
