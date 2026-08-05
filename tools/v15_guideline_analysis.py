# -*- coding: utf-8 -*-
"""v15 결정로그에서 일반화 가능한 정책 기전과 가이드라인 후보를 추출한다.

정책 성능 검정과 설명모형 적합을 분리한다. 얕은 CART는 정책을 대체하기 위한 것이 아니라
class/mode/switch 기전을 요약하며, 시군구를 GroupKFold 단위로 분리해 설명의 지역전이를 잰다.
LB-T 계열 특징·행동은 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO / "results/scoreboard/v15/explanation/portfolio_p3_decisions_seed9200.csv"
DEFAULT_OUT = REPO / "results/scoreboard/v15/explanation/analysis"
KEY = ["region", "seed", "decision"]
STATE_FEATURES = [
    "red_at_site", "yellow_at_site", "amb_available", "uav_available", "time_min",
    "red_unrescued", "yellow_unrescued", "red_in_transport", "yellow_in_transport",
    "total_p_sent", "total_in_flight", "amb_busy", "amb_min_return", "amb_mean_return",
    "uav_busy", "uav_min_return", "uav_mean_return", "fleet_critical", "rho",
    "red_at_hospital", "yellow_at_hospital", "red_done", "yellow_done", "total_cap_remain",
]


def _bootstrap(x: np.ndarray, seed: int = 20260804, n_boot: int = 10000) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    if len(a) <= 1:
        v = float(a.mean()) if len(a) else math.nan
        return v, v
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def _fit_tree(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, out: Path) -> dict:
    keep = X.columns[X.nunique(dropna=False) > 1]
    X = X[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.astype(int)
    if len(y) == 0:
        (out / f"{name}_tree.txt").write_text("적격 선택상황 0건 — 분기 없음\n", encoding="utf-8")
        return {
            "target": name, "n": 0, "positive_rate": math.nan,
            "group_balanced_accuracy": math.nan, "group_roc_auc": math.nan,
            "n_features": len(X.columns),
            "importance": pd.DataFrame(columns=["target", "feature", "importance"]),
        }
    group_values = np.asarray(groups.astype(str))
    group_count = pd.Series(group_values).map(pd.Series(group_values).value_counts()).to_numpy(float)
    sample_weight = 1.0 / group_count
    sample_weight /= sample_weight.mean()
    if y.nunique() < 2:
        (out / f"{name}_tree.txt").write_text(
            f"단일 표적값({int(y.iloc[0])}) — 분기 없음\n", encoding="utf-8"
        )
        return {
            "target": name, "n": len(y), "positive_rate": float(y.mean()),
            "group_balanced_accuracy": math.nan, "group_roc_auc": math.nan,
            "n_features": len(X.columns),
            "importance": pd.DataFrame(columns=["target", "feature", "importance"]),
        }
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    n_splits = min(5, groups.nunique())
    if n_splits >= 2:
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, group_values):
            model = DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=max(30, len(tr) // 100),
                class_weight="balanced", random_state=20260804,
            )
            model.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weight[tr])
            pred[va] = model.predict(X.iloc[va])
            prob[va] = model.predict_proba(X.iloc[va])[:, list(model.classes_).index(1)]
    final = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=max(30, len(X) // 100),
        class_weight="balanced", random_state=20260804,
    ).fit(X, y, sample_weight=sample_weight)
    if n_splits < 2:  # 1지역 smoke 경로; 본 분석은 지역 GroupKFold를 사용한다.
        pred = final.predict(X)
        prob = final.predict_proba(X)[:, list(final.classes_).index(1)]
    text = export_text(final, feature_names=list(X.columns), decimals=3)
    (out / f"{name}_tree.txt").write_text(text, encoding="utf-8")
    importance = pd.DataFrame({"feature": X.columns, "importance": final.feature_importances_})
    importance = importance[importance.importance > 0].sort_values("importance", ascending=False)
    importance.insert(0, "target", name)
    macro_bacc, macro_auc = [], []
    for group in np.unique(group_values):
        m = group_values == group
        if len(np.unique(y.to_numpy()[m])) < 2:
            continue
        macro_bacc.append(balanced_accuracy_score(y.to_numpy()[m], pred[m]))
        macro_auc.append(roc_auc_score(y.to_numpy()[m], prob[m]))
    return {
        "target": name, "n": len(y), "positive_rate": float(y.mean()),
        "group_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "group_roc_auc": float(roc_auc_score(y, prob)),
        "region_macro_balanced_accuracy": float(np.mean(macro_bacc)) if macro_bacc else math.nan,
        "region_macro_roc_auc": float(np.mean(macro_auc)) if macro_auc else math.nan,
        "region_equal_weight_fit": True,
        "n_features": len(X.columns), "importance": importance,
    }


def _quality(d: pd.DataFrame) -> None:
    required = set(KEY + [
        "role", "action", "class", "destination", "mode", "source", "switched",
        "episode_pdr_woG",
    ] + STATE_FEATURES)
    if required - set(d):
        raise ValueError(f"설명 로그 컬럼 누락: {sorted(required-set(d))}")
    if d.duplicated(KEY + ["role"]).any():
        raise ValueError("결정·role 복합키 중복")
    if set(d.role) != {"GBDT_BASE", "PPO_GREEDY", "EXEC"}:
        raise ValueError(f"role 오류: {sorted(d.role.unique())}")
    if not (d.groupby(KEY).role.nunique() == 3).all():
        raise ValueError("결정마다 role 3개가 아님")
    if not d.episode_pdr_woG.between(0, 1).all():
        raise ValueError("PDR 범위 오류")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=str(DEFAULT_LOG))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    path, out = Path(args.log).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(path)
    _quality(d)
    role = {name: g.set_index(KEY).sort_index() for name, g in d.groupby("role")}
    base, ppo, exe = role["GBDT_BASE"], role["PPO_GREEDY"], role["EXEC"]
    if not base.index.equals(ppo.index) or not base.index.equals(exe.index):
        raise ValueError("role별 결정키 불일치")

    decision = pd.DataFrame(index=exe.index)
    for axis in ("action", "class", "destination", "mode"):
        decision[f"exec_{axis}"] = exe[axis]
        decision[f"base_{axis}"] = base[axis]
        decision[f"ppo_{axis}"] = ppo[axis]
    decision["switch"] = decision.exec_action != decision.base_action
    decision["class_change"] = decision.exec_class != decision.base_class
    decision["destination_change"] = decision.exec_destination != decision.base_destination
    decision["mode_change"] = decision.exec_mode != decision.base_mode
    decision["source"] = exe.source
    decision["dpdr"] = exe.dpdr
    decision.reset_index().to_csv(out / "decision_corrections.csv", index=False, encoding="utf-8-sig")

    rates = {
        "n_regions": int(exe.reset_index().region.nunique()),
        "n_decisions": int(len(exe)),
        "ppo_gbdt_exact_agreement": float((ppo.action == base.action).mean()),
        "ppo_gbdt_class_agreement": float((ppo["class"] == base["class"]).mean()),
        "ppo_gbdt_destination_agreement": float((ppo.destination == base.destination).mean()),
        "ppo_gbdt_mode_agreement": float((ppo["mode"] == base["mode"]).mean()),
        "planner_switch_from_gbdt": float(decision.switch.mean()),
        "switch_class_change": float(decision.loc[decision.switch, "class_change"].mean()),
        "switch_destination_change": float(decision.loc[decision.switch, "destination_change"].mean()),
        "switch_mode_change": float(decision.loc[decision.switch, "mode_change"].mean()),
        "exec_red_rate": float((exe["class"] == 0).mean()),
        "exec_uav_rate": float((exe["mode"] == 1).mean()),
        "lb_t_included": False,
    }
    both_class = (exe.red_at_site > 0) & (exe.yellow_at_site > 0)
    both_mode = (
        (exe.is_stay < 0.5) & (exe.has_helipad > 0.5)
        & (exe.amb_available > 0) & (exe.uav_available > 0)
    )
    rates.update({
        "n_both_class_available": int(both_class.sum()),
        "exec_red_rate_both_class_available": float((exe.loc[both_class, "class"] == 0).mean()),
        "n_both_mode_available": int(both_mode.sum()),
        "exec_uav_rate_both_mode_available": float((exe.loc[both_mode, "mode"] == 1).mean()),
    })
    pd.DataFrame([rates]).to_csv(out / "behavior_rates.csv", index=False, encoding="utf-8-sig")

    # 어떤 제안기가 실제 실행행동에 관여했는지 중복 허용 membership으로 센다.
    # 동일 행동을 여러 모델이 제안할 수 있으므로 합계는 100%가 될 필요가 없다.
    source_patterns = {
        "PPO_TOPK": "PPO_TOPK",
        "GBDT_G1": "TREE:G1",
        "GBDT_G2": "TREE:G2",
        "GBDT_G3": "TREE:G3",
        "MILP": "MILP",
        "G1_BASE": "G1_BASE",
    }
    source_rows = []
    for name, pattern in source_patterns.items():
        member = exe.source.astype(str).str.contains(pattern, regex=False).to_numpy()
        switched = decision.switch.to_numpy(bool)
        selected = member & switched
        source_rows.append({
            "source": name,
            "exec_membership_n": int(member.sum()),
            "exec_membership_rate": float(member.mean()),
            "switched_exec_n": int(selected.sum()),
            "share_of_switched_decisions": float(selected.sum() / max(switched.sum(), 1)),
            "mean_dpdr_when_switched_exec": (
                float(exe.dpdr.to_numpy(float)[selected].mean()) if selected.any() else math.nan
            ),
            "interpretation": "overlapping source membership; descriptive, not causal",
        })
    pd.DataFrame(source_rows).to_csv(
        out / "execution_source_membership.csv", index=False, encoding="utf-8-sig"
    )

    # 교정 방향을 class·mode 전이로 직접 요약한다. 목적지는 ID보다 속성 contrast로 해석한다.
    transition_rows = []
    for axis, labels in (
        ("class", {0: "Red", 1: "Yellow"}),
        ("mode", {0: "AMB", 1: "UAV"}),
    ):
        b = decision[f"base_{axis}"].astype(int)
        e = decision[f"exec_{axis}"].astype(int)
        for before in sorted(set(b)):
            denom = int((b == before).sum())
            for after in sorted(set(e)):
                n = int(((b == before) & (e == after)).sum())
                transition_rows.append({
                    "axis": axis,
                    "from": labels.get(before, str(before)),
                    "to": labels.get(after, str(after)),
                    "n": n,
                    "rate_given_from": float(n / max(denom, 1)),
                })
    pd.DataFrame(transition_rows).to_csv(
        out / "correction_transition_rates.csv", index=False, encoding="utf-8-sig"
    )

    # 실행행동의 class/mode와 GBDT 이탈을 사전 관측 가능한 특징만으로 요약한다.
    X_state = exe[STATE_FEATURES].copy()
    class_result = _fit_tree(
        "class_red", X_state.loc[both_class], (exe.loc[both_class, "class"] == 0),
        exe.reset_index().loc[both_class.to_numpy(), "region"], out,
    )
    mode_X = exe[STATE_FEATURES + [
        "is_red", "uav_advantage_min", "eta_raw_min", "cand_cap_remain",
        "cand_occ_ratio", "cand_p_sent", "cand_in_flight",
    ]].copy()
    mode_result = _fit_tree(
        "mode_uav", mode_X.loc[both_mode], (exe.loc[both_mode, "mode"] == 1),
        exe.reset_index().loc[both_mode.to_numpy(), "region"], out,
    )
    switch_X = base[STATE_FEATURES + [
        "is_red", "is_uav", "eta_raw_min", "uav_advantage_min", "cand_cap_remain",
        "cand_occ_ratio", "cand_p_sent", "cand_in_flight",
    ]].copy()
    switch_X["ppo_class_diff"] = (ppo["class"] != base["class"]).astype(int)
    switch_X["ppo_destination_diff"] = (ppo.destination != base.destination).astype(int)
    switch_X["ppo_mode_diff"] = (ppo["mode"] != base["mode"]).astype(int)
    for feature in ("eta_raw_min", "cand_cap_remain", "cand_occ_ratio", "cand_p_sent"):
        switch_X[f"ppo_minus_base_{feature}"] = ppo[feature] - base[feature]
    switch_result = _fit_tree("planner_switch", switch_X, decision.switch,
                              exe.reset_index().region, out)
    metrics = pd.DataFrame([
        {k: v for k, v in result.items() if k != "importance"}
        for result in (class_result, mode_result, switch_result)
    ])
    metrics.to_csv(out / "explanation_tree_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([x["importance"] for x in (class_result, mode_result, switch_result)]).to_csv(
        out / "explanation_tree_importance.csv", index=False, encoding="utf-8-sig"
    )

    # 병원 변경 때 어떤 속성을 얻고 잃는지 동일상태 paired contrast로 계산한다.
    changed = decision.destination_change.to_numpy(bool)
    contrast_rows = []
    for feature in (
        "eta_raw_min", "eta_rank", "cand_arrive_min", "cand_cap_remain",
        "cand_occ_ratio", "cand_p_sent", "cand_in_flight", "uav_advantage_min",
    ):
        delta = (exe[feature] - base[feature]).to_numpy(float)[changed]
        lo, hi = _bootstrap(delta)
        contrast_rows.append({
            "feature": feature, "exec_minus_gbdt_mean": float(delta.mean()),
            "bootstrap_lo": lo, "bootstrap_hi": hi, "n": len(delta),
        })
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(out / "destination_correction_contrasts.csv", index=False, encoding="utf-8-sig")

    # UAV 시간절감량 구간별 실제 실행률: 임계값을 자의적으로 고르지 않고 분위수로 제시한다.
    bins = pd.qcut(exe.uav_advantage_min, q=6, duplicates="drop")
    uav_bins = (
        pd.DataFrame({"bin": bins.astype(str), "uav": (exe["mode"] == 1).astype(float),
                      "adv": exe.uav_advantage_min})
        .groupby("bin", observed=True, as_index=False)
        .agg(n=("uav", "size"), uav_rate=("uav", "mean"),
             advantage_mean=("adv", "mean"), advantage_min=("adv", "min"),
             advantage_max=("adv", "max"))
        .sort_values("advantage_mean")
    )
    uav_bins.to_csv(out / "uav_advantage_bins.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(uav_bins.advantage_mean, uav_bins.uav_rate, marker="o", color="#2878b5")
    axes[0].set_xlabel("UAV 시간절감량 평균(분)")
    axes[0].set_ylabel("최종정책 UAV 선택률")
    axes[0].set_title("시간절감량에 따른 UAV 선택")
    axes[0].grid(alpha=.2)
    c = contrast.sort_values("exec_minus_gbdt_mean")
    xerr = np.vstack([
        c.exec_minus_gbdt_mean - c.bootstrap_lo,
        c.bootstrap_hi - c.exec_minus_gbdt_mean,
    ])
    axes[1].barh(c.feature, c.exec_minus_gbdt_mean, xerr=xerr, color="#d97627", capsize=3)
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set_title("병원 교정 시 실행안 - GBDT 기준안")
    fig.tight_layout()
    fig.savefig(out / "guideline_evidence.png", dpi=220)
    plt.close(fig)

    quality = {
        "input": str(path), "rows": len(d), "decisions": len(exe),
        "regions": int(exe.reset_index().region.nunique()), "roles_complete": True,
        "fit_coordinate_role": "p3 explanation/development; not final performance",
        "group_cv": "5-fold by district coordinate", "tree_depth": 3,
        "lb_t_included": False,
        "interpretation_limit": "Shallow trees summarize behavior; thresholds require closed-loop validation before guideline claims.",
    }
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rates, ensure_ascii=False, indent=2))
    print("\n", metrics.to_string(index=False))
    print("\n", contrast.to_string(index=False))
    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
