# -*- coding: utf-8 -*-
"""v15 통합정책 이득을 사전 지역·인프라 특징으로 설명하는 공간 일반화 분석.

정책 결과를 입력 특징으로 쓰지 않는다. 병원/Tier3/헬기장 구성, 도로·직선 이송시간,
AMB/UAV 초기 접근성, 치료용량만 시나리오 원천파일에서 추출한다. 시군구를 무작위로
섞지 않고 광역시도 코드 GroupKFold로 분리해 인접지역 누수를 완화한다.

이 분석은 어느 지역에서 후보 포트폴리오가 더 유용한지를 설명하는 것이며 의료적 인과효과나
실제 사망감소를 직접 추정하지 않는다. LB-T3는 특징·정책 어디에도 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EFFECT = REPO / "results/scoreboard/v15/final/analysis/final_region_effects.csv"
DEFAULT_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
DEFAULT_OUT = REPO / "results/scoreboard/v15/final/analysis/region_profile"
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def _resolve(path: str, cfg: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    # 현행 YAML은 저장 위치가 아니라 repo 루트 기준 './scenarios/...' 경로다.
    repo_path = (REPO / p).resolve()
    return repo_path if repo_path.exists() else (cfg.parent / p).resolve()


def _q(x, q: float) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(a, q)) if len(a) else math.nan


def _effective_amb(amb: pd.DataFrame, n: int) -> pd.DataFrame:
    if "보유대수" not in amb:
        return amb.head(n).reset_index(drop=True)
    count = pd.to_numeric(amb["보유대수"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    return amb.loc[amb.index.repeat(count.to_numpy())].head(n).reset_index(drop=True)


def extract_profile(region: str, cfg_path: str) -> dict:
    cfg = Path(cfg_path).resolve()
    y = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    entity = y["entity_info"]
    hp = entity["hospital"]
    ap = entity["ambulance"]
    up = entity["uav"]
    hospital = pd.read_csv(_resolve(hp["info_path"], cfg))
    amb_all = pd.read_csv(_resolve(ap["dispatch_distance_info"], cfg))
    uav_all = pd.read_csv(_resolve(up["dispatch_distance_info"], cfg))
    amb = _effective_amb(amb_all, int(ap.get("amb_num", len(amb_all))))
    uav = uav_all.head(int(up.get("uav_num", len(uav_all)))).reset_index(drop=True)

    tier3 = pd.to_numeric(hospital["종별코드"], errors="coerce").eq(1).to_numpy()
    helipad = pd.to_numeric(hospital["헬기장 여부"], errors="coerce").fillna(0).eq(1).to_numpy()
    road_km = pd.to_numeric(hospital["road_dist"], errors="coerce").to_numpy(float)
    euc_km = pd.to_numeric(hospital["euc_dist"], errors="coerce").to_numpy(float)
    # 현행 OSRM 시뮬은 is_use_time=False이므로 거리/설정속도에서 이송시간을 계산한다.
    amb_hospital_min = road_km * 60.0 / float(ap["velocity"])
    uav_hospital_min = euc_km * 60.0 / float(up["velocity"])
    capacity = (
        pd.to_numeric(hospital["수술실수"], errors="coerce").fillna(0).to_numpy(float)
        + pd.to_numeric(hospital["병상수"], errors="coerce").fillna(0).to_numpy(float)
    )
    if len(amb):
        amb_dispatch = pd.to_numeric(amb["init_distance"], errors="coerce").to_numpy(float) \
            * 60.0 / float(ap["velocity"])
    else:
        amb_dispatch = np.array([], dtype=float)
    if len(uav):
        uav_dispatch = pd.to_numeric(uav["init_distance"], errors="coerce").to_numpy(float) \
            * 60.0 / float(up["velocity"])
    else:
        uav_dispatch = np.array([], dtype=float)

    t3h = tier3 & helipad
    code = re.search(r"_(\d{5})(?:_|$)", region)
    sigcd = code.group(1) if code else "00000"
    rec = {
        "region": region,
        "sigcd": sigcd,
        "province_code": sigcd[:2],
        "admin_type": "군" if "군" in region.rsplit("_", 1)[0] else "시·구",
        "incident_lat": float(entity["patient"]["latitude"]),
        "incident_lon": float(entity["patient"]["longitude"]),
        "n_hospital": len(hospital),
        "n_tier3": int(tier3.sum()),
        "n_helipad": int(helipad.sum()),
        "n_tier3_helipad": int(t3h.sum()),
        "tier3_helipad_fraction": float(t3h.sum() / max(tier3.sum(), 1)),
        "capacity_total": float(capacity.sum()),
        "capacity_tier3": float(capacity[tier3].sum()),
        "capacity_helipad": float(capacity[helipad].sum()),
        "amb_dispatch_p10": _q(amb_dispatch, .1),
        "amb_dispatch_median": _q(amb_dispatch, .5),
        "amb_dispatch_p90": _q(amb_dispatch, .9),
        "uav_dispatch_p10": _q(uav_dispatch, .1),
        "uav_dispatch_median": _q(uav_dispatch, .5),
        "uav_dispatch_p90": _q(uav_dispatch, .9),
        "amb_hospital_all_min": _q(amb_hospital_min, 0),
        "amb_hospital_all_median": _q(amb_hospital_min, .5),
        "amb_hospital_t3_min": _q(amb_hospital_min[tier3], 0),
        "amb_hospital_t3_median": _q(amb_hospital_min[tier3], .5),
        "amb_hospital_helipad_median": _q(amb_hospital_min[helipad], .5),
        "uav_hospital_helipad_min": _q(uav_hospital_min[helipad], 0),
        "uav_hospital_helipad_median": _q(uav_hospital_min[helipad], .5),
        "uav_hospital_t3_helipad_min": _q(uav_hospital_min[t3h], 0),
        "uav_hospital_t3_helipad_median": _q(uav_hospital_min[t3h], .5),
        "uav_advantage_helipad_median": _q(
            amb_hospital_min[helipad] - uav_hospital_min[helipad], .5
        ),
        "uav_advantage_t3_helipad_median": _q(
            amb_hospital_min[t3h] - uav_hospital_min[t3h], .5
        ),
    }
    return rec


def _ridge(alpha: float):
    # y도 표준화하여 alpha가 효과크기 단위에 의존하지 않게 한다.
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        transformer=StandardScaler(),
    )


def _choose_alpha(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    if len(unique) < 3:
        return 10.0
    cv = GroupKFold(n_splits=min(4, len(unique)))
    best = (math.inf, 10.0)
    for alpha in ALPHAS:
        err = []
        for tr, va in cv.split(X, y, groups):
            model = _ridge(alpha).fit(X.iloc[tr], y[tr])
            err.extend(np.abs(y[va] - model.predict(X.iloc[va])))
        candidate = (float(np.mean(err)), alpha)
        if candidate < best:
            best = candidate
    return float(best[1])


def _spatial_cv(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.full(len(y), np.nan)
    rows = []
    permutation_rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        alpha = _choose_alpha(X.iloc[tr], y[tr], groups[tr])
        model = _ridge(alpha).fit(X.iloc[tr], y[tr])
        pred[va] = model.predict(X.iloc[va])
        rows.append({
            "fold": fold, "alpha": alpha, "n_train": len(tr), "n_test": len(va),
            "test_provinces": "+".join(sorted(set(groups[va]))),
            "mae": float(mean_absolute_error(y[va], pred[va])),
        })
        base_mae = float(mean_absolute_error(y[va], pred[va]))
        rng = np.random.default_rng(20260804 + fold)
        for feature in X.columns:
            deltas = []
            original = X.iloc[va][feature].to_numpy(copy=True)
            for _ in range(50):
                xp = X.iloc[va].copy()
                xp[feature] = rng.permutation(original)
                deltas.append(float(mean_absolute_error(y[va], model.predict(xp))) - base_mae)
            permutation_rows.append({
                "fold": fold, "feature": feature,
                "delta_mae": float(np.mean(deltas)),
            })
    permutation = pd.DataFrame(permutation_rows)
    return pred, pd.DataFrame(rows), permutation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--region_effects", default=str(DEFAULT_EFFECT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--allow_partial", action="store_true", help="개발/스모크 전용")
    args = p.parse_args()
    effect_path = Path(args.region_effects).resolve()
    manifest_path = Path(args.manifest).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    effect = pd.read_csv(effect_path)
    # 개발 분석기(v15_portfolio_results)의 열 이름도 스모크/방향 재현용으로 수용한다.
    effect = effect.rename(columns={"final_pdr": "teacher_pdr", "base_g1_pdr": "portfolio_pdr"})
    required = {"region", "teacher_pdr", "portfolio_pdr", "improvement"}
    if required - set(effect):
        raise ValueError(f"지역효과 컬럼 누락: {sorted(required-set(effect))}")
    if effect.duplicated("region").any() or effect[list(required - {"region"})].isna().any().any():
        raise ValueError("지역효과 중복 또는 결측")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = set(effect.region) - set(manifest)
    if missing:
        raise ValueError(f"manifest에 없는 지역: {sorted(missing)[:5]}")
    if not args.allow_partial and (len(effect) != 250 or set(effect.region) != set(manifest)):
        raise ValueError("최종 분석은 대표점 250개 완전격자만 허용")

    profiles = pd.DataFrame([extract_profile(r, manifest[r]) for r in effect.region])
    effect = effect.drop(columns=[
        c for c in ("sigcd", "province_code", "admin_type", "incident_lat", "incident_lon")
        if c in effect
    ])
    d = effect.merge(profiles, on="region", validate="one_to_one")
    feature_cols = [
        c for c in profiles.columns
        if c not in {"region", "sigcd", "province_code", "admin_type", "incident_lat", "incident_lon"}
        and profiles[c].nunique(dropna=False) > 1
    ]
    X = d[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    y = d.improvement.to_numpy(float)
    groups = d.province_code.astype(str).to_numpy()
    pred, folds, permutation = _spatial_cv(X, y, groups)
    rho, rho_p = spearmanr(y, pred)
    metrics = {
        "n_regions": len(d), "n_provinces": int(pd.Series(groups).nunique()),
        "n_features": len(feature_cols), "spatial_cv": "nested GroupKFold by province code",
        "cv_mae": float(mean_absolute_error(y, pred)),
        "cv_r2": float(r2_score(y, pred)),
        "cv_spearman_r": float(rho), "cv_spearman_p": float(rho_p),
        "target": "FINAL regional PDR - V15 regional PDR",
        "lb_t_included": False,
    }
    folds.to_csv(out / "spatial_cv_folds.csv", index=False, encoding="utf-8-sig")
    permutation_summary = (
        permutation.groupby("feature", as_index=False)
        .agg(mean_delta_mae=("delta_mae", "mean"),
             median_delta_mae=("delta_mae", "median"),
             positive_fold_fraction=("delta_mae", lambda x: float((x > 0).mean())),
             n_folds=("fold", "nunique"))
        .sort_values("mean_delta_mae", ascending=False)
    )
    permutation.to_csv(out / "spatial_permutation_importance_folds.csv", index=False, encoding="utf-8-sig")
    permutation_summary.to_csv(
        out / "spatial_permutation_importance.csv", index=False, encoding="utf-8-sig"
    )

    # 전체자료 설명계수: 예측성은 위 공간 CV만으로 판단하고, 아래 계수는 방향 요약에만 쓴다.
    alpha = _choose_alpha(X, y, groups)
    final = _ridge(alpha).fit(X, y)
    ridge = final.regressor_.named_steps["ridge"]
    coef = pd.DataFrame({"feature": feature_cols, "standardized_coef": ridge.coef_})
    coef["abs_coef"] = coef.standardized_coef.abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    coef.to_csv(out / "regional_ridge_coefficients.csv", index=False, encoding="utf-8-sig")

    # 상위 20% 수혜지역을 사전 인프라 특징으로 요약하는 깊이2 트리. 임계값은 가이드라인
    # 후보일 뿐이며 별도 좌표/재시뮬레이션 검증 전에는 운영 임계값으로 주장하지 않는다.
    high = (y >= np.quantile(y, .8)).astype(int)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    prob = np.zeros(len(y), dtype=float)
    pred_class = np.zeros(len(y), dtype=int)
    for tr, va in cv.split(X, high, groups):
        tree = DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=max(5, len(tr) // 12),
            class_weight="balanced", random_state=20260804,
        ).fit(X.iloc[tr], high[tr])
        pred_class[va] = tree.predict(X.iloc[va])
        if len(tree.classes_) == 2:
            prob[va] = tree.predict_proba(X.iloc[va])[:, list(tree.classes_).index(1)]
        else:
            prob[va] = float(tree.classes_[0])
    tree = DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=max(5, len(X) // 12),
        class_weight="balanced", random_state=20260804,
    ).fit(X, high)
    (out / "high_benefit_tree.txt").write_text(
        export_text(tree, feature_names=feature_cols, decimals=3), encoding="utf-8"
    )
    pd.DataFrame({
        "feature": feature_cols, "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        out / "high_benefit_tree_importance.csv", index=False, encoding="utf-8-sig"
    )
    metrics.update({
        "high_benefit_definition": "top 20% paired improvement",
        "high_benefit_group_balanced_accuracy": float(balanced_accuracy_score(high, pred_class)),
        "high_benefit_group_auc": float(roc_auc_score(high, prob)) if len(np.unique(high)) == 2 else math.nan,
        "final_ridge_alpha": alpha,
    })

    d["spatial_cv_predicted_improvement"] = pred
    d["high_benefit"] = high
    d.to_csv(out / "regional_profiles_and_effects.csv", index=False, encoding="utf-8-sig")
    (out / "regional_generalization_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    quality = {
        "region_effects": str(effect_path), "manifest": str(manifest_path),
        "complete_representative250": len(d) == 250 and set(effect.region) == set(manifest),
        "features_are_predecision_static": True,
        "spatial_split": "province-code GroupKFold",
        "interpretation_limit": "predictive regional heterogeneity, not causal mortality effect",
        "lb_t_included": False,
    }
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].scatter(y, pred, c=np.where(d.admin_type == "군", "#d97627", "#7f8c8d"), alpha=.75)
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    axes[0].plot(lim, lim, color="black", lw=.8)
    axes[0].set_xlabel("관측 paired 개선")
    axes[0].set_ylabel("광역시도 분리 CV 예측 개선")
    axes[0].set_title(f"지역특성 기반 외삽 가능성: r_s={rho:.2f}")
    top = coef.head(10).sort_values("standardized_coef")
    axes[1].barh(top.feature, top.standardized_coef, color=np.where(top.standardized_coef > 0, "#2878b5", "#c44e52"))
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set_title("지역 이득과 연관된 표준화 Ridge 계수")
    fig.tight_layout()
    fig.savefig(out / "regional_generalization.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"완료 → {out}")


if __name__ == "__main__":
    main()
