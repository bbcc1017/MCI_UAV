"""collect_decisions.py 로그를 통계적으로 해석한다 (피드백 #2).

축별 서로게이트 (자유선택 결정만):
  A. 우선순위(R vs Y)   — START 휴리스틱과 대조
  B. 이송수단(UAV vs AMB) — mode_R/mode_Y 와 대조
  C. 목적지(병원 라우팅)  — RL 의 주 차별점. 'load-balancing' 가설을 paired 검정.
  D. 분기(divergence)    — RL 이 휴리스틱과 다르게 행동하는 상태 특성화.

해석용 압축 피처셋: 환자/차량 집계 + 현장 카운트 + time + 병원 부하 집계
(개별 병원 180컬럼은 얕은 트리에 노이즈라 합/평균/최대로 요약).

출력:
  results/analysis/policy_analysis_<tag>.txt   (전체 리포트)
  results/analysis/fig_tree_<axis>_<tag>.png   (의사결정트리)
  results/analysis/fig_importance_<tag>.png    (feature importance)

사용:
  CUDA_VISIBLE_DEVICES="" python src/rl_src/analyze_policy.py --tag plan1nat_f3
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from plot_variant_eval import _set_korean_font

_set_korean_font()


# ---------------------------------------------------------------- data
def load(tag, out_dir):
    obs = np.load(os.path.join(out_dir, f"decisions_{tag}.npz"))["obs"]
    meta = pd.read_csv(os.path.join(out_dir, f"decisions_{tag}_meta.csv"))
    with open(os.path.join(out_dir, f"decisions_{tag}_labels.json"), encoding="utf-8") as f:
        info = json.load(f)
    labels = info["labels"]
    obs_df = pd.DataFrame(obs, columns=labels)
    return obs_df, meta, labels, info


def engineer_features(obs_df, labels):
    """해석용 압축 피처셋 (얕은 트리 친화)."""
    keep = ([c for c in labels if c.startswith("pa_")] +
            [c for c in labels if c.startswith("ve_")] +
            [c for c in labels if c.startswith("atsite_")] +
            ["n_amb_at_site", "n_uav_at_site", "time"])
    F = obs_df[keep].copy()
    occ = [c for c in labels if c.startswith("h") and c.endswith("_occ")]
    idle = [c for c in labels if c.startswith("h") and c.endswith("_idle")]
    queue = [c for c in labels if c.startswith("h") and c.endswith("_queue")]
    psent = [c for c in labels if c.startswith("psent_")]
    F["hosp_occ_sum"] = obs_df[occ].sum(1)
    F["hosp_occ_mean"] = obs_df[occ].mean(1)
    F["hosp_occ_max"] = obs_df[occ].max(1)
    F["hosp_idle_sum"] = obs_df[idle].sum(1)
    F["hosp_queue_sum"] = obs_df[queue].sum(1)
    F["psent_sum"] = obs_df[psent].sum(1)
    F["psent_max"] = obs_df[psent].max(1)
    return F


# ---------------------------------------------------------------- surrogate
def fit_surrogate(X, y, name, report, fig_path=None, class_names=None, max_depth=4):
    """DecisionTree + Logistic 적합, fidelity·중요도·트리 텍스트 기록."""
    n = len(y)
    vc = pd.Series(y).value_counts().to_dict()
    report.write(f"\n{'='*70}\n[{name}]  n={n}  타깃분포={vc}\n{'='*70}\n")
    if n < 40 or pd.Series(y).nunique() < 2:
        report.write("  → 표본 부족 또는 단일 클래스 (서로게이트 생략).\n")
        return None
    base = max(pd.Series(y).value_counts(normalize=True))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(20, n // 100),
                                  random_state=0)
    tree.fit(Xtr, ytr)
    acc_tr, acc_te = tree.score(Xtr, ytr), tree.score(Xte, yte)
    report.write(f"  DecisionTree(depth≤{max_depth}) fidelity: train={acc_tr:.3f} "
                 f"test={acc_te:.3f}  (다수결 baseline={base:.3f})\n")

    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    report.write("  상위 feature importance:\n")
    for f, v in imp[imp > 0].head(8).items():
        report.write(f"      {f:24s} {v:.3f}\n")

    report.write("  --- 트리 규칙 ---\n")
    txt = export_text(tree, feature_names=list(X.columns), max_depth=max_depth)
    report.write("    " + txt.replace("\n", "\n    ") + "\n")

    # 로지스틱 (표준화) — 부호/크기 해석
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(scaler.transform(Xtr), ytr)
    acc_lr = lr.score(scaler.transform(Xte), yte)
    coef = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    report.write(f"  LogisticRegression fidelity test={acc_lr:.3f}; 상위 |계수|:\n")
    for f, v in coef.head(8).items():
        report.write(f"      {f:24s} {v:+.3f}\n")

    if fig_path is not None:
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=list(X.columns),
                  class_names=class_names, filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=130)
        plt.close()
        report.write(f"  [트리 그림 저장: {fig_path}]\n")
    return imp


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="plan1nat_f3")
    ap.add_argument("--out_dir", default="results/analysis")
    args = ap.parse_args()

    obs_df, meta, labels, info = load(args.tag, args.out_dir)
    F = engineer_features(obs_df, labels)
    hos_props = info.get("hospital_props", {})
    report = io.StringIO()
    report.write(f"RL 정책 해석 리포트 — tag={args.tag}\n")
    report.write(f"결정 {len(meta)} 행, 지역 {meta['region'].nunique()}, "
                 f"에피소드/지역 {info.get('n_episodes')}\n")

    # 전역 일치율
    report.write(f"\n[전역 RL↔휴리스틱 일치] class={meta['agree_class'].mean():.3f} "
                 f"mode={meta['agree_mode'].mean():.3f} full={meta['agree_full'].mean():.3f}\n")

    # ---- 축 A: 우선순위 (R vs Y 둘 다 현장대기) ----
    both = (obs_df["atsite_Red"] > 0) & (obs_df["atsite_Yellow"] > 0)
    selA = both & meta["rl_class"].isin([0, 1]).values
    A = meta.loc[selA]
    report.write(f"\n##### 축 A — 우선순위 (R·Y 동시 대기, n={selA.sum()}) #####\n")
    if selA.sum() > 0:
        report.write(f"  이 부분집합 RL↔START 일치(class)={A['agree_class'].mean():.3f}  "
                     f"(RL이 Red 선택 비율={(A['rl_class']==0).mean():.3f})\n")
        fit_surrogate(F.loc[selA], (meta.loc[selA, "rl_class"] == 0).astype(int).values,
                      "A. 우선순위: Red 우선=1 vs Yellow=0", report,
                      os.path.join(args.out_dir, f"fig_tree_priority_{args.tag}.png"),
                      class_names=["Yellow", "Red"])

    # ---- 축 B: 이송수단 (free_mode) ----
    selB = (meta["free_mode"] == 1).values
    B = meta.loc[selB]
    report.write(f"\n##### 축 B — 이송수단 (AMB·UAV 동시 가용, n={selB.sum()}) #####\n")
    if selB.sum() > 0:
        report.write(f"  이 부분집합 RL UAV 사용률={(B['rl_mode']==1).mean():.3f}  "
                     f"휴리스틱 UAV={ (B['heur_mode']==1).mean():.3f}  일치={B['agree_mode'].mean():.3f}\n")
        fit_surrogate(F.loc[selB], (meta.loc[selB, "rl_mode"] == 1).astype(int).values,
                      "B. 이송수단: UAV=1 vs AMB=0", report,
                      os.path.join(args.out_dir, f"fig_tree_mode_{args.tag}.png"),
                      class_names=["AMB", "UAV"])

    # ---- 축 C: 목적지 라우팅 ----
    report.write(f"\n##### 축 C — 목적지 병원 라우팅 #####\n")
    sent = meta[(meta["rl_dest"] > 0) & (meta["heur_dest"] > 0)].copy()
    report.write(f"  (이송 결정 n={len(sent)}; dest 일치={ (sent['rl_dest']==sent['heur_dest']).mean():.3f})\n")
    # C-load: RL vs 휴리스틱이 고른 병원의 점유/p_sent/tier (동일 상태 paired)
    obs_arr = obs_df.values
    lab_idx = {c: i for i, c in enumerate(labels)}

    def hosp_stat(row, dest, stat):
        h = int(dest) - 1
        return obs_arr[row, lab_idx.get(f"h{h}_{stat}", 0)]

    rl_occ, heur_occ, rl_ps, heur_ps, rl_t3, heur_t3 = [], [], [], [], [], []
    for ridx in sent.index:
        reg = meta.at[ridx, "region"]
        t3 = set(hos_props.get(reg, {}).get("tier3_idx", []))
        rl_d, h_d = int(meta.at[ridx, "rl_dest"]), int(meta.at[ridx, "heur_dest"])
        rl_occ.append(hosp_stat(ridx, rl_d, "occ")); heur_occ.append(hosp_stat(ridx, h_d, "occ"))
        rl_ps.append(obs_arr[ridx, lab_idx.get(f"psent_{rl_d-1}", 0)])
        heur_ps.append(obs_arr[ridx, lab_idx.get(f"psent_{h_d-1}", 0)])
        rl_t3.append(int((rl_d - 1) in t3)); heur_t3.append(int((h_d - 1) in t3))
    rl_occ, heur_occ = np.array(rl_occ), np.array(heur_occ)
    rl_ps, heur_ps = np.array(rl_ps), np.array(heur_ps)
    report.write(f"  선택병원 점유율(occupied): RL 평균={rl_occ.mean():.2f} vs 휴리스틱={heur_occ.mean():.2f} "
                 f"(차이 {rl_occ.mean()-heur_occ.mean():+.2f})\n")
    report.write(f"  선택병원 누적이송(p_sent): RL 평균={rl_ps.mean():.2f} vs 휴리스틱={heur_ps.mean():.2f} "
                 f"(차이 {rl_ps.mean()-heur_ps.mean():+.2f})\n")
    report.write(f"  선택병원 Tier3 비율: RL={np.mean(rl_t3):.3f} vs 휴리스틱={np.mean(heur_t3):.3f}\n")
    diff = rl_occ - heur_occ
    if np.any(diff != 0):
        try:
            w, p = wilcoxon(rl_occ, heur_occ)
            report.write(f"  Wilcoxon(점유율 RL vs 휴리스틱): stat={w:.0f} p={p:.2e} "
                         f"→ {'RL이 덜 혼잡한 병원 선호 (load-balancing)' if rl_occ.mean()<heur_occ.mean() else 'RL이 더 혼잡 선호'}\n")
        except ValueError as e:
            report.write(f"  Wilcoxon 생략: {e}\n")

    # C-tier 서로게이트: RL 의 tier3 선택 예측
    selC = (meta["rl_dest"] > 0).values
    fit_surrogate(F.loc[selC], meta.loc[selC, "rl_dest_tier3"].clip(lower=0).astype(int).values,
                  "C. 목적지: Tier3=1 vs 비Tier3=0", report,
                  os.path.join(args.out_dir, f"fig_tree_dest_tier_{args.tag}.png"),
                  class_names=["non-T3", "Tier3"])

    # ---- 축 D: 분기 (RL ≠ 휴리스틱 full) ----
    report.write(f"\n##### 축 D — 분기 특성화 (RL≠휴리스틱 full 결정) #####\n")
    yD = (meta["agree_full"] == 0).astype(int).values
    fit_surrogate(F, yD, "D. 분기: 다름=1 vs 같음=0", report,
                  os.path.join(args.out_dir, f"fig_tree_divergence_{args.tag}.png"),
                  class_names=["same", "diff"])

    out_txt = os.path.join(args.out_dir, f"policy_analysis_{args.tag}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report.getvalue())
    print(report.getvalue())
    print(f"\n[리포트 저장: {out_txt}]", file=sys.stderr)


if __name__ == "__main__":
    main()
