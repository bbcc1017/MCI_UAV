"""피드백 #2·#4 결과 그림 (정책 해석 + 증류 성능).

fig_interpretation_<tag>.png  (2x2):
  (a) RL↔휴리스틱 결정 일치율 (축별)
  (b) 선택 병원 점유율 분포 RL vs 휴리스틱 + Tier3 사용률 (부하분산 발견)
  (c) 우선순위(축A) feature importance (해석 압축 피처)
  (d) 충실 트리 fidelity vs depth (정책 복잡성)

fig_distill_perf_<tag>.png  (1x2):  distill CSV 있을 때만
  휴리스틱 / 풀 PPO / 증류정책 지역별 mean_R + Δ vs 휴리스틱

사용:
  CUDA_VISIBLE_DEVICES="" python src/rl_src/plot_interpretation.py --tag plan1nat_f3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from analyze_policy import load, engineer_features
from plot_variant_eval import _set_korean_font

C_HEUR, C_PPO, C_DIST = "#888888", "#1f77b4", "#d62728"


def fig_interpretation(tag, out_dir):
    obs_df, meta, labels, info = load(tag, out_dir)
    F = engineer_features(obs_df, labels)
    hos_props = info.get("hospital_props", {})
    H = sum(1 for c in labels if c.startswith("h") and c.endswith("_occ"))

    fig, ax = plt.subplots(2, 2, figsize=(18, 11))

    # (a) 일치율 (축별)
    sent = meta[(meta["rl_dest"] > 0) & (meta["heur_dest"] > 0)]
    agree = {
        "우선순위\n(class)": meta["agree_class"].mean(),
        "이송수단\n(mode)": meta["agree_mode"].mean(),
        "목적지\n(dest)": (sent["rl_dest"] == sent["heur_dest"]).mean(),
        "전체\n(full)": meta["agree_full"].mean(),
    }
    bars = ax[0, 0].bar(list(agree.keys()), list(agree.values()),
                        color=["#4c78a8", "#54a24b", "#e45756", "#79706e"])
    for b, v in zip(bars, agree.values()):
        ax[0, 0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=11)
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].set_ylabel("RL = 휴리스틱 결정 비율")
    ax[0, 0].set_title("(a) RL↔휴리스틱 의사결정 일치율 — 라우팅에서 최대 분기")
    ax[0, 0].grid(axis="y", alpha=0.3)

    # (b) 선택 병원 점유율 + Tier3
    occ_mat = obs_df[[f"h{i}_occ" for i in range(H)]].values
    ps_mat = obs_df[[f"psent_{i}" for i in range(H)]].values
    rows = sent.index.values
    rl_d = sent["rl_dest"].values - 1
    he_d = sent["heur_dest"].values - 1
    rl_occ = occ_mat[rows, rl_d]
    he_occ = occ_mat[rows, he_d]
    ax[0, 1].hist([rl_occ, he_occ], bins=20, label=[f"RL (평균 {rl_occ.mean():.2f})",
                  f"휴리스틱 (평균 {he_occ.mean():.2f})"], color=[C_PPO, C_HEUR])
    ax[0, 1].set_xlabel("선택 병원의 현재 점유 환자 수 (n_occupied)")
    ax[0, 1].set_ylabel("결정 수")
    # Tier3 비율 inset
    rl_t3 = np.array([int(d in set(hos_props.get(meta.at[r, "region"], {}).get("tier3_idx", [])))
                      for r, d in zip(rows, rl_d)])
    he_t3 = np.array([int(d in set(hos_props.get(meta.at[r, "region"], {}).get("tier3_idx", [])))
                      for r, d in zip(rows, he_d)])
    ax[0, 1].set_title(f"(b) RL은 덜 혼잡한 병원으로 라우팅 (부하분산) | Tier3 사용 RL {rl_t3.mean():.0%} vs 휴리스틱 {he_t3.mean():.0%}")
    ax[0, 1].legend()
    ax[0, 1].grid(axis="y", alpha=0.3)

    # (c) 우선순위 feature importance (압축 피처, depth4)
    both = (obs_df["atsite_Red"] > 0) & (obs_df["atsite_Yellow"] > 0)
    selA = both.values & meta["rl_class"].isin([0, 1]).values
    yA = (meta.loc[selA, "rl_class"] == 0).astype(int).values
    tA = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=0).fit(F.loc[selA], yA)
    impA = pd.Series(tA.feature_importances_, index=F.columns).sort_values(ascending=False).head(8)[::-1]
    ax[1, 0].barh(impA.index, impA.values, color="#4c78a8")
    ax[1, 0].set_title(f"(c) 적응형 우선순위(R vs Y) 결정요인 — START 일치 {meta.loc[selA,'agree_class'].mean():.0%}")
    ax[1, 0].set_xlabel("feature importance (의사결정트리)")
    ax[1, 0].grid(axis="x", alpha=0.3)

    # (d) 충실 트리 fidelity vs depth
    _, encode_g = _make_codec(H)
    y_action = np.array([encode_g(c, d, m) for c, d, m in
                         zip(meta["rl_class"], meta["rl_dest"], meta["rl_mode"])])
    Xtr, Xte, ytr, yte = train_test_split(obs_df.values, y_action, test_size=0.3, random_state=0)
    depths = [4, 6, 8, 12, 16]
    tr_acc, te_acc = [], []
    for dep in depths:
        t = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(Xtr, ytr)
        tr_acc.append(t.score(Xtr, ytr)); te_acc.append(t.score(Xte, yte))
    ax[1, 1].plot(depths, tr_acc, "o-", label="train", color="#bbbbbb")
    ax[1, 1].plot(depths, te_acc, "o-", label="test (재현 충실도)", color=C_DIST)
    ax[1, 1].axhline(1.0, ls=":", color="#444", lw=1)
    ax[1, 1].set_xlabel("의사결정트리 최대 깊이")
    ax[1, 1].set_ylabel("RL action 재현율")
    ax[1, 1].set_title("(d) 단일 트리로 RL 전체 정책 재현 한계 (라우팅 복잡성)")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    plt.suptitle(f"RL 정책 해석 — MaskablePPO ({tag}), 결정 {len(meta):,}개 / 17지역", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(out_dir, f"fig_interpretation_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[저장] {out}")


def _make_codec(H):
    n_dest, n_mode = H + 1, 2
    return (lambda a: (a // (n_dest * n_mode), (a % (n_dest * n_mode)) // n_mode, a % n_mode),
            lambda c, d, m: int(c) * (n_dest * n_mode) + int(d) * n_mode + int(m))


def fig_distill_perf(tag, out_dir, csv_path):
    df = pd.read_csv(csv_path).sort_values("PPO_vs_heur", ascending=False)
    x = np.arange(len(df)); w = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].bar(x - w, df["heur_R"], w, label="휴리스틱(best)", color=C_HEUR)
    ax[0].bar(x, df["PPO_R"], w, label="MaskablePPO(풀)", color=C_PPO)
    ax[0].bar(x + w, df["distill_R"], w, label="증류 해석규칙", color=C_DIST)
    ax[0].set_xticks(x); ax[0].set_xticklabels(df["region"], rotation=45)
    ax[0].set_ylabel("평균 보상 R"); ax[0].legend()
    ax[0].set_title("(a) 지역별 평균 보상")
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].set_ylim(df[["heur_R", "PPO_R", "distill_R"]].min().min() * 0.97,
                   df[["heur_R", "PPO_R", "distill_R"]].max().max() * 1.02)

    ax[1].plot(x, df["PPO_vs_heur"], "o-", label="풀 PPO", color=C_PPO)
    ax[1].plot(x, df["distill_vs_heur"], "s-", label="증류 규칙", color=C_DIST)
    ax[1].axhline(0, color=C_HEUR, lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(df["region"], rotation=45)
    ax[1].set_ylabel("Δ vs 휴리스틱"); ax[1].legend()
    ax[1].set_title(f"(b) 휴리스틱 대비 우위 | 평균 PPO {df['PPO_vs_heur'].mean():+.2f}, 증류 {df['distill_vs_heur'].mean():+.2f}")
    ax[1].grid(alpha=0.3)

    plt.suptitle(f"증류 해석규칙 vs 풀 RL vs 휴리스틱 — {tag}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, f"fig_distill_perf_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[저장] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="plan1nat_f3")
    ap.add_argument("--out_dir", default="results/analysis")
    ap.add_argument("--distill_csv", default="results/plan1nat_f3_distill_eval.csv")
    args = ap.parse_args()
    _set_korean_font()
    fig_interpretation(args.tag, args.out_dir)
    if os.path.exists(args.distill_csv):
        fig_distill_perf(args.tag, args.out_dir, args.distill_csv)
    else:
        print(f"[skip] distill CSV 없음: {args.distill_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
