# -*- coding: utf-8 -*-
"""v15 정책 후보 포트폴리오 평가의 클러스터-paired 통계와 도표."""
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
from scipy.stats import spearmanr, wilcoxon

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def bootstrap_ci(x, seed=20260804, n_boot=20000):
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, 1000):
        n = min(1000, n_boot - start)
        means.append(a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1))
    b = np.concatenate(means)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def paired_wtl(df: pd.DataFrame, method: str, ref: str) -> tuple[int, int, int]:
    p = df[df.policy.isin([method, ref])].pivot(
        index=["region", "seed"], columns="policy", values="pdr_woG"
    )
    wins = ties = losses = 0
    for _, g in p.groupby(level="region"):
        d = (g[ref] - g[method]).to_numpy(float)
        ci = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.inf
        if d.mean() > ci:
            wins += 1
        elif d.mean() < -ci:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm family-wise 보정; 원래 index를 보존한다."""
    p = pvalues.astype(float)
    order = p.sort_values().index
    adjusted = pd.Series(index=p.index, dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p.loc[idx])))
        adjusted.loc[idx] = running
    return adjusted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--refs", default="FINAL,PURE_G1")
    p.add_argument("--role", choices=["screen", "development", "final"], default="development")
    p.add_argument("--seed_min", type=int, default=None)
    p.add_argument("--seed_max", type=int, default=None)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    paths = [Path(x).resolve() for x in args.csv.split(",") if x]
    if not paths:
        raise ValueError("입력 CSV 0개")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if args.seed_min is not None:
        d = d[d.seed >= args.seed_min].copy()
    if args.seed_max is not None:
        d = d[d.seed <= args.seed_max].copy()
    if d.empty:
        raise ValueError("seed 필터 적용 후 행 0개")
    required = {"region", "policy", "seed", "pdr_woG", "n_decisions", "n_switch"}
    if required - set(d):
        raise ValueError(f"컬럼 누락: {sorted(required-set(d))}")
    if d.duplicated(["region", "policy", "seed"]).any():
        raise ValueError("(region, policy, seed) 중복")
    if d.isna().any().any() or not d.pdr_woG.between(0, 1).all():
        raise ValueError("결측 또는 PDR 범위 오류")
    policies = sorted(d.policy.unique())
    regions = sorted(d.region.unique())
    seeds = sorted(d.seed.unique())
    expected = len(policies) * len(regions) * len(seeds)
    if len(d) != expected:
        raise ValueError(f"완전격자 아님: {len(d)} != {expected}")
    if not (d.groupby("policy").size() == len(regions) * len(seeds)).all():
        raise ValueError("정책별 격자 불완전")

    region = d.groupby(["region", "policy"], as_index=False).pdr_woG.mean()
    score_rows = []
    for policy, g in region.groupby("policy"):
        x = g.pdr_woG.to_numpy(float)
        lo, hi = bootstrap_ci(x)
        raw = d[d.policy == policy]
        dec = max(float(raw.n_decisions.sum()), 1.0)
        score_rows.append({
            "policy": policy, "pdr_woG_mean": float(x.mean()),
            "region_ci95_halfwidth": float(1.96 * x.std(ddof=1) / math.sqrt(len(x))),
            "cluster_boot_lo": lo, "cluster_boot_hi": hi,
            "n_regions": len(x), "n_seeds": len(seeds),
            "switch_rate": float(raw.n_switch.sum() / dec),
            "tree_novel_exec_rate": float(raw.get("n_novel_tree_exec", pd.Series(0, index=raw.index)).sum() / dec),
            "milp_exec_rate": float(raw.get("n_milp_exec", pd.Series(0, index=raw.index)).sum() / dec),
            "ms_per_decision": float(raw.ms_per_decision.mean()) if "ms_per_decision" in raw else np.nan,
        })
    score = pd.DataFrame(score_rows).sort_values("pdr_woG_mean")
    score.to_csv(out / "portfolio_scoreboard.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    refs = [x for x in args.refs.split(",") if x in policies]
    for ref in refs:
        pv = d.pivot(index=["region", "seed"], columns="policy", values="pdr_woG")
        for policy in policies:
            if policy == ref:
                continue
            diff = (pv[ref] - pv[policy]).groupby(level="region").mean()
            lo, hi = bootstrap_ci(diff.to_numpy(float))
            try:
                pval = float(wilcoxon(diff, zero_method="wilcox").pvalue) if not np.allclose(diff, 0) else 1.0
            except ValueError:
                pval = 1.0
            w, t, l = paired_wtl(d, policy, ref)
            pair_rows.append({
                "method": policy, "reference": ref,
                "mean_pdr_reduction": float(diff.mean()),
                "bootstrap_lo": lo, "bootstrap_hi": hi,
                "wilcoxon_p": pval, "W": w, "T": t, "L": l,
            })
    pair = pd.DataFrame(pair_rows)
    if not pair.empty:
        pair["wilcoxon_holm_p"] = pair.groupby("reference", group_keys=False)["wilcoxon_p"].apply(
            holm_adjust
        )
        pair["significant_after_holm_0_05"] = pair.wilcoxon_holm_p < 0.05
    pair.to_csv(out / "portfolio_pairwise.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(score))))
    y = np.arange(len(score))
    err = score.region_ci95_halfwidth.to_numpy(float)
    colors = ["#9e9e9e" if x in {"PPO", "FINAL", "PURE_G1"} else "#2878b5" for x in score.policy]
    ax.barh(y, score.pdr_woG_mean, xerr=err, color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(y, score.policy)
    ax.invert_yaxis()
    ax.set_xlabel("평균 PDR_woG (낮을수록 우수)")
    role = {"screen": "탐색", "development": "개발", "final": "최종"}[args.role]
    ax.set_title(f"정책 후보 포트폴리오 {role} 비교 · {len(regions)}지역 × {len(seeds)} seed")
    ax.grid(axis="x", alpha=0.2)
    for yi, val in zip(y, score.pdr_woG_mean):
        ax.text(val + err[yi] + 0.001, yi, f"{val:.5f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "portfolio_scoreboard.png", dpi=200)
    plt.close(fig)

    # 절대 PDR의 지역 이질성이 큰 문제이므로 정책 간 paired 차이를 별도 도표로 제시한다.
    paired_final = pair[pair.reference == "FINAL"].sort_values("mean_pdr_reduction")
    if not paired_final.empty:
        fig, ax = plt.subplots(figsize=(9, max(4.0, 0.55 * len(paired_final))))
        y = np.arange(len(paired_final))
        mean = paired_final.mean_pdr_reduction.to_numpy(float)
        xerr = np.vstack([
            mean - paired_final.bootstrap_lo.to_numpy(float),
            paired_final.bootstrap_hi.to_numpy(float) - mean,
        ])
        ax.errorbar(mean, y, xerr=xerr, fmt="o", color="#2878b5", capsize=4)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(y, paired_final.method)
        ax.set_xlabel("FINAL 대비 평균 PDR 감소량 (양수=개선, 지역 bootstrap 95% CI)")
        ax.set_title(f"FINAL 대비 paired 정책효과 · {len(regions)}지역")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(out / "portfolio_paired_vs_final.png", dpi=200)
        plt.close(fig)

    heterogeneity = {}
    if "FINAL" in policies and "BASE_G1" in policies:
        pv = d.pivot(index=["region", "seed"], columns="policy", values="pdr_woG")
        reg = pd.DataFrame({
            "final_pdr": pv["FINAL"].groupby(level="region").mean(),
            "base_g1_pdr": pv["BASE_G1"].groupby(level="region").mean(),
        })
        reg["improvement"] = reg.final_pdr - reg.base_g1_pdr
        reg["admin_type"] = ["군" if "군" in x.rsplit("_", 2)[0] else "시·구" for x in reg.index]
        reg["difficulty_quintile"] = pd.qcut(
            reg.final_pdr.rank(method="first"), 5, labels=["Q1 쉬움", "Q2", "Q3", "Q4", "Q5 어려움"]
        )
        reg.reset_index().to_csv(out / "base_g1_region_effects.csv", index=False, encoding="utf-8-sig")
        bins = reg.groupby("difficulty_quintile", observed=True).agg(
            n=("improvement", "size"), final_pdr=("final_pdr", "mean"),
            improvement=("improvement", "mean"),
        ).reset_index()
        bins.to_csv(out / "base_g1_difficulty_bins.csv", index=False, encoding="utf-8-sig")
        rho, p_rho = spearmanr(reg.final_pdr, reg.improvement)
        heterogeneity = {
            "difficulty_improvement_spearman_rho": float(rho),
            "difficulty_improvement_spearman_p": float(p_rho),
            "military_county_improvement": float(reg.loc[reg.admin_type == "군", "improvement"].mean()),
            "city_district_improvement": float(reg.loc[reg.admin_type == "시·구", "improvement"].mean()),
        }
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for label, g in reg.groupby("admin_type"):
            ax.scatter(g.final_pdr, g.improvement, label=label, alpha=.75)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("기존 FINAL 지역 평균 PDR_woG")
        ax.set_ylabel("FINAL - BASE_G1 PDR (양수=BASE_G1 개선)")
        ax.set_title(f"지역 난이도와 포트폴리오 이득 · Spearman r_s={rho:.2f}")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / "base_g1_difficulty_effect.png", dpi=200)
        plt.close(fig)

    quality = {
        "inputs": [str(path) for path in paths], "role": args.role, "n_rows": len(d),
        "seed_filter": {"min": args.seed_min, "max": args.seed_max},
        "n_regions": len(regions), "n_seeds": len(seeds), "policies": policies,
        "complete_grid": True, "duplicate_keys": False, "finite_pdr": True,
        "lb_t_included": False,
        "heterogeneity": heterogeneity,
        "inference_limit": "screen/development 결과는 후보 선택용이며 최종 일반화 성능 주장이 아님"
        if args.role != "final" else "대표점 최종 평가",
    }
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(score.to_string(index=False))
    print("\n", pair.to_string(index=False))
    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
