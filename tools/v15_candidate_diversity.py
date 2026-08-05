# -*- coding: utf-8 -*-
"""v15 다중정책 후보 포트폴리오의 오프라인 다양성 감사.

최종 교사 p3 결정 로그의 동일 상태에서 PPO top-K, MILP 제안, 증류 학생들이
각각 어떤 행동을 제안하는지 비교한다. 이 단계는 성능 실험이 아니라 후보 확장의
필요성과 중복도를 확인하는 데이터 품질 게이트다. LB-T 계열은 의도적으로 읽지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/rl_src"))

from tree_distill_policy import FEATURE_NAMES, decode_action, load_tree_package, tree_scores

DEFAULT_DATA = REPO / "results/scoreboard/v13/sota_distill/data/hybrid_val250_p3_seed7000.npz"
DEFAULT_TREE_DIR = REPO / "results/scoreboard/v13/sota_distill/students_full1000"
DEFAULT_OUT = REPO / "results/scoreboard/v15/candidate_diversity"
DEFAULT_CASES = [
    "I3_CONNECTED_GBDT_L63_BASE",
    "I3_CONNECTED_GBDT_L31_BASE",
    "I1_FIELD_GBDT_L63_BASE",
    "I1_FIELD_GBDT_L31_BASE",
    "I1_FIELD_EBM_I04",
    "I3_CONNECTED_CART_L384",
]


def _choose_scores(actions: np.ndarray, X: np.ndarray, score: np.ndarray) -> int:
    best = np.flatnonzero(np.isclose(score, score.max(), rtol=0.0, atol=1e-12))
    if len(best) > 1:
        stay = X[best, FEATURE_NAMES.index("is_stay")]
        eta = X[best, FEATURE_NAMES.index("eta_rank")]
        order = np.lexsort((actions[best], eta, stay))
        j = int(best[order[0]])
    else:
        j = int(best[0])
    return int(actions[j])


def _ppo_topk(actions: np.ndarray, probs: np.ndarray, k: int) -> list[int]:
    """planner_policy와 같은 top-K + stay 목적지 중복 제거."""
    order = np.argsort(-probs)
    out: list[int] = []
    seen_stay = False
    for j in order[:k]:
        a = int(actions[j])
        if probs[j] <= 0:
            continue
        _, dest, _ = decode_action(a, 192, 47)
        if int(dest) == 0:
            if seen_stay:
                continue
            seen_stay = True
        out.append(a)
    greedy = int(actions[int(np.argmax(probs))])
    if greedy not in out:
        out.append(greedy)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--tree_dir", default=str(DEFAULT_TREE_DIR))
    p.add_argument("--cases", default=",".join(DEFAULT_CASES))
    p.add_argument("--extra_packages", default="", help="name=/abs/path.pkl 쉼표구분")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    data_path = Path(args.data).resolve()
    tree_dir = Path(args.tree_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [x for x in args.cases.split(",") if x]
    packages = {case: load_tree_package(str(tree_dir / f"{case}.pkl")) for case in cases}
    for spec in (x for x in args.extra_packages.split(",") if x):
        name, path = spec.split("=", 1)
        if name in packages:
            raise ValueError(f"패키지 이름 중복: {name}")
        packages[name] = load_tree_package(str(Path(path).resolve()))
        cases.append(name)

    z = np.load(data_path, allow_pickle=True)
    required = {
        "X", "cand_action", "ppo_prob", "offsets", "teacher_action", "ppo_action",
        "milp_action0", "milp_action1", "state_key", "state_seed", "decision_index",
    }
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"결정 로그 컬럼 누락: {sorted(missing)}")
    X = np.asarray(z["X"], dtype=np.float32)
    cand = np.asarray(z["cand_action"], dtype=int)
    prob = np.asarray(z["ppo_prob"], dtype=float)
    offsets = np.asarray(z["offsets"], dtype=np.int64)
    n = len(offsets) - 1
    if offsets[0] != 0 or offsets[-1] != len(X) or len(cand) != len(X) or len(prob) != len(X):
        raise ValueError("후보행/offsets 정합성 실패")
    if not np.isfinite(X).all() or not np.isfinite(prob).all():
        raise ValueError("비유한 특징 또는 PPO 확률")
    keys = list(zip(z["state_key"].astype(str), z["state_seed"].astype(int), z["decision_index"].astype(int)))
    if len(set(keys)) != n:
        raise ValueError("상태 복합키 중복")

    teacher = np.asarray(z["teacher_action"], dtype=int)
    ppo = np.asarray(z["ppo_action"], dtype=int)
    milp0 = np.asarray(z["milp_action0"], dtype=int)
    milp1 = np.asarray(z["milp_action1"], dtype=int)
    pred = {case: np.empty(n, dtype=int) for case in cases}
    # 앙상블 추론은 상태별 수만 번 호출하지 않고 전체 후보행을 모델당 한 번만 수행한다.
    # 상태별 선택은 아래에서 offsets로 잘라 동일한 tie-break를 적용한다.
    all_scores = {case: tree_scores(package, X) for case, package in packages.items()}
    topk_sets: list[set[int]] = []
    base_sets: list[set[int]] = []

    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        ai, xi, pi = cand[lo:hi], X[lo:hi], prob[lo:hi]
        if len(ai) == 0 or len(set(ai.tolist())) != len(ai):
            raise ValueError(f"상태 {i} 유효후보 0 또는 action 중복")
        tk = set(_ppo_topk(ai, pi, args.K))
        base = set(tk)
        for a in (milp0[i], milp1[i]):
            if a >= 0:
                if a not in set(ai.tolist()):
                    raise ValueError(f"상태 {i} MILP 행동이 유효후보에 없음: {a}")
                base.add(int(a))
        topk_sets.append(tk)
        base_sets.append(base)
        for case in cases:
            pred[case][i] = _choose_scores(ai, xi, all_scores[case][lo:hi])

    rows = []
    for case in cases:
        a = pred[case]
        novel_topk = np.fromiter((int(x) not in topk_sets[i] for i, x in enumerate(a)), bool, n)
        novel_base = np.fromiter((int(x) not in base_sets[i] for i, x in enumerate(a)), bool, n)
        c = a // 96
        rem = a % 96
        d, m = rem // 2, rem % 2
        rows.append({
            "policy": case,
            "n_states": n,
            "agreement_teacher": float(np.mean(a == teacher)),
            "agreement_ppo": float(np.mean(a == ppo)),
            "novel_vs_ppo_topk": float(np.mean(novel_topk)),
            "novel_vs_ppo_topk_milp": float(np.mean(novel_base)),
            "teacher_hit_when_novel": float(np.mean(a[novel_base] == teacher[novel_base])) if novel_base.any() else np.nan,
            "red_rate": float(np.mean(c == 0)),
            "uav_rate_transport": float(np.mean(m[d > 0] == 1)) if np.any(d > 0) else np.nan,
            "stay_rate": float(np.mean(d == 0)),
        })
    summary = pd.DataFrame(rows).sort_values(["agreement_teacher", "novel_vs_ppo_topk_milp"], ascending=[False, False])
    summary.to_csv(out_dir / "policy_candidate_summary.csv", index=False, encoding="utf-8-sig")

    names = ["PPO"] + cases
    all_actions = {"PPO": ppo, **pred}
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for x in names:
        for y in names:
            agree.loc[x, y] = float(np.mean(all_actions[x] == all_actions[y]))
    agree.to_csv(out_dir / "pairwise_action_agreement.csv", encoding="utf-8-sig")

    union_rows = []
    ordered = sorted(cases, key=lambda x: float(summary.set_index("policy").loc[x, "agreement_teacher"]), reverse=True)
    gbdt = [x for x in ordered if "GBDT" in x]
    family = []
    for token in ("GBDT", "EBM", "CART"):
        hit = next((x for x in ordered if token in x), None)
        if hit is not None:
            family.append(hit)
    subsets = [("BASE", []), ("TOP_GBDT", gbdt[:1]), ("TOP3_GBDT", gbdt[:3])]
    if family:
        subsets.append(("FAMILY_BEST", family))
    subsets.append(("ALL_SELECTED", ordered))
    for subset_name, subset in subsets:
        sizes, novel, coverage = [], [], []
        for i in range(n):
            base = set(base_sets[i])
            ext = {int(pred[x][i]) for x in subset}
            merged = base | ext
            sizes.append(len(merged))
            novel.append(len(ext - base))
            coverage.append(int(teacher[i]) in merged)
        union_rows.append({
            "candidate_set": subset_name,
            "members": "|".join(subset),
            "mean_candidate_count": float(np.mean(sizes)),
            "mean_novel_actions": float(np.mean(novel)),
            "states_with_any_novel": float(np.mean(np.asarray(novel) > 0)),
            "teacher_action_coverage": float(np.mean(coverage)),
        })
    union = pd.DataFrame(union_rows)
    union.to_csv(out_dir / "candidate_union_summary.csv", index=False, encoding="utf-8-sig")

    # 정책별 새 후보율과 교사일치도: '다른 후보'와 '좋은 모방'을 동시에 본다.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(summary["novel_vs_ppo_topk_milp"] * 100, summary["agreement_teacher"] * 100, s=80)
    for r in summary.itertuples(index=False):
        ax.annotate(r.policy.replace("_CONNECTED", "").replace("_FIELD", ""),
                    (r.novel_vs_ppo_topk_milp * 100, r.agreement_teacher * 100),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("기존 PPO top-K + MILP 밖의 새 행동 제안률 (%)")
    ax.set_ylabel("최종 교사 행동 일치율 (%)")
    ax.set_title("증류 정책의 후보 다양성 감사 (random4 p3 250개, seed 7000)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_diversity.png", dpi=180)
    plt.close(fig)

    quality = {
        "data": str(data_path),
        "tree_dir": str(tree_dir),
        "n_states": n,
        "n_candidate_rows": int(len(X)),
        "n_unique_state_keys": len(set(keys)),
        "finite_features": bool(np.isfinite(X).all()),
        "finite_probabilities": bool(np.isfinite(prob).all()),
        "lb_t_included": False,
        "interpretation": "후보 다양성 게이트이며 폐루프 성능 증거가 아님",
    }
    (out_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\n", union.to_string(index=False))
    print(f"\n완료 → {out_dir}")


if __name__ == "__main__":
    main()
