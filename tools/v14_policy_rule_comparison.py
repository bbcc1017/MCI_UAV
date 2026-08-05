# -*- coding: utf-8 -*-
"""PPO 기본정책과 최종 하이브리드 교사의 규칙을 나란히 비교한다.

분석 단위를 섞지 않는다.

* PPO on-policy: PPO가 실제로 방문한 p0~p2 / p3 궤적
* final on-policy: PPO+NCRP+MILP가 실제로 방문한 p0~p2 / p3 궤적
* PPO@final-state: final 궤적의 동일 상태에서 PPO가 냈을 행동

첫 두 보기는 정책이 유도한 상태분포까지 포함한 기술통계다. 세 번째 보기만
동일 상태의 보정 비교이므로 NCRP·MILP의 행동 변경을 직접 설명할 수 있다.
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/rl_src"))
sys.path.insert(0, str(REPO / "tools"))

import v13_sota_rule_analysis as v13


PPO_TRAIN = REPO / "results/scoreboard/v10/distill/data/ppo_train1000_seed5000.npz"
PPO_VAL = REPO / "results/scoreboard/v10/distill/data/ppo_val250_p3_seed7000.npz"
FINAL_TRAIN = REPO / "results/scoreboard/v13/sota_distill/data/hybrid_train750_p0p2_seed5000.npz"
FINAL_VAL = REPO / "results/scoreboard/v13/sota_distill/data/hybrid_val250_p3_seed7000.npz"
DEFAULT_OUT = REPO / "results/scoreboard/v14/policy_rule_comparison"

POLICY_LABELS = {
    "PPO_ON_POLICY": "PPO 기본정책",
    "PPO_AT_FINAL_STATES": "동일 상태의 PPO",
    "FINAL_TEACHER": "PPO+NCRP+MILP 최종교사",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode_action(action: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(action, dtype=int)
    cls = x // 96
    rem = x % 96
    return cls, rem // 2, rem % 2


def base_region(key: str) -> str:
    return re.sub(r"_p[0-3]$", "", str(key))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def subset_states(z: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """ragged 후보행 NPZ를 상태 마스크로 무손실 부분집합화한다."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(z["ncand"]),):
        raise ValueError("상태 마스크 길이 불일치")
    state_idx = np.flatnonzero(keep)
    row_idx = np.concatenate([
        np.arange(int(z["offsets"][i]), int(z["offsets"][i + 1])) for i in state_idx
    ])
    state_keys = {
        "ncand", "teacher_action", "behavior_action", "ppo_action", "state_key",
        "state_seed", "decision_index", "teacher_switched", "teacher_in_milp",
        "planner_lookahead", "planner_dpdr", "planner_q_greedy", "planner_q_exec",
        "planner_n_cand", "planner_n_extra", "milp_action0", "milp_action1",
    }
    row_keys = {"X", "target", "weight", "chosen", "cand_action", "ppo_prob"}
    out: dict[str, np.ndarray] = {}
    for key, value in z.items():
        if key in row_keys:
            out[key] = value[row_idx]
        elif key in state_keys:
            out[key] = value[state_idx]
        elif key == "offsets":
            continue
        else:
            out[key] = value.copy()
    out["offsets"] = np.concatenate([[0], np.cumsum(out["ncand"], dtype=np.int64)])
    return out


def action_view(z: dict[str, np.ndarray], action_key: str) -> dict[str, np.ndarray]:
    """같은 후보 특징에서 지정 행동을 양성으로 바꾼 정책 관점."""
    actions = np.asarray(z[action_key], dtype=np.int16)
    chosen = np.zeros(len(z["X"]), dtype=bool)
    for i, (start, end) in enumerate(zip(z["offsets"][:-1], z["offsets"][1:])):
        hit = np.flatnonzero(z["cand_action"][start:end] == actions[i])
        if len(hit) != 1:
            raise ValueError(f"{action_key} 후보 매칭 실패 state={i}")
        chosen[int(start) + int(hit[0])] = True
    out = dict(z)
    out["teacher_action"] = actions
    out["chosen"] = chosen
    out["target"] = chosen.astype(np.float32)
    # 분석에는 쓰지 않지만 데이터 계약을 보존한다.
    weight = np.empty(len(chosen), dtype=np.float32)
    for start, end in zip(z["offsets"][:-1], z["offsets"][1:]):
        n = int(end - start)
        local = chosen[start:end]
        weight[start:end] = 0.5 / max(n - 1, 1)
        weight[start:end][local] = 0.5
    out["weight"] = weight
    return out


def validate_and_frame(
    z: dict[str, np.ndarray], *, policy: str, dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(z["offsets"][-1]) != len(z["X"]) or int(z["ncand"].sum()) != len(z["X"]):
        raise ValueError(f"{policy}/{dataset}: offsets 불일치")
    if not np.isfinite(z["X"]).all() or not np.isfinite(z["weight"]).all():
        raise ValueError(f"{policy}/{dataset}: 비유한 특징/가중치")
    positives = np.add.reduceat(z["chosen"].astype(np.int64), z["offsets"][:-1])
    if not np.all(positives == 1):
        raise ValueError(f"{policy}/{dataset}: 상태별 양성행이 1개가 아님")
    chosen_pos = np.flatnonzero(z["chosen"])
    if not np.array_equal(z["cand_action"][chosen_pos], z["teacher_action"]):
        raise ValueError(f"{policy}/{dataset}: 양성행과 행동 불일치")

    both_class = np.zeros(len(z["ncand"]), dtype=bool)
    both_mode = np.zeros(len(z["ncand"]), dtype=bool)
    for i, (start, end) in enumerate(zip(z["offsets"][:-1], z["offsets"][1:])):
        actions = z["cand_action"][start:end]
        cls, dest, mode = decode_action(actions)
        c, d, m = (int(x) for x in decode_action(int(z["teacher_action"][i])))
        both_class[i] = bool(
            np.any((cls == 0) & (dest > 0)) and np.any((cls == 1) & (dest > 0))
        )
        both_mode[i] = bool(np.any((cls == c) & (dest == d) & (mode != m)))

    names = [str(x) for x in z["feature_names"]]
    chosen_x = z["X"][chosen_pos]
    cls, dest, mode = decode_action(z["teacher_action"])
    keys = np.asarray(z["state_key"]).astype(str)
    if "decision_index" in z:
        decision_index = z["decision_index"].astype(int)
    else:
        decision_index = (
            pd.DataFrame({"key": keys, "seed": z["state_seed"].astype(int)})
            .groupby(["key", "seed"], sort=False).cumcount().to_numpy(int)
        )
    frame = pd.DataFrame({
        "policy": policy,
        "dataset": dataset,
        "state_key": keys,
        "region_base": [base_region(x) for x in keys],
        "state_seed": z["state_seed"].astype(int),
        "decision_index": decision_index,
        "selected_class": cls.astype(int),
        "selected_dest": dest.astype(int),
        "selected_mode": mode.astype(int),
        "both_class_available": both_class,
        "both_mode_available": both_mode,
    })
    for j, name in enumerate(names):
        frame[name] = chosen_x[:, j]
    ids = frame[["state_key", "state_seed", "decision_index"]]
    if ids.duplicated().any():
        raise ValueError(f"{policy}/{dataset}: 상태키 중복")
    quality = {
        "policy": policy,
        "dataset": dataset,
        "states": int(len(frame)),
        "candidate_rows": int(len(z["X"])),
        "coordinates": int(frame.state_key.nunique()),
        "base_regions": int(frame.region_base.nunique()),
        "feature_count": len(names),
        "duplicate_state_ids": 0,
        "states_with_one_positive": int(len(frame)),
        "pdr_feature_finite": True,
    }
    return frame, quality


def load_views() -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], pd.DataFrame, dict[str, Any]]:
    ppo_train_all = load_npz(PPO_TRAIN)
    keep = np.asarray([
        str(x).endswith(("_p0", "_p1", "_p2")) for x in ppo_train_all["state_key"]
    ])
    ppo_train = subset_states(ppo_train_all, keep)
    ppo_val = load_npz(PPO_VAL)
    final_train = load_npz(FINAL_TRAIN)
    final_val = load_npz(FINAL_VAL)

    raw = {
        "PPO_ON_POLICY": {"train": ppo_train, "validation": ppo_val},
        "FINAL_TEACHER": {"train": final_train, "validation": final_val},
        "PPO_AT_FINAL_STATES": {
            "train": action_view(final_train, "ppo_action"),
            "validation": action_view(final_val, "ppo_action"),
        },
    }
    frames, quality = [], []
    for policy, split in raw.items():
        for dataset, z in split.items():
            frame, q = validate_and_frame(z, policy=policy, dataset=dataset)
            frames.append(frame)
            quality.append(q)

    # p0~p2와 p3의 좌표 분리는 정책별로 확인한다.
    for policy in raw:
        tr = set(next(x for x in frames if x.policy.iloc[0] == policy and x.dataset.iloc[0] == "train").state_key)
        va = set(next(x for x in frames if x.policy.iloc[0] == policy and x.dataset.iloc[0] == "validation").state_key)
        if tr & va:
            raise ValueError(f"{policy}: train/validation 좌표키 중복")

    # PPO와 final의 p0~p2 좌표, p3 좌표, 초기 seed가 같은지 확인한다.
    checks = {}
    for dataset in ("train", "validation"):
        a = raw["PPO_ON_POLICY"][dataset]
        b = raw["FINAL_TEACHER"][dataset]
        checks[f"{dataset}_coordinate_set_equal"] = bool(
            set(a["state_key"].astype(str)) == set(b["state_key"].astype(str))
        )
        checks[f"{dataset}_seed_set_equal"] = bool(
            set(a["state_seed"].astype(int)) == set(b["state_seed"].astype(int))
        )
    if not all(checks.values()):
        raise ValueError(f"PPO/final 좌표·seed 정합성 실패: {checks}")
    return raw, pd.concat(frames, ignore_index=True), {
        "status": "pass", "views": quality, "cross_policy_checks": checks,
        "source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in (PPO_TRAIN, PPO_VAL, FINAL_TRAIN, FINAL_VAL)},
        "interpretation_guardrail": (
            "PPO_ON_POLICY vs FINAL_TEACHER is descriptive because trajectories diverge; "
            "PPO_AT_FINAL_STATES vs FINAL_TEACHER is the matched-state correction comparison"
        ),
    }


def equal_region_rate(frame: pd.DataFrame, mask: np.ndarray, value: np.ndarray) -> dict[str, Any]:
    x = pd.DataFrame({
        "region": frame.loc[mask, "region_base"].to_numpy(),
        "value": np.asarray(value)[np.asarray(mask, dtype=bool)].astype(float),
    })
    per = x.groupby("region", observed=True).value.mean()
    lo, hi = v13.bootstrap_ci(per)
    return {
        "rate_equal_region": float(per.mean()),
        "rate_pooled": float(x.value.mean()),
        "boot_lo": lo, "boot_hi": hi,
        "n_regions": int(len(per)), "n_decisions": int(len(x)),
    }


def behavior_rates(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, dataset), frame in frames.groupby(["policy", "dataset"], sort=False):
        specs = [
            ("Red 선택(양 등급 가능)", frame.both_class_available.to_numpy(), frame.selected_class.to_numpy() == 0),
            ("UAV 선택(이송행동)", frame.selected_dest.to_numpy() > 0, frame.selected_mode.to_numpy() == 1),
            ("Red 환자 UAV", (frame.selected_dest.to_numpy() > 0) & (frame.selected_class.to_numpy() == 0), frame.selected_mode.to_numpy() == 1),
            ("Yellow 환자 UAV", (frame.selected_dest.to_numpy() > 0) & (frame.selected_class.to_numpy() == 1), frame.selected_mode.to_numpy() == 1),
            ("현장 대기", np.ones(len(frame), dtype=bool), frame.selected_dest.to_numpy() == 0),
        ]
        for metric, mask, value in specs:
            rows.append({"policy": policy, "dataset": dataset, "metric": metric, **equal_region_rate(frame, mask, value)})
    return pd.DataFrame(rows)


def conditional_bins(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adv_edges = [-np.inf, 0, 5, 10, 15, 20, 30, np.inf]
    adv_labels = ["≤0", "0–5", "5–10", "10–15", "15–20", "20–30", ">30"]
    fleet_edges = [-np.inf, 4, 8, 12, 16, 20, np.inf]
    fleet_labels = ["≤4", "5–8", "9–12", "13–16", "17–20", ">20"]
    yellow_edges = [-np.inf, 9, 13, 19, np.inf]
    yellow_labels = ["0–9", "10–13", "14–19", "20+"]
    for (policy, dataset), frame in frames.groupby(["policy", "dataset"], sort=False):
        moved = frame[(frame.selected_dest > 0) & frame.both_mode_available].copy()
        moved["bin"] = pd.cut(moved.uav_advantage_min, adv_edges, labels=adv_labels, include_lowest=True)
        moved["class"] = np.where(moved.selected_class == 0, "Red", "Yellow")
        for (cls, label), group in moved.groupby(["class", "bin"], observed=True):
            rec = equal_region_rate(group, np.ones(len(group), dtype=bool), group.selected_mode.to_numpy() == 1)
            rows.append({"policy": policy, "dataset": dataset, "condition": "UAV 시간절감(분)", "stratum": str(label), "patient_class": cls, "outcome": "UAV 선택", **rec})

        both = frame[frame.both_class_available].copy()
        both["bin"] = pd.cut(both.fleet_critical, fleet_edges, labels=fleet_labels, include_lowest=True)
        for label, group in both.groupby("bin", observed=True):
            rec = equal_region_rate(group, np.ones(len(group), dtype=bool), group.selected_class.to_numpy() == 0)
            rows.append({"policy": policy, "dataset": dataset, "condition": "이송 중·복귀 중 차량수", "stratum": str(label), "patient_class": "R/Y 경쟁", "outcome": "Red 선택", **rec})

        # 교사의 중증도 선택을 직접 규칙화하기 위한 현장 Yellow 적체 구간.
        both["yellow_bin"] = pd.cut(
            both.yellow_at_site, yellow_edges, labels=yellow_labels, include_lowest=True,
        )
        for label, group in both.groupby("yellow_bin", observed=True):
            rec = equal_region_rate(
                group, np.ones(len(group), dtype=bool), group.selected_class.to_numpy() == 0,
            )
            rows.append({
                "policy": policy, "dataset": dataset,
                "condition": "현장 Yellow 대기자수", "stratum": str(label),
                "patient_class": "R/Y 경쟁", "outcome": "Red 선택", **rec,
            })

        both["amb_bin"] = np.where(both.amb_available > 0.5, "AMB 가용", "AMB 불가")
        for label, group in both.groupby("amb_bin", observed=True):
            rec = equal_region_rate(
                group, np.ones(len(group), dtype=bool), group.selected_class.to_numpy() == 0,
            )
            rows.append({
                "policy": policy, "dataset": dataset,
                "condition": "현장 AMB 가용성", "stratum": str(label),
                "patient_class": "R/Y 경쟁", "outcome": "Red 선택", **rec,
            })
    return pd.DataFrame(rows)


def fit_axis_trees(frames: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, rules = [], []
    specs = [
        ("class", v13.CLASS_FEATURES, "select_red"),
        ("mode", v13.MODE_FEATURES, "select_uav"),
    ]
    for policy in ("PPO_ON_POLICY", "FINAL_TEACHER"):
        train = frames[(frames.policy == policy) & (frames.dataset == "train")].copy()
        val = frames[(frames.policy == policy) & (frames.dataset == "validation")].copy()
        train["select_red"] = (train.selected_class == 0).astype(int)
        val["select_red"] = (val.selected_class == 0).astype(int)
        train["select_uav"] = (train.selected_mode == 1).astype(int)
        val["select_uav"] = (val.selected_mode == 1).astype(int)
        for axis, features, target in specs:
            if axis == "class":
                tm, vm = train.both_class_available, val.both_class_available
            else:
                tm = (train.selected_dest > 0) & train.both_mode_available
                vm = (val.selected_dest > 0) & val.both_mode_available
            tr, va = train.loc[tm].copy(), val.loc[vm].copy()
            model = DecisionTreeClassifier(
                max_depth=3, max_leaf_nodes=8,
                min_samples_leaf=max(100, int(0.01 * len(tr))),
                class_weight="balanced", random_state=20260803,
            )
            weights = 1.0 / tr.groupby("region_base", observed=True).region_base.transform("size").to_numpy(float)
            weights /= weights.mean()
            model.fit(tr[features].to_numpy(float), tr[target].to_numpy(int), sample_weight=weights)
            pred = model.predict(va[features].to_numpy(float))
            prob = model.predict_proba(va[features].to_numpy(float))
            auc = float("nan")
            if len(np.unique(va[target])) == 2 and 1 in model.classes_:
                auc = float(roc_auc_score(va[target], prob[:, list(model.classes_).index(1)]))
            metrics.append({
                "policy": policy, "axis": axis,
                "n_train": int(len(tr)), "n_validation": int(len(va)),
                "positive_rate_train": float(tr[target].mean()),
                "positive_rate_validation": float(va[target].mean()),
                "validation_balanced_accuracy": float(balanced_accuracy_score(va[target], pred)),
                "validation_auc": auc,
                "root_feature": features[int(model.tree_.feature[0])],
                "root_threshold": float(model.tree_.threshold[0]),
                "depth": int(model.get_depth()), "leaves": int(model.get_n_leaves()),
            })
            text = export_text(model, feature_names=features, decimals=2)
            (out_dir / f"{policy.lower()}_{axis}_tree.txt").write_text(text, encoding="utf-8")
            for feature, importance in zip(features, model.feature_importances_):
                rules.append({"policy": policy, "axis": axis, "feature": feature, "importance": float(importance)})
    return pd.DataFrame(metrics), pd.DataFrame(rules)


def hospital_models(
    raw: dict[str, dict[str, dict[str, np.ndarray]]], frames: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coef_all, metric_all, strata_all = [], [], []
    for policy in ("PPO_ON_POLICY", "FINAL_TEACHER"):
        train = frames[(frames.policy == policy) & (frames.dataset == "train")].reset_index(drop=True)
        val = frames[(frames.policy == policy) & (frames.dataset == "validation")].reset_index(drop=True)
        coef, metrics, strata = v13.conditional_choice_analysis(
            raw[policy]["train"], train, raw[policy]["validation"], val,
        )
        for table in (coef, metrics, strata):
            table.insert(0, "policy", policy)
        coef_all.append(coef); metric_all.append(metrics); strata_all.append(strata)
    return pd.concat(coef_all, ignore_index=True), pd.concat(metric_all, ignore_index=True), pd.concat(strata_all, ignore_index=True)


def matched_corrections(raw: dict[str, dict[str, dict[str, np.ndarray]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, strata = [], []
    for dataset in ("train", "validation"):
        z = raw["FINAL_TEACHER"][dataset]
        fc, fd, fm = decode_action(z["teacher_action"])
        pc, pdest, pm = decode_action(z["ppo_action"])
        changed = z["teacher_action"] != z["ppo_action"]
        n = len(changed)
        types = {
            "행동 전체": changed,
            "환자등급": fc != pc,
            "병원": fd != pdest,
            "이송수단": fm != pm,
            "병원만": (fd != pdest) & (fc == pc) & (fm == pm),
            "둘 이상 축": ((fc != pc).astype(int) + (fd != pdest).astype(int) + (fm != pm).astype(int)) >= 2,
            "MILP 후보 채택": z["teacher_in_milp"].astype(bool),
        }
        keys = np.asarray(z["state_key"]).astype(str)
        base = np.asarray([base_region(x) for x in keys])
        for name, values in types.items():
            frame = pd.DataFrame({"region_base": base})
            rec = equal_region_rate(frame, np.ones(n, dtype=bool), values)
            rows.append({"dataset": dataset, "change_type": name, **rec})

        # 같은 상태에서 무엇이 달라지는지 고부하 구간별로 본다.
        chosen_pos = np.flatnonzero(z["chosen"])
        names = [str(x) for x in z["feature_names"]]
        fleet = z["X"][chosen_pos, names.index("fleet_critical")]
        bins = pd.cut(fleet, [-np.inf, 4, 8, 12, 16, 20, np.inf], labels=["≤4", "5–8", "9–12", "13–16", "17–20", ">20"])
        tmp = pd.DataFrame({"region_base": base, "bin": bins, "changed": changed, "class_changed": fc != pc, "dest_changed": fd != pdest, "mode_changed": fm != pm})
        for label, group in tmp.groupby("bin", observed=True):
            for outcome in ("changed", "class_changed", "dest_changed", "mode_changed"):
                rec = equal_region_rate(group, np.ones(len(group), dtype=bool), group[outcome].to_numpy())
                strata.append({"dataset": dataset, "fleet_bin": str(label), "outcome": outcome, **rec})
    return pd.DataFrame(rows), pd.DataFrame(strata)


def plot_action_rates(rates: pd.DataFrame, out: Path) -> None:
    configure_font()
    data = rates[rates.dataset == "validation"].copy()
    metrics = ["Red 선택(양 등급 가능)", "UAV 선택(이송행동)", "현장 대기"]
    policies = ["PPO_ON_POLICY", "PPO_AT_FINAL_STATES", "FINAL_TEACHER"]
    palette = {"PPO_ON_POLICY": "#2B6F9F", "PPO_AT_FINAL_STATES": "#8C99A5", "FINAL_TEACHER": "#D18B2C"}
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(metrics)); width = 0.23
    for j, policy in enumerate(policies):
        g = data[data.policy == policy].set_index("metric").loc[metrics]
        xpos = x + (j - 1) * width
        ax.bar(xpos, g.rate_equal_region, width,
               yerr=[g.rate_equal_region - g.boot_lo, g.boot_hi - g.rate_equal_region],
               label=POLICY_LABELS[policy], color=palette[policy], edgecolor="#39434D", capsize=3)
        for xi, value in zip(xpos, g.rate_equal_region):
            ax.text(xi, value + 0.012, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("지역 동일가중 행동 비율")
    ax.set_ylim(0, 0.47)
    ax.set_title("PPO 기본판단과 최종교사 행동 비율")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uav_bins(bins: pd.DataFrame, out: Path) -> None:
    configure_font()
    data = bins[(bins.dataset == "validation") & (bins.condition == "UAV 시간절감(분)")]
    order = ["0–5", "5–10", "10–15", "15–20", "20–30", ">30"]
    policies = ["PPO_ON_POLICY", "PPO_AT_FINAL_STATES", "FINAL_TEACHER"]
    palette = {"PPO_ON_POLICY": "#2B6F9F", "PPO_AT_FINAL_STATES": "#8C99A5", "FINAL_TEACHER": "#D18B2C"}
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for policy in policies:
        g = data[(data.patient_class == "Red") & (data.policy == policy)].set_index("stratum").reindex(order)
        ax.plot(order, g.rate_equal_region, marker="o", linewidth=2, color=palette[policy], label=POLICY_LABELS[policy])
    ax.set_xlabel("UAV 시간절감 구간(분)")
    ax.set_ylabel("UAV 선택률")
    ax.set_ylim(0, 1.03)
    ax.grid(color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Red 환자: UAV 시간절감이 커질수록 UAV 선택 증가")
    ax.text(
        0.01, 0.98,
        "p3에서 동일 목적지의 두 수단이 모두 유효한 Yellow 결정은 17건으로 적어 그림에서 제외",
        transform=ax.transAxes, va="top", fontsize=8, color="#59636D",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_priority(bins: pd.DataFrame, out: Path) -> None:
    configure_font()
    data = bins[
        (bins.dataset == "validation")
        & (bins.condition == "현장 Yellow 대기자수")
    ]
    order = ["0–9", "10–13", "14–19", "20+"]
    policies = ["PPO_ON_POLICY", "PPO_AT_FINAL_STATES", "FINAL_TEACHER"]
    palette = {
        "PPO_ON_POLICY": "#2B6F9F", "PPO_AT_FINAL_STATES": "#8C99A5",
        "FINAL_TEACHER": "#D18B2C",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for policy in policies:
        g = data[data.policy == policy].set_index("stratum").reindex(order)
        ax.plot(
            order, g.rate_equal_region, marker="o", linewidth=2.2,
            color=palette[policy], label=POLICY_LABELS[policy],
        )
    ax.set_xlabel("현장 Yellow 대기자수")
    ax.set_ylabel("Red 선택률 (Red·Yellow 모두 선택 가능)")
    ax.set_ylim(0, 0.55)
    ax.set_title("Yellow 적체가 커질수록 Yellow 이송 비중이 증가")
    ax.grid(color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hospital_coefficients(coef: pd.DataFrame, out: Path) -> None:
    configure_font()
    data = coef[coef.feature.isin(v13.CHOICE_FEATURES)].copy()
    feature_label = {
        "eta_rank": "ETA 순위", "cand_p_sent_rel": "상대 누적발송",
        "cand_in_flight": "해당 병원 이송 중", "cand_occ_ratio": "병원 점유비",
        "max_send": "병원 발송상한",
    }
    order = list(feature_label)
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    y = np.arange(len(order)); width = 0.34
    for j, (policy, color) in enumerate((("PPO_ON_POLICY", "#2B6F9F"), ("FINAL_TEACHER", "#D18B2C"))):
        g = data[data.policy == policy].set_index("feature").loc[order]
        ax.barh(y + (j - 0.5) * width, g.standardized_coef_train_all, width,
                label=POLICY_LABELS[policy], color=color, edgecolor="#39434D")
    ax.axvline(0, color="#39434D", linewidth=1)
    ax.set_yticks(y, [feature_label[x] for x in order])
    ax.set_xlabel("조건부 병원선택 표준화 계수")
    ax.set_title("같은 환자등급·이송수단 안에서 선호한 병원 조건")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def configure_font() -> None:
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def write_report(
    out: Path, rates: pd.DataFrame, trees: pd.DataFrame, choice_coef: pd.DataFrame,
    choice_metrics: pd.DataFrame, corrections: pd.DataFrame, bins: pd.DataFrame,
    quality: dict[str, Any],
) -> None:
    val = rates[rates.dataset == "validation"].pivot(index="metric", columns="policy", values="rate_equal_region")
    tree = trees.set_index(["policy", "axis"])
    cm = choice_metrics[choice_metrics.fit == "train_all_to_p3_external"].set_index("policy")
    corr = corrections[corrections.dataset == "validation"].set_index("change_type")
    class_bins = bins[
        (bins.dataset == "validation")
        & (bins.condition == "현장 Yellow 대기자수")
        & (bins.policy == "FINAL_TEACHER")
    ].set_index("stratum")
    report = f"""# PPO 기본규칙과 NCRP·MILP 보정규칙의 분리 분석

## 기술 요약

- PPO와 최종교사는 같은 43개 현장 특징을 사용하지만 서로 다른 궤적을 만든다. 정책 전체 비교는 기술통계로, 최종교사 로그 안의 PPO 행동과 최종 행동 비교만 동일 상태 보정효과로 해석했다.
- p3 검증좌표에서 양 등급이 모두 가능한 경우 Red 선택률은 PPO on-policy **{val.loc['Red 선택(양 등급 가능)','PPO_ON_POLICY']:.1%}**, 최종교사 **{val.loc['Red 선택(양 등급 가능)','FINAL_TEACHER']:.1%}**였다.
- 이송행동 중 UAV 선택률은 PPO on-policy **{val.loc['UAV 선택(이송행동)','PPO_ON_POLICY']:.1%}**, 최종교사 **{val.loc['UAV 선택(이송행동)','FINAL_TEACHER']:.1%}**였다. 절대 비율보다 UAV 시간절감 구간별 반응을 가이드라인 근거로 사용한다.
- 동일한 최종교사 상태에서 최종 행동은 PPO 행동을 **{corr.loc['행동 전체','rate_equal_region']:.1%}** 변경했다. 병원 변경 **{corr.loc['병원','rate_equal_region']:.1%}**, 환자등급 변경 **{corr.loc['환자등급','rate_equal_region']:.1%}**, 수단 변경 **{corr.loc['이송수단','rate_equal_region']:.1%}**로 목적지 보정이 가장 큰 축이다.

## 기본판단과 최종행동은 환자·수단 배분도 바꾼다

![정책별 행동 비율](action_rates_comparison.png)

회색 막대는 최종교사가 방문한 바로 그 상태에서 PPO가 냈을 행동이다. 파란색 PPO on-policy와 회색의 차이는 정책이 만든 상태분포 차이이고, 회색과 주황색의 차이만 NCRP·MILP가 같은 상태에서 가한 보정이다.

## 환자등급은 Red 고정우선이 아니라 현장 적체에 따라 바뀌었다

![Yellow 대기자수와 Red 선택률](class_priority_policy_comparison.png)

최종교사의 Red 선택률은 Yellow 대기자 0–9명에서 **{class_bins.loc['0–9','rate_equal_region']:.1%}**, 10–13명 **{class_bins.loc['10–13','rate_equal_region']:.1%}**, 14–19명 **{class_bins.loc['14–19','rate_equal_region']:.1%}**, 20명 이상 **{class_bins.loc['20+','rate_equal_region']:.1%}**로 낮아졌다. 즉 Yellow가 누적될수록 Yellow 이송 비중을 높여 추가 적체를 방지하는 패턴이다. 이는 인과율이 아닌 관찰 패턴이며, 폐루프 재시뮬레이션에서 성능을 따로 검증했다.

## UAV 가치는 절대적인 UAV 우선이 아니라 시간절감 조건으로 나타난다

![UAV 시간절감 구간](uav_advantage_policy_comparison.png)

각 환자등급에서 AMB와 UAV가 모두 가능한 결정만 포함했다. 이 표면은 전국 공통 가이드라인의 핵심 후보이며, 단일 임계값은 p0~p2에서 고정한 뒤 p3 및 폐루프 재시뮬레이션으로 검증해야 한다.

## 병원 선택은 거리와 부하의 결합 문제다

![병원 조건부 계수](hospital_choice_coefficients.png)

동일 상태·환자등급·이송수단 안의 유효 병원끼리 비교한 조건부 softmax 계수다. PPO p3 top-1={cm.loc['PPO_ON_POLICY','top1_accuracy']:.1%}, 최종교사 p3 top-1={cm.loc['FINAL_TEACHER','top1_accuracy']:.1%}다. 누적발송량과 발송상한은 정책 행동의 결과 또는 대리변수일 수 있으므로, 인과적 권고에는 ETA와 실시간 점유비를 우선 사용한다.

## 축소 규칙의 검증력

| 정책 | 축 | p3 balanced accuracy | p3 AUC | 첫 분기 |
|---|---|---:|---:|---|
| PPO | 환자등급 | {tree.loc[('PPO_ON_POLICY','class'),'validation_balanced_accuracy']:.3f} | {tree.loc[('PPO_ON_POLICY','class'),'validation_auc']:.3f} | {tree.loc[('PPO_ON_POLICY','class'),'root_feature']} ≤ {tree.loc[('PPO_ON_POLICY','class'),'root_threshold']:.2f} |
| PPO | 이송수단 | {tree.loc[('PPO_ON_POLICY','mode'),'validation_balanced_accuracy']:.3f} | {tree.loc[('PPO_ON_POLICY','mode'),'validation_auc']:.3f} | {tree.loc[('PPO_ON_POLICY','mode'),'root_feature']} ≤ {tree.loc[('PPO_ON_POLICY','mode'),'root_threshold']:.2f} |
| 최종교사 | 환자등급 | {tree.loc[('FINAL_TEACHER','class'),'validation_balanced_accuracy']:.3f} | {tree.loc[('FINAL_TEACHER','class'),'validation_auc']:.3f} | {tree.loc[('FINAL_TEACHER','class'),'root_feature']} ≤ {tree.loc[('FINAL_TEACHER','class'),'root_threshold']:.2f} |
| 최종교사 | 이송수단 | {tree.loc[('FINAL_TEACHER','mode'),'validation_balanced_accuracy']:.3f} | {tree.loc[('FINAL_TEACHER','mode'),'validation_auc']:.3f} | {tree.loc[('FINAL_TEACHER','mode'),'root_feature']} ≤ {tree.loc[('FINAL_TEACHER','mode'),'root_threshold']:.2f} |

축소 트리는 교사의 행동을 완전히 복제하는 모델이 아니라 반복되는 조건을 찾는 사후 설명기다. 임계값 자체를 정책으로 채택하려면 별도 폐루프 실험이 필요하다.

## 데이터와 해석 한계

- 분석학습: random4 p0~p2 750좌표, 검증: p3 250좌표. 최종 대표점 250개는 규칙 적합에 사용하지 않았다.
- PPO와 최종교사 모두 train/validation 좌표키가 분리됐고, 두 정책의 좌표·초기 seed 집합이 일치했다.
- PPO와 최종교사의 후속 상태는 행동 때문에 달라진다. 따라서 on-policy 비율 차이를 NCRP·MILP의 순수 인과효과로 읽지 않는다.
- 현장 대기와 목적지 선택은 마스크·가용자원·병원 적격성에 조건부다. 보고된 비율의 분모를 각 표와 CSV에 보존했다.

## 다음 검증

1. AMB는 도로 ETA와 실시간 병원부하, UAV는 헬기장 후보의 항공 ETA를 사용하는 모드별 병원규칙을 분리한다.
2. UAV 확보를 위해 추가된 원거리 헬기장병원이 AMB에 강제되지 않도록 AMB 후보를 도로 접근성으로 제한한다.
3. LB-T3는 결합하지 않고 별도 휴리스틱 기준선으로만 비교한다.
4. 규칙을 고정한 뒤 p3와 신규 외부좌표에서 paired 재시뮬레이션한다.
"""
    (out / "technical_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw, frames, quality = load_views()
    rates = behavior_rates(frames)
    bins = conditional_bins(frames)
    trees, importance = fit_axis_trees(frames, out)
    choice_coef, choice_metrics, choice_strata = hospital_models(raw, frames)
    corrections, correction_strata = matched_corrections(raw)

    tables = {
        "policy_behavior_rates.csv": rates,
        "conditional_action_bins.csv": bins,
        "axis_tree_metrics.csv": trees,
        "axis_tree_importance.csv": importance,
        "hospital_choice_coefficients.csv": choice_coef,
        "hospital_choice_metrics.csv": choice_metrics,
        "hospital_choice_strata.csv": choice_strata,
        "matched_correction_rates.csv": corrections,
        "matched_correction_by_load.csv": correction_strata,
    }
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    (out / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    chart_map = {
        "action_rates_comparison.png": {"family": "grouped bar", "source": "policy_behavior_rates.csv"},
        "class_priority_policy_comparison.png": {"family": "multi-series line", "source": "conditional_action_bins.csv"},
        "uav_advantage_policy_comparison.png": {"family": "faceted line", "source": "conditional_action_bins.csv"},
        "hospital_choice_coefficients.png": {"family": "grouped horizontal bar", "source": "hospital_choice_coefficients.csv"},
    }
    (out / "chart_map.json").write_text(json.dumps(chart_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_action_rates(rates, out / "action_rates_comparison.png")
    plot_class_priority(bins, out / "class_priority_policy_comparison.png")
    plot_uav_bins(bins, out / "uav_advantage_policy_comparison.png")
    plot_hospital_coefficients(choice_coef, out / "hospital_choice_coefficients.png")
    write_report(out, rates, trees, choice_coef, choice_metrics, corrections, bins, quality)
    print("[v14-policy-rule] 데이터 품질 PASS")
    print(rates[rates.dataset == "validation"][["policy", "metric", "rate_equal_region", "n_decisions"]].to_string(index=False))
    print(f"산출물 → {out}")


if __name__ == "__main__":
    main()
