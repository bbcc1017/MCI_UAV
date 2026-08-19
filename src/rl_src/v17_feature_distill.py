# -*- coding: utf-8 -*-
"""v17 순수 PPO 교사 → 논문형 특징증강 후보랭킹 트리 적합·충실도 사다리.

교사는 결합 알고리즘(NCRP·MILP)이 아닌 **순수 PPO** ``v10_random4_1000_pointer_s0``
하나다. 이미 수집된 교사 결정 데이터셋(43특징)을 재사용하고, v17 증강 25특징은
``v17_tree_features.augment_dataset`` 으로 같은 state 안에서 재계산한다. 따라서 이
스크립트는 시뮬레이터를 돌리지 않으며 대표점250 도 열지 않는다.

실험군은 두 축이다.

* 특징집합: BASE43 / +RANK / +CAT / +REL / +GLOBAL / AUG68 / AUGONLY
* 정보단계: I0AUG / I1AUG / I2AUG (I3 = AUG68)

모델은 기존 C1~C4 CART, v10 Track E 의 CART_L384·GBDT_L31 을 그대로 쓴다.

subcommands
  fit       : 실험군 병렬 적합 + 충실도 지표 + 중요도 저장
  logstats  : 교사 결정로그 자체의 순위별 선택비율(논문 Fig.9-10 대응)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

from tree_distill_policy import COMPLEXITY_SPECS, FEATURE_NAMES, decode_action, tree_scores
from v10_tree_distill import load_datasets
from v17_tree_features import (
    ALL_FEATURE_NAMES,
    AUG_FAMILY,
    AUG_NAMES,
    FEATURE_SCHEMA,
    augment_dataset,
    family_indices,
    info_levels_v2,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/scoreboard/v10/distill/data"
TRAIN_NPZ = DATA / "ppo_train1000_seed5000.npz"
VAL_NPZ = DATA / "ppo_val250_p3_seed7000.npz"
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"

_I_STAY = ALL_FEATURE_NAMES.index("is_stay")
_I_ETARANK = ALL_FEATURE_NAMES.index("eta_rank")
_I_RANK_ETA_ALL = ALL_FEATURE_NAMES.index("rank_eta_all")

_LEVELS = info_levels_v2()

FEATURE_SETS: dict[str, list[int]] = {
    "BASE43": list(range(len(FEATURE_NAMES))),
    "RANK": family_indices(["RANK"]),
    "CAT": family_indices(["CAT"]),
    "REL": family_indices(["REL"]),
    "GLOBAL": family_indices(["GLOBAL"]),
    "AUG68": list(range(len(ALL_FEATURE_NAMES))),
    "AUGONLY": family_indices(["RANK", "CAT", "REL", "GLOBAL"], base=False),
    "I0AUG": _LEVELS["I0_MINIMAL"],
    "I1AUG": _LEVELS["I1_FIELD"],
    "I2AUG": _LEVELS["I2_TELEMETRY"],
}

MODEL_SPECS: dict[str, dict] = {
    **{k: {"family": "cart", **v} for k, v in COMPLEXITY_SPECS.items()},
    "L384": {"family": "cart", "max_depth": 10, "max_leaf_nodes": 384, "min_samples_leaf": 40},
    "G31": {"family": "lgbm", "num_leaves": 31},
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_augmented(paths: list[str]) -> dict:
    data = load_datasets(paths)
    data["X"] = np.hstack([data["X"], augment_dataset(data["X"], data["offsets"])])
    if not np.isfinite(data["X"]).all():
        raise RuntimeError("증강 특징에 비유한 값")
    return data


def _pick_row(actions, X, scores) -> tuple[int, int]:
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0.0, atol=1e-12))
    if len(best) > 1:
        order = np.lexsort((actions[best], X[best, _I_ETARANK], X[best, _I_STAY]))
        row = int(best[order[0]])
    else:
        row = int(best[0])
    return row, int(actions[row])


ETA_RANK_BUCKETS = np.asarray([0.5, 1.5, 2.5, 5.5], dtype=np.float32)


def _fidelity_core(scores: np.ndarray, data: dict) -> dict:
    """행동 충실도 + 확률보존 + 논문식 출력공간 축소 충실도.

    논문(Appendix F.1)은 앰뷸런스를 DC 로 묶어 출력공간을 줄인 뒤 F1 0.83~0.92 를
    보고한다. 우리 문제의 대응 축소는 병원 인덱스를 **현장 기준 ETA 순위**로 묶는
    것이다. ``fidelity_reduced`` 는 (등급, 수단, ETA순위) 가 모두 일치한 비율,
    ``fidelity_reduced_bucket`` 은 순위를 0/1/2/3-5/6+ 로 더 굵게 묶은 비율이다.
    """
    off = data["offsets"]
    n = len(off) - 1
    full = cls_ok = dest_ok = mode_ok = top3 = rank_ok = 0
    red_ok = redb_ok = 0
    p_student = p_teacher = 0.0
    chance = 0.0
    r_teacher = r_student = 0.0
    for s in range(n):
        a, b = int(off[s]), int(off[s + 1])
        acts = data["cand_action"][a:b]
        Xs = data["X"][a:b]
        row, action = _pick_row(acts, Xs, scores[a:b])
        prob = data["target"][a:b]
        teacher = int(data["teacher_action"][s])
        t_row = int(np.flatnonzero(acts == teacher)[0])
        full += action == teacher
        ca, ct = decode_action(action, 192, 47), decode_action(teacher, 192, 47)
        cls_ok += ca[0] == ct[0]
        dest_ok += ca[1] == ct[1]
        mode_ok += ca[2] == ct[2]
        k = min(3, len(prob))
        top3 += row in set(np.argpartition(prob, -k)[-k:].tolist())
        p_student += float(prob[row])
        p_teacher += float(prob[t_row])
        chance += 1.0 / len(prob)
        rs, rt = float(Xs[row, _I_RANK_ETA_ALL]), float(Xs[t_row, _I_RANK_ETA_ALL])
        rank_ok += rs == rt
        r_student += rs
        r_teacher += rt
        same_cm = (ca[0] == ct[0]) and (ca[2] == ct[2])
        red_ok += same_cm and rs == rt
        redb_ok += same_cm and int(np.searchsorted(ETA_RANK_BUCKETS, rs, "right")) == int(
            np.searchsorted(ETA_RANK_BUCKETS, rt, "right"))
    return {
        "n_states": n,
        "fidelity_full": full / n,
        "fidelity_class": cls_ok / n,
        "fidelity_dest": dest_ok / n,
        "fidelity_mode": mode_ok / n,
        "fidelity_reduced": red_ok / n,
        "fidelity_reduced_bucket": redb_ok / n,
        "teacher_top3_hit": top3 / n,
        "teacher_prob_student": p_student / n,
        "teacher_prob_max": p_teacher / n,
        "prob_retention": (p_student / n) / max(p_teacher / n, 1e-12),
        "chance_full": chance / n,
        "eta_rank_match": rank_ok / n,
        "eta_rank_mean_student": r_student / n,
        "eta_rank_mean_teacher": r_teacher / n,
    }


def fidelity(package: dict, data: dict) -> dict:
    return _fidelity_core(tree_scores(package, data["X"]), data)


_I_ETA_NORM = ALL_FEATURE_NAMES.index("eta_norm")
_I_P_SENT = ALL_FEATURE_NAMES.index("cand_p_sent")
_I_IS_UAV = ALL_FEATURE_NAMES.index("is_uav")

RULE_BASELINES = ["RULE_NEAREST", "RULE_NEAREST_AMB", "RULE_LB3", "RULE_LEAST_SENT"]


def rule_scores(name: str, data: dict) -> np.ndarray:
    """트리 없이 손으로 쓴 기준규칙의 후보점수. 충실도 문맥용 하한선."""
    X = data["X"]
    stay = X[:, _I_STAY] > 0.5
    eta = X[:, _I_ETA_NORM].astype(np.float64)
    p_sent = X[:, _I_P_SENT].astype(np.float64)
    uav = X[:, _I_IS_UAV] > 0.5
    big = 1e6
    if name == "RULE_NEAREST":
        sc = -eta
    elif name == "RULE_NEAREST_AMB":
        sc = -eta - np.where(uav, big, 0.0)
    elif name == "RULE_LB3":
        sc = -eta - np.where(p_sent >= 3.0, big, 0.0)
    elif name == "RULE_LEAST_SENT":
        sc = -(p_sent * 1e3 + eta)
    else:
        raise ValueError(name)
    return sc - np.where(stay, 2 * big, 0.0)


def remetric_main(args) -> None:
    """이미 적합된 패키지들의 지표를 재계산하고 손규칙 하한선을 함께 낸다."""
    val = load_augmented([str(Path(x).resolve()) for x in args.val_data.split(",") if x])
    fit_dir = Path(args.fit_dir).resolve()
    rows = list(csv.DictReader(open(fit_dir / "fit_summary.csv", encoding="utf-8-sig")))
    import pickle as _pk

    for r in rows:
        with open(r["pkl"], "rb") as f:
            pkg = _pk.load(f)
        m = fidelity(pkg, val)
        for k, v in m.items():
            r[k] = v
    for name in RULE_BASELINES:
        m = _fidelity_core(rule_scores(name, val), val)
        row = {k: "" for k in rows[0]}
        row.update({"policy": name, "feature_set": "RULE", "model": "-",
                    "family": "handrule", "n_features": 0, "n_features_used": 0,
                    "n_aug_used": 0, "depth": 0, "leaves": 0, "nodes": 0,
                    "fit_sec": 0, "pkl": "", "pkl_sha256": ""})
        row.update(m)
        rows.append(row)
    rows.sort(key=lambda x: -float(x["fidelity_full"]))
    out = fit_dir / "fit_summary_v2.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"  {r['policy']:<18} exact={float(r['fidelity_full']):.4f} "
              f"reduced={float(r['fidelity_reduced']):.4f} "
              f"reduced_b={float(r['fidelity_reduced_bucket']):.4f} "
              f"class={float(r['fidelity_class']):.4f} mode={float(r['fidelity_mode']):.4f}")
    print(f"[remetric] → {out}", flush=True)


_TRAIN: dict | None = None
_VAL: dict | None = None
_ARGS = None


def _fit_worker(spec: tuple[str, str]) -> dict:
    fset, mkey = spec
    tag = f"{fset}_{mkey}"
    try:
        assert _TRAIN is not None and _VAL is not None and _ARGS is not None
        idx = np.asarray(FEATURE_SETS[fset], dtype=int)
        names = [ALL_FEATURE_NAMES[i] for i in idx]
        mspec = dict(MODEL_SPECS[mkey])
        family = mspec.pop("family")
        X = _TRAIN["X"][:, idx]
        y = _TRAIN["target"]
        w = _TRAIN["weight"]
        t0 = time.time()
        if family == "cart":
            from sklearn.tree import DecisionTreeRegressor, export_text

            model = DecisionTreeRegressor(random_state=0, **mspec)
            model.fit(X, y, sample_weight=w)
            depth, leaves = int(model.get_depth()), int(model.get_n_leaves())
            nodes = int(model.tree_.node_count)
            importance = np.asarray(model.feature_importances_, dtype=float)
            rules = export_text(
                model, feature_names=names,
                max_depth=min(int(mspec["max_depth"]), 6), decimals=3,
            )
        else:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="regression_l2", num_leaves=int(mspec["num_leaves"]),
                learning_rate=0.04, n_estimators=600, min_child_samples=40,
                subsample=0.85, subsample_freq=1, colsample_bytree=0.90,
                reg_lambda=1.0, random_state=0, n_jobs=int(_ARGS.lgbm_jobs),
                verbosity=-1, deterministic=True, force_col_wise=True,
            )
            model.fit(X, y, sample_weight=w)
            depth, leaves = -1, int(mspec["num_leaves"])
            nodes = int(model.booster_.num_trees())
            gain = np.asarray(model.booster_.feature_importance("gain"), dtype=float)
            importance = gain / max(gain.sum(), 1e-12)
            rules = ""
        fit_sec = time.time() - t0

        package = {
            "schema_version": 1,
            "feature_schema": FEATURE_SCHEMA,
            "tree": model,
            "estimator_kind": "regressor",
            "objective": "prob",
            "info_level": fset,
            "info_label": fset,
            "complexity": mkey,
            "feature_indices": idx.tolist(),
            "feature_names": names,
            "model_family": family,
            "model_spec": mspec,
            "teacher": str(MODEL_DIR),
            "teacher_kind": "pure_PPO",
            "n_train_states": int(len(_TRAIN["ncand"])),
            "n_train_candidate_rows": int(len(_TRAIN["X"])),
            "git_sha": git_sha(),
        }
        metrics = fidelity(package, _VAL)
        package["validation"] = metrics
        out_dir = Path(_ARGS.out_dir)
        path = out_dir / f"{tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(package, f)
        order = np.argsort(-importance)
        imp_rows = [
            {
                "policy": tag, "feature": names[i], "rank": int(r + 1),
                "importance": float(importance[i]),
                "family": AUG_FAMILY.get(names[i], "BASE"),
            }
            for r, i in enumerate(order) if importance[i] > 0
        ]
        if rules:
            (out_dir / f"{tag}_rules.txt").write_text(
                f"{tag} teacher=pure_PPO depth={depth} leaves={leaves}\n"
                f"validation={json.dumps(metrics, ensure_ascii=False)}\n\n{rules}\n",
                encoding="utf-8",
            )
        return {
            "ok": True,
            "row": {
                "policy": tag, "feature_set": fset, "model": mkey, "family": family,
                "n_features": len(idx),
                "n_features_used": int(np.count_nonzero(importance)),
                "n_aug_used": int(sum(
                    1 for i in range(len(idx)) if importance[i] > 0 and names[i] in AUG_NAMES
                )),
                "depth": depth, "leaves": leaves, "nodes": nodes,
                "fit_sec": round(fit_sec, 1),
                **metrics,
                "pkl": str(path), "pkl_sha256": sha256_file(path),
            },
            "importance": imp_rows,
        }
    except Exception as exc:
        import traceback

        return {"ok": False, "tag": tag, "err": (str(exc) + traceback.format_exc())[:2000]}


def fit_main(args) -> None:
    global _TRAIN, _VAL, _ARGS
    _ARGS = args
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    t0 = time.time()
    _TRAIN = load_augmented([str(Path(x).resolve()) for x in args.train_data.split(",") if x])
    _VAL = load_augmented([str(Path(x).resolve()) for x in args.val_data.split(",") if x])
    print(
        f"[v17-fit] train states={len(_TRAIN['ncand'])} rows={len(_TRAIN['X'])} "
        f"val states={len(_VAL['ncand'])} feat={_TRAIN['X'].shape[1]} "
        f"증강 wall={time.time()-t0:.0f}s",
        flush=True,
    )

    specs: list[tuple[str, str]] = []
    for item in args.arms.split(","):
        item = item.strip()
        if not item:
            continue
        fset, mkey = item.rsplit("_", 1)
        if fset not in FEATURE_SETS or mkey not in MODEL_SPECS:
            raise ValueError(f"미지 실험군: {item}")
        specs.append((fset, mkey))
    if not specs:
        raise ValueError("실험군 0개")

    rows, imps = [], []
    workers = min(args.workers, len(specs))
    print(f"[v17-fit] arms={len(specs)} workers={workers}", flush=True)
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_fit_worker, specs), 1):
            if not res["ok"]:
                raise RuntimeError(f"적합 실패 {res['tag']}: {res['err']}")
            r = res["row"]
            rows.append(r)
            imps.extend(res["importance"])
            print(
                f"  [{i}/{len(specs)}] {r['policy']:<18} feat={r['n_features']:>2} "
                f"aug_used={r['n_aug_used']:>2} leaves={r['leaves']:>4} "
                f"fid={r['fidelity_full']:.4f} dest={r['fidelity_dest']:.4f} "
                f"top3={r['teacher_top3_hit']:.4f} pret={r['prob_retention']:.4f} "
                f"({r['fit_sec']:.0f}s)",
                flush=True,
            )

    rows.sort(key=lambda x: -x["fidelity_full"])
    summary = out_dir / "fit_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    imp_csv = out_dir / "feature_importance.csv"
    with open(imp_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["policy", "rank", "feature", "family", "importance"])
        wr.writeheader()
        wr.writerows([{k: r[k] for k in wr.fieldnames} for r in imps])
    meta = {
        "schema_version": 1,
        "teacher": str(MODEL_DIR),
        "teacher_kind": "pure_PPO",
        "teacher_sha256": sha256_file(MODEL_DIR / "final_model.zip"),
        "train_data": {p: sha256_file(p) for p in args.train_data.split(",") if p},
        "val_data": {p: sha256_file(p) for p in args.val_data.split(",") if p},
        "feature_schema": FEATURE_SCHEMA,
        "base_features": FEATURE_NAMES,
        "aug_features": {n: AUG_FAMILY[n] for n in AUG_NAMES},
        "feature_sets": {k: len(v) for k, v in FEATURE_SETS.items()},
        "arms": [f"{a}_{b}" for a, b in specs],
        "eval250_not_opened": True,
        "git_sha": git_sha(),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (out_dir / "fit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[v17-fit] 완료 {out_dir} wall={(time.time()-t0)/60:.1f}분", flush=True)


def logstats_main(args) -> None:
    """교사(순수 PPO) 결정로그의 순위·구간별 선택비율. 트리와 무관한 원 정책 통계."""
    data = load_augmented([str(Path(x).resolve()) for x in args.data.split(",") if x])
    off = data["offsets"]
    n = len(off) - 1
    axes = {
        "rank_eta_all": "ETA 순위(전 후보)",
        "rank_p_sent": "누적발송 순위",
        "rank_occ_ratio": "점유율 순위",
        "rank_arrive": "차량복귀 순위",
        "rank_uav_adv": "UAV 이득 순위",
        "eta_bin": "ETA 구간",
        "p_sent_bin": "누적발송 구간",
        "occ_bin": "점유율 구간",
    }
    rows = []
    for col, label in axes.items():
        j = ALL_FEATURE_NAMES.index(col)
        chosen, avail = {}, {}
        for s in range(n):
            a, b = int(off[s]), int(off[s + 1])
            acts = data["cand_action"][a:b]
            Xs = data["X"][a:b]
            teacher = int(data["teacher_action"][s])
            t_row = int(np.flatnonzero(acts == teacher)[0])
            v = float(Xs[t_row, j])
            chosen[v] = chosen.get(v, 0) + 1
            for u in np.unique(Xs[:, j]):
                avail[float(u)] = avail.get(float(u), 0) + 1
        for v in sorted(avail):
            rows.append({
                "axis": col, "axis_label": label, "value": v,
                "n_chosen": chosen.get(v, 0),
                "n_states_with_value": avail[v],
                "share_of_decisions": chosen.get(v, 0) / n,
                "pick_rate_when_available": chosen.get(v, 0) / avail[v],
            })
    # class·mode 별 선택 분해
    for s in range(n):
        pass
    cm = {}
    for s in range(n):
        teacher = int(data["teacher_action"][s])
        c, d, m = decode_action(teacher, 192, 47)
        key = (c, m, d == 0)
        cm[key] = cm.get(key, 0) + 1
    for (c, m, stay), cnt in sorted(cm.items()):
        rows.append({
            "axis": "action_decomp", "axis_label": "행동 분해",
            "value": float(c * 100 + m * 10 + int(stay)),
            "n_chosen": cnt, "n_states_with_value": n,
            "share_of_decisions": cnt / n, "pick_rate_when_available": cnt / n,
        })
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"[v17-logstats] states={n} rows={len(rows)} → {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--train_data", default=str(TRAIN_NPZ))
    f.add_argument("--val_data", default=str(VAL_NPZ))
    f.add_argument("--arms", required=True)
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--lgbm_jobs", type=int, default=2)
    f.add_argument("--out_dir", required=True)
    r = sub.add_parser("remetric")
    r.add_argument("--fit_dir", required=True)
    r.add_argument("--val_data", default=str(VAL_NPZ))
    g = sub.add_parser("logstats")
    g.add_argument("--data", default=str(TRAIN_NPZ))
    g.add_argument("--out", required=True)
    args = p.parse_args()
    {"fit": fit_main, "remetric": remetric_main, "logstats": logstats_main}[args.command](args)


if __name__ == "__main__":
    main()
