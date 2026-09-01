# -*- coding: utf-8 -*-
"""v10 PPO → 지역불변 후보랭킹 의사결정나무 증류.

두 단계 CLI:

1. ``collect``: train1000 시나리오에서 PPO(또는 1차 트리) 롤아웃 상태를 방문하고,
   각 유효 [class,dest,mode] 후보의 현장 특징과 PPO masked 확률을 저장한다.
2. ``fit``: 정보수준 4단계 × 복잡도 4단계의 CART 랭킹 트리를 같은 데이터에 적합한다.

최종 대표점250은 이 스크립트에서 읽지 않는다. 학습·DAgger·검증 데이터는 모두
``sigungu_osrm_train1000_random4_manifest.json``에서만 생성해야 한다.
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

from tree_distill_policy import (
    COMPLEXITY_SPECS,
    FEATURE_NAMES,
    INFO_LABELS,
    INFO_LEVELS,
    ActionFeatureBuilder,
    decode_action,
    load_tree_package,
    tree_scores,
)

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _coord(path: str) -> str:
    return Path(path).resolve().parent.name


def validate_train_manifest(path: Path) -> dict[str, str]:
    manifest = json.load(open(path, encoding="utf-8"))
    eval_manifest = json.load(open(EVAL_MANIFEST, encoding="utf-8"))
    if len(manifest) != 1000:
        raise ValueError(f"증류 manifest는 train1000이어야 함: N={len(manifest)}")
    groups: dict[str, set[str]] = {}
    for key in manifest:
        toks = key.rsplit("_", 2)
        if len(toks) != 3 or not toks[-2].isdigit() or toks[-1] not in {"p0", "p1", "p2", "p3"}:
            raise ValueError(f"train1000 키 형식 오류: {key}")
        groups.setdefault(toks[-2], set()).add(toks[-1])
    bad = {k: sorted(v) for k, v in groups.items() if v != {"p0", "p1", "p2", "p3"}}
    if len(groups) != 250 or bad:
        raise ValueError(f"train1000 그룹 오류: groups={len(groups)} bad={list(bad.items())[:3]}")
    overlap = {_coord(x) for x in manifest.values()} & {_coord(x) for x in eval_manifest.values()}
    if overlap:
        raise ValueError(f"증류 학습/최종평가 좌표 중복 {len(overlap)}개")
    return manifest


def _choose_from_scores(actions, X, scores):
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0.0, atol=1e-12))
    if len(best) == 1:
        return int(actions[int(best[0])])
    stay = X[best, FEATURE_NAMES.index("is_stay")]
    eta = X[best, FEATURE_NAMES.index("eta_rank")]
    order = np.lexsort((actions[best], eta, stay))
    return int(actions[int(best[order[0]])])


def collect_worker(job):
    entries, model_dir, n_eps, seed, behavior_path = job
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT=os.environ.get("MCI_COLLECT_OBS_VARIANT", "essential+load+valid"),
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from sb3_contrib import MaskablePPO
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env

        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        norm = load_vecnorm(os.path.join(model_dir, "vecnormalize.pkl"))
        behavior = load_tree_package(behavior_path) if behavior_path else None

        xs, targets, weights, chosens, cand_actions = [], [], [], [], []
        ncand, teacher_actions, behavior_actions = [], [], []
        state_keys, state_seeds = [], []
        n_mismatch = 0

        def raw_logits(obs):
            ot = th.as_tensor(np.asarray(obs, np.float32), device=model.device).unsqueeze(0)
            with th.no_grad():
                dist = model.policy.get_distribution(ot)
                return dist.distribution.logits.squeeze(0).cpu().numpy()

        with _suppress_stdout():
            for key, cfg in entries:
                fac = make_feature_env(cfg, norm)
                env = fac(seed=seed)
                builder = ActionFeatureBuilder(h_pad=47)
                for ep in range(n_eps):
                    episode_seed = seed + ep
                    obs, _ = env.reset(seed=episode_seed)
                    done = False
                    while not done:
                        mask = np.asarray(env.action_masks(), dtype=bool)
                        actions, X = builder.build(env.unwrapped, mask)
                        logits = raw_logits(obs)[actions]
                        logits = logits - float(np.max(logits))
                        prob = np.exp(logits)
                        prob /= float(np.sum(prob))
                        teacher = int(actions[int(np.argmax(logits))])
                        pred, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        if int(pred) != teacher:
                            n_mismatch += 1

                        ordered = np.sort(prob)
                        gap = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0
                        row_weight = ((0.05 + gap) / len(actions)) * (1.0 + 4.0 * prob)
                        chosen = actions == teacher

                        if behavior is None:
                            action = teacher
                        else:
                            action = _choose_from_scores(actions, X, tree_scores(behavior, X))

                        xs.append(X.astype(np.float32))
                        targets.append(prob.astype(np.float32))
                        weights.append(row_weight.astype(np.float32))
                        chosens.append(chosen)
                        cand_actions.append(actions.astype(np.int16))
                        ncand.append(len(actions))
                        teacher_actions.append(teacher)
                        behavior_actions.append(action)
                        state_keys.append(key)
                        state_seeds.append(episode_seed)

                        obs, _, term, trunc, _ = env.step(action)
                        done = term or trunc

        n_rows = int(sum(ncand))
        return {
            "ok": True,
            "X": np.vstack(xs) if xs else np.zeros((0, len(FEATURE_NAMES)), np.float32),
            "target": np.concatenate(targets) if targets else np.zeros(0, np.float32),
            "weight": np.concatenate(weights) if weights else np.zeros(0, np.float32),
            "chosen": np.concatenate(chosens) if chosens else np.zeros(0, bool),
            "cand_action": np.concatenate(cand_actions) if cand_actions else np.zeros(0, np.int16),
            "ncand": np.asarray(ncand, np.int16),
            "teacher_action": np.asarray(teacher_actions, np.int16),
            "behavior_action": np.asarray(behavior_actions, np.int16),
            "state_key": np.asarray(state_keys),
            "state_seed": np.asarray(state_seeds, np.int32),
            "n_mismatch": n_mismatch,
            "n_rows": n_rows,
        }
    except Exception as exc:
        import traceback

        return {"ok": False, "err": (str(exc) + traceback.format_exc())[:1500]}


def collect_main(args) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = validate_train_manifest(manifest_path)
    items = list(manifest.items())
    if args.key_filter:
        items = [x for x in items if args.key_filter in x[0]]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise ValueError("수집 대상 0개")
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    jobs = [
        (chunk, str(Path(args.model_dir).resolve()), args.n_eps, args.seed, args.behavior_tree)
        for chunk in chunks
    ]
    print(
        f"[collect] points={len(items)} chunks={len(chunks)} n_eps={args.n_eps} "
        f"seed={args.seed} behavior={args.behavior_tree or 'PPO'} workers={min(args.workers,len(jobs))}",
        flush=True,
    )

    packs, t0 = [], time.time()
    with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(collect_worker, jobs), 1):
            if not result["ok"]:
                raise RuntimeError(f"수집 실패 [{i}/{len(jobs)}]: {result['err']}")
            packs.append(result)
            print(
                f"  [{i}/{len(jobs)}] states={len(result['ncand'])} "
                f"rows={result['n_rows']} mismatch={result['n_mismatch']} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )

    X = np.vstack([p["X"] for p in packs])
    target = np.concatenate([p["target"] for p in packs])
    weight = np.concatenate([p["weight"] for p in packs])
    chosen = np.concatenate([p["chosen"] for p in packs])
    cand_action = np.concatenate([p["cand_action"] for p in packs])
    ncand = np.concatenate([p["ncand"] for p in packs])
    teacher_action = np.concatenate([p["teacher_action"] for p in packs])
    behavior_action = np.concatenate([p["behavior_action"] for p in packs])
    state_key = np.concatenate([p["state_key"] for p in packs])
    state_seed = np.concatenate([p["state_seed"] for p in packs])
    offsets = np.concatenate([[0], np.cumsum(ncand, dtype=np.int64)])
    if offsets[-1] != len(X):
        raise RuntimeError(f"offset 불일치 {offsets[-1]} != {len(X)}")
    if not np.isfinite(X).all() or not np.isfinite(target).all() or not np.isfinite(weight).all():
        raise RuntimeError("수집 데이터에 비유한 값")
    mismatch = int(sum(p["n_mismatch"] for p in packs))
    if mismatch:
        raise RuntimeError(f"masked argmax와 model.predict 불일치 {mismatch}건")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X, target=target, weight=weight, chosen=chosen,
        cand_action=cand_action, ncand=ncand, offsets=offsets,
        teacher_action=teacher_action, behavior_action=behavior_action,
        state_key=state_key, state_seed=state_seed,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    meta = {
        "schema_version": 1,
        "role": args.role,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "excluded_final_eval_manifest": str(EVAL_MANIFEST),
        "coordinate_overlap": 0,
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_sha256": sha256_file(Path(args.model_dir) / "final_model.zip"),
        "behavior_tree": args.behavior_tree or "PPO",
        "behavior_tree_sha256": (
            sha256_file(args.behavior_tree) if args.behavior_tree else None
        ),
        "seed_start": args.seed,
        "n_eps_per_point": args.n_eps,
        "n_points": len(items),
        "n_states": len(ncand),
        "n_candidate_rows": len(X),
        "n_features": X.shape[1],
        "teacher_predict_mismatch": mismatch,
        "git_sha": git_sha(),
        "output": str(out),
        "output_sha256": sha256_file(out),
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[collect] 저장 {out} states={len(ncand)} rows={len(X)} "
        f"size={out.stat().st_size/2**20:.1f}MiB wall={(time.time()-t0)/60:.1f}분",
        flush=True,
    )


def load_datasets(paths: list[str]) -> dict:
    arrays = []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        if list(z["feature_names"]) != FEATURE_NAMES:
            raise ValueError(f"특징 스키마 불일치: {path}")
        arrays.append({k: z[k] for k in (
            "X", "target", "weight", "chosen", "cand_action", "ncand",
            "teacher_action", "behavior_action", "state_key", "state_seed",
        )})
    data = {
        k: np.concatenate([a[k] for a in arrays])
        for k in arrays[0]
    }
    data["offsets"] = np.concatenate([[0], np.cumsum(data["ncand"], dtype=np.int64)])
    return data


def _select_indices(actions, X, scores):
    """state별 후보행 index와 선택 action."""
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0.0, atol=1e-12))
    if len(best) > 1:
        stay = X[best, FEATURE_NAMES.index("is_stay")]
        eta = X[best, FEATURE_NAMES.index("eta_rank")]
        order = np.lexsort((actions[best], eta, stay))
        row = int(best[order[0]])
    else:
        row = int(best[0])
    return row, int(actions[row])


def rank_metrics(package: dict, data: dict, max_states: int = 0) -> dict:
    scores = tree_scores(package, data["X"])
    offsets = data["offsets"]
    n_states = len(offsets) - 1
    ids = np.arange(n_states)
    if max_states and n_states > max_states:
        ids = np.linspace(0, n_states - 1, max_states, dtype=int)
    full = cls_ok = dest_ok = mode_ok = 0
    for state in ids:
        s, e = int(offsets[state]), int(offsets[state + 1])
        _, action = _select_indices(
            data["cand_action"][s:e], data["X"][s:e], scores[s:e]
        )
        teacher = int(data["teacher_action"][state])
        if action == teacher:
            full += 1
        ca = decode_action(action, 192, 47)
        ct = decode_action(teacher, 192, 47)
        cls_ok += ca[0] == ct[0]
        dest_ok += ca[1] == ct[1]
        mode_ok += ca[2] == ct[2]
    n = len(ids)
    return {
        "n_states": n,
        "fidelity_full": full / n,
        "fidelity_class": cls_ok / n,
        "fidelity_dest": dest_ok / n,
        "fidelity_mode": mode_ok / n,
    }


def fit_main(args) -> None:
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        export_text,
    )

    train_paths = [str(Path(x).resolve()) for x in args.train_data.split(",") if x]
    val_paths = [str(Path(x).resolve()) for x in args.val_data.split(",") if x]
    train = load_datasets(train_paths)
    val = load_datasets(val_paths) if val_paths else train
    levels = [x for x in args.info_levels.split(",") if x]
    complexities = [x for x in args.complexities.split(",") if x]
    unknown_l = set(levels) - set(INFO_LEVELS)
    unknown_c = set(complexities) - set(COMPLEXITY_SPECS)
    if unknown_l or unknown_c:
        raise ValueError(f"미지 실험군 info={sorted(unknown_l)} complexity={sorted(unknown_c)}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    print(
        f"[fit] train states={len(train['ncand'])} rows={len(train['X'])} "
        f"val states={len(val['ncand'])} objective={args.objective} "
        f"cases={len(levels)*len(complexities)}",
        flush=True,
    )
    for level in levels:
        feat_idx = np.asarray(INFO_LEVELS[level], dtype=int)
        X = train["X"][:, feat_idx]
        for complexity in complexities:
            spec = dict(COMPLEXITY_SPECS[complexity])
            tag = f"{level}_{complexity}"
            if args.objective == "prob":
                tree = DecisionTreeRegressor(random_state=0, **spec)
                tree.fit(X, train["target"], sample_weight=train["weight"])
                kind = "regressor"
            else:
                # state마다 양성 1개이므로 양성행에 후보수를 곱해 클래스 불균형을 보정한다.
                row_n = np.repeat(train["ncand"], train["ncand"].astype(int))
                sw = train["weight"].copy()
                sw[train["chosen"]] *= row_n[train["chosen"]]
                tree = DecisionTreeClassifier(random_state=0, **spec)
                tree.fit(X, train["chosen"].astype(np.int8), sample_weight=sw)
                kind = "classifier"
            package = {
                "schema_version": 1,
                "tree": tree,
                "estimator_kind": kind,
                "objective": args.objective,
                "info_level": level,
                "info_label": INFO_LABELS[level],
                "feature_indices": feat_idx.tolist(),
                "feature_names": [FEATURE_NAMES[i] for i in feat_idx],
                "complexity": complexity,
                "complexity_spec": spec,
                "actual_depth": int(tree.get_depth()),
                "actual_leaves": int(tree.get_n_leaves()),
                "n_train_states": int(len(train["ncand"])),
                "n_train_candidate_rows": int(len(train["X"])),
                "train_data": train_paths,
                "val_data": val_paths,
                "git_sha": git_sha(),
            }
            metrics = rank_metrics(package, val, max_states=args.max_val_states)
            package["validation"] = metrics
            path = out_dir / f"{tag}.pkl"
            with open(path, "wb") as f:
                pickle.dump(package, f)
            rules = export_text(
                tree,
                feature_names=package["feature_names"],
                max_depth=min(spec["max_depth"], args.rule_print_depth),
                decimals=3,
            )
            (out_dir / f"{tag}_rules.txt").write_text(
                f"{tag} ({INFO_LABELS[level]})\n"
                f"objective={args.objective} depth={tree.get_depth()} leaves={tree.get_n_leaves()}\n"
                f"validation={json.dumps(metrics, ensure_ascii=False)}\n\n{rules}\n",
                encoding="utf-8",
            )
            rows.append({
                "policy": tag,
                "info_level": level,
                "info_label": INFO_LABELS[level],
                "complexity": complexity,
                "objective": args.objective,
                "n_features_available": len(feat_idx),
                "n_features_used": int(np.count_nonzero(tree.feature_importances_)),
                "depth": int(tree.get_depth()),
                "leaves": int(tree.get_n_leaves()),
                "nodes": int(tree.tree_.node_count),
                **metrics,
                "pkl": str(path),
                "pkl_sha256": sha256_file(path),
            })
            print(
                f"  {tag}: depth={tree.get_depth()} leaves={tree.get_n_leaves()} "
                f"fid={metrics['fidelity_full']:.3f} "
                f"(class={metrics['fidelity_class']:.3f},dest={metrics['fidelity_dest']:.3f},"
                f"mode={metrics['fidelity_mode']:.3f})",
                flush=True,
            )

    summary = out_dir / "fit_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "schema_version": 1,
        "objective": args.objective,
        "train_data": {
            p: {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            for p in train_paths
        },
        "val_data": {
            p: {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            for p in val_paths
        },
        "info_levels": {k: [FEATURE_NAMES[i] for i in INFO_LEVELS[k]] for k in levels},
        "complexity_specs": {k: COMPLEXITY_SPECS[k] for k in complexities},
        "n_cases": len(rows),
        "git_sha": git_sha(),
        "summary": str(summary),
    }
    (out_dir / "fit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[fit] 저장 {out_dir} wall={(time.time()-t0)/60:.1f}분", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--manifest", default=str(TRAIN_MANIFEST))
    c.add_argument("--model_dir", default=str(MODEL_DIR))
    c.add_argument("--n_eps", type=int, default=1)
    c.add_argument("--seed", type=int, default=5000)
    c.add_argument("--workers", type=int, default=64)
    c.add_argument("--chunk", type=int, default=10)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--key_filter", default="")
    c.add_argument("--behavior_tree", default="")
    c.add_argument("--role", choices=["train", "dagger", "validation"], default="train")
    c.add_argument("--out", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--train_data", required=True, help="쉼표구분 npz")
    f.add_argument("--val_data", default="", help="쉼표구분 npz; 비우면 train 재사용")
    f.add_argument("--objective", choices=["prob", "chosen"], default="prob")
    f.add_argument("--info_levels", default=",".join(INFO_LEVELS))
    f.add_argument("--complexities", default=",".join(COMPLEXITY_SPECS))
    f.add_argument("--max_val_states", type=int, default=100000)
    f.add_argument("--rule_print_depth", type=int, default=6)
    f.add_argument("--out_dir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        collect_main(args)
    else:
        fit_main(args)


if __name__ == "__main__":
    main()
