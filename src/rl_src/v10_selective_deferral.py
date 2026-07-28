# -*- coding: utf-8 -*-
"""증류 학생정책의 선택적 PPO/NCRP 위임 평가.

학생 후보점수의 top1-top2 margin이 작은 결정만 교사에게 넘긴다. 임계값은 최종 평가와
분리한 수집 NPZ에서 목표 학생 처리율(coverage)별 분위수로 먼저 동결한다.

* PPO 위임: 불확실한 결정에서 v10 Pointer의 deterministic masked action 사용
* NCRP 위임: 불확실한 결정에서 v10 Pointer+NCRP(K8,h10,m16)만 실행

모든 경로는 같은 hard action mask를 공유한다. 출력에는 실제 학생 처리율, 위임률,
NCRP가 PPO 행동을 바꾼 비율과 의사결정 지연을 함께 기록한다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

from tree_distill_policy import (
    FEATURE_NAMES,
    ActionFeatureBuilder,
    load_tree_package,
    tree_scores,
)
from v10_tree_distill import load_datasets

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"
COLS = [
    "region", "method", "episode", "seed", "pdr_woG", "reward_woG", "sim_time",
    "n_decisions", "n_student", "n_deferred", "n_planner_switched",
    "student_coverage", "defer_rate", "ms_per_decision",
]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _choose(actions: np.ndarray, X: np.ndarray, score: np.ndarray) -> tuple[int, float]:
    order = np.argsort(-np.asarray(score), kind="stable")
    top = float(score[order[0]])
    second = float(score[order[1]]) if len(order) > 1 else -np.inf
    best = np.flatnonzero(np.isclose(score, top, rtol=0.0, atol=1e-12))
    if len(best) > 1:
        stay = X[best, FEATURE_NAMES.index("is_stay")]
        eta = X[best, FEATURE_NAMES.index("eta_rank")]
        tie_order = np.lexsort((actions[best], eta, stay))
        row = int(best[tie_order[0]])
    else:
        row = int(best[0])
    return int(actions[row]), top - second


def calibrate_thresholds(package: dict, data_paths: list[str], coverages: list[float]) -> dict:
    data = load_datasets(data_paths)
    score = tree_scores(package, data["X"])
    margins = np.empty(len(data["ncand"]), dtype=np.float64)
    for i, (s, e) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        values = score[int(s):int(e)]
        if len(values) <= 1:
            margins[i] = np.inf
        else:
            top = np.partition(values, -2)[-2:]
            margins[i] = float(top[-1] - top[-2])
    finite = margins[np.isfinite(margins)]
    if not len(finite):
        raise RuntimeError("임계값 보정용 유한 margin이 없음")
    result = {}
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"coverage 범위 오류: {coverage}")
        threshold = -np.inf if coverage == 1 else float(
            np.quantile(finite, 1.0 - coverage, method="linear")
        )
        actual = float(np.mean(margins >= threshold))
        result[f"{coverage:.2f}"] = {
            "target_student_coverage": coverage,
            "threshold": threshold,
            "calibration_student_coverage": actual,
            "n_calibration_states": int(len(margins)),
        }
    return result


def _method_specs(thresholds: dict, fallbacks: set[str], include_full: bool) -> list[dict]:
    specs = [{"method": "STUDENT", "fallback": "none", "threshold": -np.inf}]
    if include_full:
        specs.append({"method": "PPO", "fallback": "ppo", "threshold": np.inf})
        if "ncrp" in fallbacks:
            specs.append({"method": "NCRP_ALL", "fallback": "ncrp", "threshold": np.inf})
    for key, item in thresholds.items():
        coverage = float(key)
        suffix = f"C{int(round(coverage * 100)):02d}"
        if "ppo" in fallbacks and coverage < 1:
            specs.append({
                "method": f"STUDENT_PPO_{suffix}",
                "fallback": "ppo",
                "threshold": item["threshold"],
            })
        if "ncrp" in fallbacks and coverage < 1 and coverage >= 0.75:
            specs.append({
                "method": f"STUDENT_NCRP_{suffix}",
                "fallback": "ncrp",
                "threshold": item["threshold"],
            })
    return specs


def _rollout(factory, package, model, planner, method: dict, seed: int):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    builder = ActionFeatureBuilder(h_pad=47)
    done = False
    reward = 0.0
    n_dec = n_student = n_deferred = n_switched = 0
    policy_sec = 0.0
    info = {}
    while not done:
        t0 = time.perf_counter()
        mask = np.asarray(env.action_masks(), dtype=bool)
        actions, X = builder.build(env.unwrapped, mask)
        student_action, margin = _choose(actions, X, tree_scores(package, X))
        if margin >= method["threshold"] or method["fallback"] == "none":
            action = student_action
            n_student += 1
        elif method["fallback"] == "ppo":
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            n_deferred += 1
        elif method["fallback"] == "ncrp":
            action = int(planner.act(env, seed, obs=obs))
            n_deferred += 1
            n_switched += int(planner.last_info.get("switched", False))
        else:
            raise ValueError(method["fallback"])
        policy_sec += time.perf_counter() - t0
        n_dec += 1
        obs, _, term, trunc, info = env.step(action)
        reward += float(info.get("r_woG", 0.0))
        done = term or trunc
    preventable = float(env.unwrapped.preventable_woG)
    pdr = 1.0 - reward / preventable if preventable > 0 else 0.0
    return {
        "pdr_woG": pdr,
        "reward_woG": reward,
        "sim_time": float(info.get("time", np.nan)),
        "n_decisions": n_dec,
        "n_student": n_student,
        "n_deferred": n_deferred,
        "n_planner_switched": n_switched,
        "student_coverage": n_student / max(n_dec, 1),
        "defer_rate": n_deferred / max(n_dec, 1),
        "ms_per_decision": policy_sec * 1000.0 / max(n_dec, 1),
    }


def _worker(job):
    region, cfg, package_path, model_dir, methods, n_eps, seed0 = job
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from sb3_contrib import MaskablePPO
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
        from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
        from planner_policy import TruncatedRolloutPlanner
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env

        package = load_tree_package(package_path)
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        norm = load_vecnorm(os.path.join(model_dir, "vecnormalize.pkl"))
        factory = make_feature_env(cfg, norm)
        planners = {
            m["method"]: TruncatedRolloutPlanner(
                model, K=8, h=10, m=16, leaf_fn=None, clairvoyant=False
            )
            for m in methods if m["fallback"] == "ncrp"
        }
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                for method in methods:
                    result = _rollout(
                        factory,
                        package,
                        model,
                        planners.get(method["method"]),
                        method,
                        seed,
                    )
                    rows.append({
                        "region": region,
                        "method": method["method"],
                        "episode": ep,
                        "seed": seed,
                        **result,
                    })
        return {"ok": True, "region": region, "rows": rows}
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "region": region,
            "err": (str(exc) + "\n" + traceback.format_exc())[:3000],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--model_dir", default=str(MODEL_DIR))
    ap.add_argument("--calibration_data", required=True, help="쉼표구분 npz")
    ap.add_argument("--coverages", default="1.0,0.9,0.75,0.5")
    ap.add_argument("--fallbacks", default="ppo,ncrp")
    ap.add_argument("--include_full", action="store_true")
    ap.add_argument("--regions", default="")
    ap.add_argument("--n_eps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    keys = [x for x in args.regions.split(",") if x in manifest] if args.regions else list(manifest)
    package_path = str(Path(args.student).resolve())
    package = load_tree_package(package_path)
    calibration_paths = [
        str(Path(x).resolve()) for x in args.calibration_data.split(",") if x
    ]
    coverages = [float(x) for x in args.coverages.split(",") if x]
    thresholds = calibrate_thresholds(package, calibration_paths, coverages)
    methods = _method_specs(thresholds, set(args.fallbacks.split(",")), args.include_full)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    done_regions = set()
    if out.exists():
        old = list(csv.DictReader(open(out, encoding="utf-8")))
        by = {}
        for row in old:
            by.setdefault(row["region"], set()).add((row["method"], int(row["seed"])))
        expected = {(m["method"], args.seed0 + ep) for m in methods for ep in range(args.n_eps)}
        done_regions = {region for region, got in by.items() if got == expected}
        incomplete = set(by) - done_regions
        if incomplete:
            raise RuntimeError(f"부분 기록 지역 존재: {sorted(incomplete)[:3]}")

    jobs = [
        (
            key,
            manifest[key],
            package_path,
            str(Path(args.model_dir).resolve()),
            methods,
            args.n_eps,
            args.seed0,
        )
        for key in keys if key not in done_regions
    ]
    print(
        f"[selective] regions={len(keys)} remaining={len(jobs)} methods={len(methods)} "
        f"n_eps={args.n_eps} workers={min(args.workers,max(len(jobs),1))}",
        flush=True,
    )
    for key, item in thresholds.items():
        print(
            f"  coverage {key}: threshold={item['threshold']:.8g} "
            f"cal_actual={item['calibration_student_coverage']:.3f}",
            flush=True,
        )

    new_file = not out.exists()
    fout = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        writer.writeheader()
    t0 = time.time()
    if jobs:
        with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, jobs), 1):
                if not result["ok"]:
                    fout.close()
                    raise RuntimeError(f"{result['region']} 실패: {result['err']}")
                writer.writerows(result["rows"])
                fout.flush()
                print(
                    f"  [{i}/{len(jobs)}] {result['region']} "
                    f"rows={len(result['rows'])} wall={(time.time()-t0)/60:.1f}분",
                    flush=True,
                )
    fout.close()

    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    seen = set()
    for row in rows:
        key = (row["region"], row["method"], int(row["seed"]))
        if key in seen:
            raise RuntimeError(f"평가 중복: {key}")
        seen.add(key)
        pdr = float(row["pdr_woG"])
        if not np.isfinite(pdr) or not 0 <= pdr <= 1:
            raise RuntimeError(f"PDR 오류: {key}={pdr}")
    expected_n = len(keys) * len(methods) * args.n_eps
    if len(rows) != expected_n:
        raise RuntimeError(f"평가 행수 {len(rows)} != {expected_n}")
    meta = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "student": package_path,
        "student_sha256": _sha256(package_path),
        "student_policy": f"{package['info_level']}_{package['complexity']}",
        "model_dir": str(Path(args.model_dir).resolve()),
        "calibration_data": {p: _sha256(p) for p in calibration_paths},
        "thresholds": thresholds,
        "methods": methods,
        "n_regions": len(keys),
        "n_eps": args.n_eps,
        "seed_start": args.seed0,
        "seed_end": args.seed0 + args.n_eps - 1,
        "ncrp": {"K": 8, "h": 10, "m": 16, "clairvoyant": False},
        "output": str(out),
        "output_sha256": _sha256(out),
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[selective] 완료 rows={len(rows)} wall={(time.time()-t0)/60:.1f}분 → {out}", flush=True)


if __name__ == "__main__":
    main()
