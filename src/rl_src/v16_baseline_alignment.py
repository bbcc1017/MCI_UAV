"""v16 기준선 정합화 전수실험 — LB-T3 세분화 + Shin 병원규칙 정렬.

공통 프로토콜
-------------
* random4 train 1,000좌표 + 대표점 eval 250좌표
* 좌표·정책당 1,000 episodes, simulation seed 0..999
* occ gate, essential+load+valid, H_PAD=47, PDR_woG
* 좌표별 원자 체크포인트로 재개 가능; 기존 v10/v12/Shin16 산출물은 읽기만 한다.

LB 계열
-------
* LB3_AGNOSTIC_RR_FASTEST: 원규칙 없는 R/Y 교대+최단 가용수단
* LB3_CAP_F64: HEUR64 각 규칙에 T3 적용한 64개 원자료
* LB3_HEURBEST_THEN_CAP: HEUR64를 먼저 고른 기존 방식(후처리 집계)
* LB3_CAPBEST64: T3 적용 후 64개 중 좌표별 최적(사후 상한)
* LB3_STARTBEST32: 위 상한을 START 32개로 제한

Shin 계열
---------
* SHIN_ALIGN_HOSP16: Threshold/2Step class 산식과 기존 공통 mode 규칙은 보존하고
  병원 선택만 RedOnly/YellowNearest로 분리(2×2×4=16). 기존 SHIN_LIT16은 불변.

실행 예:
  python src/rl_src/v16_baseline_alignment.py --workers 104
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from v10_full_baselines import (  # noqa: E402
    EVAL_MANIFEST,
    METRIC_NAMES,
    REPO,
    TRAIN_MANIFEST,
    all_rule_names,
    atomic_savez,
    base_row,
    git_sha,
    scenario_bundle_sha256,
    sha256_file,
    validate_inputs,
    write_csv_atomic,
)
from shin_full_baselines import policy_seed, rollout_checked  # noqa: E402


DEFAULT_OUT = REPO / "results/scoreboard/v16/baseline_alignment_full1000"
HEUR_BEST = REPO / "results/scoreboard/v17/heur64_eta_aligned_full1000/heuristic_best_summary.csv"
SIM_SRC = REPO / "src/sim_src"
SOURCE_PATHS = (
    REPO / "src/rl_src/lb3_policy.py",
    REPO / "src/rl_src/loadbalance_heuristic.py",
    REPO / "src/sim_src/ShinAlignedHeuristics.py",
    REPO / "src/sim_src/ShinHeuristics.py",
    REPO / "src/sim_src/RuleManager.py",
    REPO / "src/sim_src/EventManager.py",
    REPO / "src/sim_src/ScenarioManager.py",
    Path(__file__).resolve(),
)


def source_hashes() -> dict[str, str]:
    return {str(p.relative_to(REPO)): sha256_file(p) for p in SOURCE_PATHS}


def source_bundle_sha256(hashes: dict[str, str] | None = None) -> str:
    hashes = hashes or source_hashes()
    h = hashlib.sha256()
    for path, digest in sorted(hashes.items()):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def shin_aligned_specs() -> list[tuple[str, str, str]]:
    specs = [
        (method, hospital, mode)
        for method in ("Threshold", "2Step")
        for hospital in ("RedOnly", "YellowNearest")
        for mode in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB")
    ]
    if len(specs) != 16 or len(set(specs)) != 16:
        raise AssertionError("Shin aligned 16 구성 오류")
    return specs


def shin_aligned_names() -> list[str]:
    return [f"ShinAlignHOS {m}, {h}, Mode {v}" for m, h, v in shin_aligned_specs()]


def lb_policy_names() -> list[str]:
    return ["LB3_AGNOSTIC_RR_FASTEST"] + [f"LB3_CAP_F64 | {r}" for r in all_rule_names()]


def work_path(out_dir: Path, family: str, entry: dict) -> Path:
    return out_dir / "work" / family / entry["dataset"] / f"{entry['key']}.npz"


def checkpoint_valid(path: Path, names: list[str], n_eps: int, seed: int, bundle: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["values"].shape == (len(names), n_eps, len(METRIC_NAMES))
                and data["done"].shape == (len(names),)
                and bool(np.asarray(data["done"]).all())
                and data["policy_names"].tolist() == names
                and np.array_equal(data["seeds"], np.arange(seed, seed + n_eps))
                and str(data["source_bundle_sha256"].item()) == bundle
                and np.isfinite(data["values"]).all()
            )
    except Exception:
        return False


def save_checkpoint(path, values, done, seeds, names, bundle):
    atomic_savez(
        path,
        values=values,
        done=done,
        seeds=seeds,
        policy_names=np.asarray(names),
        source_bundle_sha256=np.asarray(bundle),
    )


def _adapt_rule(rule, episode_policy_seed: int):
    """Rule.select([c,d,m])를 flat hard-mask 정책으로 감싼다."""
    state = {"en_manager": None, "encode": None}

    def policy(ro, mask, env):
        if state["en_manager"] is not env.en_manager:
            from loadbalance_heuristic import _codec_from_mask

            H_layout = len(mask) // 4 - 1 if env.amb_num > 0 and env.uav_num > 0 else len(mask) // 2 - 1
            state["encode"] = _codec_from_mask(len(mask), H_layout)
            rule.set_seed(np.random.default_rng(episode_policy_seed))
            rule.init_with_scenario({"EntityManager": env.en_manager})
            state["en_manager"] = env.en_manager
        obs = env.en_manager.get_full_obs()
        obs["time"] = env.ev_manager.time
        c, d, m = rule.select(obs)
        action = state["encode"](0, 0, 0) if c < 0 else state["encode"](c, d, m)
        if action < len(mask) and mask[action]:
            return int(action)
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else 0

    return policy


def _run_lb(entry: dict, path: Path, n_eps: int, seed: int, checkpoint_every: int, bundle: str):
    names = lb_policy_names()
    seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
    values = np.full((len(names), n_eps, len(METRIC_NAMES)), np.nan, dtype=np.float32)
    done = np.zeros(len(names), dtype=bool)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if (
                old["values"].shape == values.shape
                and old["policy_names"].tolist() == names
                and np.array_equal(old["seeds"], seeds)
                and str(old["source_bundle_sha256"].item()) == bundle
            ):
                values[:] = old["values"]
                done[:] = old["done"]

    from lb3_policy import make_agnostic_lb_policy
    from loadbalance_heuristic import make_cap_policy
    from viper_distill import _suppress_stdout, make_feature_env

    factory = make_feature_env(entry["config"], None)
    base_rules = [None] + all_rule_names()
    completed = 0
    with _suppress_stdout():
        for idx, base_rule in enumerate(base_rules):
            if done[idx] and np.isfinite(values[idx]).all():
                continue
            policy = make_agnostic_lb_policy(T=3) if base_rule is None else make_cap_policy(base_rule, 3)
            for ep, episode_seed in enumerate(seeds):
                values[idx, ep] = rollout_checked(factory, policy, int(episode_seed))
            done[idx] = True
            completed += 1
            if completed >= checkpoint_every:
                save_checkpoint(path, values, done, seeds, names, bundle)
                completed = 0
    save_checkpoint(path, values, done, seeds, names, bundle)
    return float(values[:, :, 3].mean(axis=1).min())


def _run_shin(entry: dict, path: Path, n_eps: int, seed: int, checkpoint_every: int, bundle: str):
    names = shin_aligned_names()
    specs = shin_aligned_specs()
    seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
    values = np.full((len(names), n_eps, len(METRIC_NAMES)), np.nan, dtype=np.float32)
    done = np.zeros(len(names), dtype=bool)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if (
                old["values"].shape == values.shape
                and old["policy_names"].tolist() == names
                and np.array_equal(old["seeds"], seeds)
                and str(old["source_bundle_sha256"].item()) == bundle
            ):
                values[:] = old["values"]
                done[:] = old["done"]

    if str(SIM_SRC) not in sys.path:
        sys.path.insert(0, str(SIM_SRC))
    from ShinAlignedHeuristics import ShinHospitalAlignedRule
    from viper_distill import _suppress_stdout, make_feature_env

    factory = make_feature_env(entry["config"], None)
    completed = 0
    with _suppress_stdout():
        for idx, (method, hospital, mode) in enumerate(specs):
            if done[idx] and np.isfinite(values[idx]).all():
                continue
            for ep, episode_seed in enumerate(seeds):
                rule = ShinHospitalAlignedRule(method, hospital, mode)
                policy = _adapt_rule(rule, policy_seed(int(episode_seed)))
                values[idx, ep] = rollout_checked(factory, policy, int(episode_seed))
            done[idx] = True
            completed += 1
            if completed >= checkpoint_every:
                save_checkpoint(path, values, done, seeds, names, bundle)
                completed = 0
    save_checkpoint(path, values, done, seeds, names, bundle)
    return float(values[:, :, 3].mean(axis=1).min())


def worker(job):
    family, entry, path_str, n_eps, seed, checkpoint_every, bundle = job
    try:
        import torch

        torch.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
            MCI_TIER_MASK="1",
        )
        if source_bundle_sha256() != bundle:
            raise RuntimeError("실행 중 v16 기준선 관련 소스 hash 변경")
        path = Path(path_str)
        metric = (
            _run_lb(entry, path, n_eps, seed, checkpoint_every, bundle)
            if family == "lb3"
            else _run_shin(entry, path, n_eps, seed, checkpoint_every, bundle)
        )
        return {"ok": True, "family": family, "dataset": entry["dataset"], "key": entry["key"], "pdr": metric}
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "family": family,
            "dataset": entry["dataset"],
            "key": entry["key"],
            "err": (str(exc) + traceback.format_exc())[:5000],
        }


def run_jobs(entries, out_dir, n_eps, seed, checkpoint_every, workers, bundle):
    jobs = []
    for entry in entries:
        for family, names in (("lb3", lb_policy_names()), ("shin_align", shin_aligned_names())):
            path = work_path(out_dir, family, entry)
            if not checkpoint_valid(path, names, n_eps, seed, bundle):
                jobs.append((family, entry, str(path), n_eps, seed, checkpoint_every, bundle))
    if not jobs:
        print("[v16] 모든 체크포인트 완료 — 실행 job=0", flush=True)
        return
    # 긴 LB와 짧은 Shin을 섞어 초반부터 두 정책군 모두 진행시키고, 끝에는 work stealing.
    jobs.sort(key=lambda x: (x[1]["dataset"], x[1]["key"], x[0]))
    n_workers = min(workers, len(jobs))
    print(f"[v16] jobs={len(jobs)} workers={n_workers} n_eps={n_eps} seed={seed}..{seed+n_eps-1}", flush=True)
    failed = []
    started = time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        for idx, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            if result["ok"]:
                print(
                    f"  [{idx}/{len(jobs)}] {result['family']} {result['dataset']}:{result['key']} "
                    f"best={result['pdr']:.5f} wall={(time.time()-started)/3600:.2f}h",
                    flush=True,
                )
            else:
                failed.append(result)
                print(f"  [{idx}/{len(jobs)}] FAIL {result['family']}:{result['key']} {result['err'][:500]}", flush=True)
    if failed:
        path = out_dir / "failed_jobs.json"
        path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"v16 실패 {len(failed)}개 — {path}")


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "std": std, "ci95": 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0}


def _metric_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, metric in enumerate(METRIC_NAMES):
        for stat, value in _stats(values[:, idx]).items():
            out[f"{prefix}_{metric}_{stat}"] = value
    return out


def aggregate(entries, out_dir, n_eps, seed, bundle):
    heur = pd.read_csv(HEUR_BEST, encoding="utf-8-sig", dtype={"sigcd": str})
    if len(heur) != 1250 or heur[["dataset", "coordinate_key"]].duplicated().any():
        raise ValueError("v10 HEUR best grain 불일치")
    heur_map = heur.set_index(["dataset", "coordinate_key"])["best_rule"].to_dict()
    rules = all_rule_names()
    rule_idx = {rule: idx + 1 for idx, rule in enumerate(rules)}  # 0=AGN
    start_indices = np.asarray([idx + 1 for idx, rule in enumerate(rules) if rule.startswith("START,")])

    lb_full_rows, lb_selected_rows = [], []
    shin_full_rows, shin_best_rows = [], []
    episode_path = out_dir / "selected_policy_episodes.csv.gz"
    temp = Path(str(episode_path) + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as fh:
        fields = ["dataset", "coordinate_key", "region", "sigcd", "point", "policy", "rule", "episode", "seed", *METRIC_NAMES]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for count, entry in enumerate(entries, 1):
            lb_path = work_path(out_dir, "lb3", entry)
            shin_path = work_path(out_dir, "shin_align", entry)
            if not checkpoint_valid(lb_path, lb_policy_names(), n_eps, seed, bundle):
                raise RuntimeError(f"LB 체크포인트 불완전: {lb_path}")
            if not checkpoint_valid(shin_path, shin_aligned_names(), n_eps, seed, bundle):
                raise RuntimeError(f"Shin 체크포인트 불완전: {shin_path}")
            with np.load(lb_path, allow_pickle=False) as data:
                lb = np.asarray(data["values"], dtype=float)
            with np.load(shin_path, allow_pickle=False) as data:
                sh = np.asarray(data["values"], dtype=float)

            lb_means = lb[1:, :, 3].mean(axis=1)
            cap_best_local = int(np.argmin(lb_means)) + 1
            start_best_local = int(start_indices[np.argmin(lb[start_indices, :, 3].mean(axis=1))])
            before_rule = heur_map[(entry["dataset"], entry["key"])]
            before_local = rule_idx[before_rule]
            selected = [
                ("LB3_AGNOSTIC_RR_FASTEST", "severity-agnostic round-robin", 0),
                ("LB3_HEURBEST_THEN_CAP", before_rule, before_local),
                ("LB3_CAPBEST64", rules[cap_best_local - 1], cap_best_local),
                ("LB3_STARTBEST32", rules[start_best_local - 1], start_best_local),
            ]

            means = lb[1:, :, 3].mean(axis=1)
            order = np.argsort(means, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for idx, rule in enumerate(rules, 1):
                row = base_row(entry, n_eps, seed)
                row.update({"policy": "LB3_CAP_F64", "base_rule_index": idx - 1, "base_rule": rule, "rank_after_cap": int(ranks[idx - 1])})
                row.update(_metric_summary("lb3", lb[idx]))
                lb_full_rows.append(row)
            for policy_name, rule, idx in selected:
                row = base_row(entry, n_eps, seed)
                row.update({"policy": policy_name, "selected_rule": rule, "source_index": idx})
                row.update(_metric_summary("lb3", lb[idx]))
                lb_selected_rows.append(row)
                for ep in range(n_eps):
                    writer.writerow({
                        "dataset": entry["dataset"], "coordinate_key": entry["key"], "region": entry["region"],
                        "sigcd": entry["sigcd"], "point": entry["point"], "policy": policy_name, "rule": rule,
                        "episode": ep, "seed": seed + ep,
                        **{metric: float(lb[idx, ep, mi]) for mi, metric in enumerate(METRIC_NAMES)},
                    })

            sh_means = sh[:, :, 3].mean(axis=1)
            sh_best = int(np.argmin(sh_means))
            order = np.argsort(sh_means, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for idx, name in enumerate(shin_aligned_names()):
                method, hospital, mode = shin_aligned_specs()[idx]
                row = base_row(entry, n_eps, seed)
                row.update({"policy": "SHIN_ALIGN_HOSP16", "rule_index": idx, "rule": name, "method": method, "hospital_rule": hospital, "common_mode_rule": mode, "rank": int(ranks[idx])})
                row.update(_metric_summary("shin_align", sh[idx]))
                shin_full_rows.append(row)
            row = base_row(entry, n_eps, seed)
            row.update({"policy": "SHIN_ALIGN_HOSP_BEST16", "best_rule_index": sh_best, "best_rule": shin_aligned_names()[sh_best]})
            row.update(_metric_summary("shin_align_best", sh[sh_best]))
            shin_best_rows.append(row)
            if count % 50 == 0:
                print(f"[aggregate] {count}/{len(entries)}", flush=True)
    os.replace(temp, episode_path)
    write_csv_atomic(out_dir / "lb3_full64_summary.csv", lb_full_rows)
    write_csv_atomic(out_dir / "lb3_selected_summary.csv", lb_selected_rows)
    write_csv_atomic(out_dir / "shin_aligned_full_summary.csv", shin_full_rows)
    write_csv_atomic(out_dir / "shin_aligned_best_summary.csv", shin_best_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default=str(TRAIN_MANIFEST))
    p.add_argument("--eval_manifest", default=str(EVAL_MANIFEST))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--n_eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=104)
    p.add_argument("--checkpoint_every", type=int, default=4)
    p.add_argument("--phase", choices=["all", "run", "aggregate"], default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no_strict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.n_eps, args.workers, args.checkpoint_every) <= 0:
        raise ValueError("n_eps/workers/checkpoint_every는 양수")
    train_path = Path(args.train_manifest).resolve()
    eval_path = Path(args.eval_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = validate_inputs(train_path, eval_path, args.n_eps, not args.no_strict)
    if args.limit:
        entries = [e for d in ("train1000", "eval250") for e in [x for x in entries if x["dataset"] == d][: args.limit]]
    hashes = source_hashes()
    bundle = source_bundle_sha256(hashes)
    meta_path = out_dir / "protocol_meta.json"
    meta = {
        "protocol": "v17_eta_aligned_lb3_shin_full1000",
        "status": "running",
        "pid": os.getpid(),
        "created_at_unix": time.time(),
        "git_sha": git_sha(),
        "source_hashes": hashes,
        "source_bundle_sha256": bundle,
        "n_coordinates": len(entries),
        "n_train_coordinates": sum(e["dataset"] == "train1000" for e in entries),
        "n_eval_coordinates": sum(e["dataset"] == "eval250" for e in entries),
        "n_episodes_per_policy_per_coordinate": args.n_eps,
        "seed_start": args.seed,
        "seed_end": args.seed + args.n_eps - 1,
        "lb_policy_count": len(lb_policy_names()),
        "shin_aligned_policy_count": len(shin_aligned_names()),
        "total_episode_count": len(entries) * (len(lb_policy_names()) + len(shin_aligned_names())) * args.n_eps,
        "workers": args.workers,
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "eval_manifest": str(eval_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "scenario_bundle_sha256": scenario_bundle_sha256(entries),
        "heur_best_source": str(HEUR_BEST),
        "heur_best_source_sha256": (
            sha256_file(HEUR_BEST) if args.phase in ("all", "aggregate") else None
        ),
        "environment": {"MCI_CAP_GATE": "occ", "MCI_OBS_VARIANT": "essential+load+valid", "MCI_H_PAD": "47", "MCI_TIER_MASK": "1", "MCI_REWARD_MODE": "woG"},
        "policy_names": {"lb": lb_policy_names(), "shin_aligned": shin_aligned_names()},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if args.phase in ("all", "run"):
            run_jobs(entries, out_dir, args.n_eps, args.seed, args.checkpoint_every, args.workers, bundle)
        if args.phase in ("all", "aggregate"):
            aggregate(entries, out_dir, args.n_eps, args.seed, bundle)
        meta["status"] = "complete"
        meta["completed_at_unix"] = time.time()
    except Exception:
        meta["status"] = "failed"
        meta["failed_at_unix"] = time.time()
        raise
    finally:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
