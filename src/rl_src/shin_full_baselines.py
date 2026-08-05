"""Shin–Lee adapted 휴리스틱 16개를 전국 1,250좌표에서 전수 평가한다.

대상과 공정성:
  * random4 학습분포 1,000좌표 + 대표점 평가분포 250좌표
  * 4개 규칙(Threshold/2Step/PIH/Integrated) × 4개 수단운용 = 16개
  * 모든 정책·좌표가 동일한 시뮬레이션 seed 0..999 사용
  * 정책 내부 RNG는 같은 episode seed에서 별도 namespace로 파생해 동역학 RNG와 분리
  * MCI_CAP_GATE=occ 및 현행 hard action mask를 기존 v10 기준선과 동일하게 적용

산출:
  results/scoreboard/v10/shin16_full1000/
    shin_full_summary.csv       1,250×16 정책 요약
    shin_best_summary.csv       좌표별 Best-of-16
    shin_best_episodes.csv.gz   좌표별 Best 정책의 1,000 episode
    work/shin/...npz            전 정책·episode 원자료 및 재개 체크포인트
    protocol_meta.json          seed·hash·환경·상태

대규모 실행:
  python src/rl_src/shin_full_baselines.py --workers 96
"""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
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

sys.path.insert(0, os.path.dirname(__file__))

from v10_full_baselines import (  # noqa: E402
    METRIC_NAMES,
    REPO,
    EVAL_MANIFEST,
    TRAIN_MANIFEST,
    atomic_savez,
    base_row,
    git_sha,
    scenario_bundle_sha256,
    sha256_file,
    validate_inputs,
    write_csv_atomic,
)


DEFAULT_OUT = REPO / "results/scoreboard/v10/shin16_full1000"
SIM_SRC = REPO / "src/sim_src"
SOURCE_PATHS = (
    REPO / "src/sim_src/ShinHeuristics.py",
    REPO / "src/sim_src/RuleManager.py",
    REPO / "src/rl_src/distill_policy.py",
    Path(__file__).resolve(),
)
POLICY_SEED_NAMESPACE = 0x5348494E  # ASCII "SHIN"


def shin_rule_names() -> list[str]:
    if str(SIM_SRC) not in sys.path:
        sys.path.insert(0, str(SIM_SRC))
    from ShinHeuristics import SHIN_METHODS, SHIN_MODE_RULES

    rules = [
        f"Shin {method}, Mode {mode}"
        for method in SHIN_METHODS
        for mode in SHIN_MODE_RULES
    ]
    if len(rules) != 16 or len(set(rules)) != 16:
        raise AssertionError(f"Shin 규칙 수·중복 오류: {len(rules)}")
    return rules


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in SOURCE_PATHS
    }


def source_bundle_sha256(hashes: dict[str, str] | None = None) -> str:
    hashes = hashes or source_hashes()
    h = hashlib.sha256()
    for path, digest in sorted(hashes.items()):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def policy_seed(episode_seed: int) -> int:
    """동역학 seed와 겹치지 않는 재현가능 정책 RNG seed를 만든다."""
    seq = np.random.SeedSequence([int(episode_seed), POLICY_SEED_NAMESPACE])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def rollout_checked(factory, policy, episode_seed: int):
    """한 episode를 실행하며 mask·truncation·지표 유한성을 강제한다."""
    env = factory(seed=episode_seed)
    obs, _ = env.reset(seed=episode_seed)
    reward = 0.0
    reward_wog = 0.0
    last_time = 0.0
    while True:
        mask = np.asarray(env.action_masks(), dtype=bool)
        action = int(policy(obs, mask, env.unwrapped))
        if action < 0 or action >= len(mask) or not mask[action]:
            raise RuntimeError(
                f"mask 위반 seed={episode_seed} action={action}/{len(mask)}"
            )
        obs, step_reward, terminated, truncated, info = env.step(action)
        reward += float(step_reward)
        reward_wog += float(info.get("r_woG", 0.0))
        last_time = float(info.get("time", 0.0))
        if truncated:
            raise RuntimeError(f"episode truncation seed={episode_seed}")
        if terminated:
            break

    preventable = float(env.unwrapped.preventable)
    preventable_wog = float(env.unwrapped.preventable_woG)
    pdr = 1.0 - reward / preventable if preventable > 0 else 0.0
    pdr_wog = (
        1.0 - reward_wog / preventable_wog
        if preventable_wog > 0
        else 0.0
    )
    result = (reward, pdr, reward_wog, pdr_wog, last_time)
    if not np.isfinite(result).all():
        raise RuntimeError(f"비유한 episode 지표 seed={episode_seed}: {result}")
    return result


def work_path(out_dir: Path, entry: dict) -> Path:
    return out_dir / "work/shin" / entry["dataset"] / f"{entry['key']}.npz"


def valid_checkpoint(
    path: Path,
    n_eps: int,
    seed: int,
    rules: list[str],
    source_bundle: str,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["values"].shape == (len(rules), n_eps, len(METRIC_NAMES))
                and data["done"].shape == (len(rules),)
                and bool(np.asarray(data["done"]).all())
                and np.array_equal(data["seeds"], np.arange(seed, seed + n_eps))
                and data["rule_names"].tolist() == rules
                and str(data["source_bundle_sha256"].item()) == source_bundle
                and np.isfinite(data["values"]).all()
            )
    except Exception:
        return False


def save_checkpoint(
    path: Path,
    values: np.ndarray,
    done: np.ndarray,
    seeds: np.ndarray,
    rules: list[str],
    source_bundle: str,
) -> None:
    atomic_savez(
        path,
        values=values,
        done=done,
        seeds=seeds,
        rule_names=np.asarray(rules),
        source_bundle_sha256=np.asarray(source_bundle),
        policy_seed_namespace=np.asarray(POLICY_SEED_NAMESPACE, dtype=np.uint64),
    )


def shin_worker(job):
    entry, path_str, n_eps, seed, checkpoint_every, expected_source_bundle = job
    path = Path(path_str)
    rules = shin_rule_names()
    try:
        import torch

        torch.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        if source_bundle_sha256() != expected_source_bundle:
            raise RuntimeError("실행 중 Shin 관련 소스 hash가 변경됨")

        from distill_policy import make_heuristic_policy
        from viper_distill import _suppress_stdout, make_feature_env

        seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
        values = np.full(
            (len(rules), n_eps, len(METRIC_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        done = np.zeros(len(rules), dtype=bool)
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if (
                    old["values"].shape == values.shape
                    and np.array_equal(old["seeds"], seeds)
                    and old["rule_names"].tolist() == rules
                    and str(old["source_bundle_sha256"].item())
                    == expected_source_bundle
                ):
                    values[:] = old["values"]
                    done[:] = old["done"]

        factory = make_feature_env(entry["config"], None)
        completed_since_save = 0
        with _suppress_stdout():
            for rule_idx, rule_name in enumerate(rules):
                if done[rule_idx] and np.isfinite(values[rule_idx]).all():
                    continue
                for ep, episode_seed in enumerate(seeds):
                    policy = make_heuristic_policy(
                        rule_name,
                        policy_seed=policy_seed(int(episode_seed)),
                    )
                    values[rule_idx, ep] = rollout_checked(
                        factory,
                        policy,
                        int(episode_seed),
                    )
                done[rule_idx] = True
                completed_since_save += 1
                if completed_since_save >= checkpoint_every:
                    save_checkpoint(
                        path,
                        values,
                        done,
                        seeds,
                        rules,
                        expected_source_bundle,
                    )
                    completed_since_save = 0
        save_checkpoint(
            path,
            values,
            done,
            seeds,
            rules,
            expected_source_bundle,
        )
        means = values[:, :, 3].mean(axis=1)
        best_idx = int(np.argmin(means))
        return {
            "ok": True,
            "dataset": entry["dataset"],
            "key": entry["key"],
            "best_rule": rules[best_idx],
            "best_pdr_wog": float(means[best_idx]),
        }
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "dataset": entry["dataset"],
            "key": entry["key"],
            "err": (str(exc) + traceback.format_exc())[:4000],
        }


def run_pool(jobs: list, workers: int, fail_path: Path) -> None:
    if not jobs:
        print("[shin] 완료 체크포인트 재사용 — 실행 job=0", flush=True)
        return
    n_workers = min(workers, len(jobs))
    print(f"[shin] jobs={len(jobs)} workers={n_workers}", flush=True)
    failed = []
    t0 = time.time()
    with Pool(n_workers, maxtasksperchild=1) as pool:
        for idx, result in enumerate(pool.imap_unordered(shin_worker, jobs), 1):
            if result["ok"]:
                print(
                    f"  [{idx}/{len(jobs)}] {result['dataset']}:{result['key']} "
                    f"Best16={result['best_pdr_wog']:.5f} "
                    f"({(time.time() - t0) / 60:.1f}분)",
                    flush=True,
                )
            else:
                failed.append(result)
                print(
                    f"  [{idx}/{len(jobs)}] FAIL "
                    f"{result['dataset']}:{result['key']} "
                    f"{result['err'][:500]}",
                    flush=True,
                )
    if failed:
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Shin 평가 실패 {len(failed)}개 — {fail_path}")
    print(f"[shin] 완료 wall={(time.time() - t0) / 3600:.2f}시간", flush=True)


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}


def metric_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    out = {}
    for metric_idx, metric in enumerate(METRIC_NAMES):
        summary = stats(values[:, metric_idx])
        out[f"{prefix}_{metric}_mean"] = summary["mean"]
        out[f"{prefix}_{metric}_std"] = summary["std"]
        out[f"{prefix}_{metric}_ci95"] = summary["ci95"]
    return out


def aggregate(
    entries: list[dict],
    out_dir: Path,
    n_eps: int,
    seed: int,
    rules: list[str],
    source_bundle: str,
) -> None:
    full_rows = []
    best_rows = []
    episode_path = out_dir / "shin_best_episodes.csv.gz"
    episode_tmp = Path(str(episode_path) + ".tmp")
    episode_tmp.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_tmp, "wt", newline="", encoding="utf-8") as f:
        fields = [
            "dataset",
            "coordinate_key",
            "region",
            "sigcd",
            "point",
            "lat",
            "lon",
            "policy",
            "rule",
            "episode",
            "seed",
            "reward",
            "pdr",
            "reward_woG",
            "pdr_woG",
            "time",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for entry_idx, entry in enumerate(entries, 1):
            path = work_path(out_dir, entry)
            if not valid_checkpoint(path, n_eps, seed, rules, source_bundle):
                raise RuntimeError(f"Shin 산출 불완전: {path}")
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["values"], dtype=np.float64)
            means = values[:, :, 3].mean(axis=1)
            best_idx = int(np.argmin(means))
            best_rule = rules[best_idx]
            best_values = values[best_idx]
            order = np.argsort(means, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(1, len(order) + 1)

            for rule_idx, rule_name in enumerate(rules):
                row = base_row(entry, n_eps, seed)
                row.update(
                    {
                        "rule_index": rule_idx,
                        "rule": rule_name,
                        "rank_by_PDR_woG": int(rank[rule_idx]),
                    }
                )
                row.update(metric_summary("shin", values[rule_idx]))
                full_rows.append(row)

            best_row = base_row(entry, n_eps, seed)
            best_row.update(
                {
                    "best_rule_index": best_idx,
                    "best_rule": best_rule,
                }
            )
            best_row.update(metric_summary("shin_best", best_values))
            best_rows.append(best_row)

            seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
            for ep, episode_seed in enumerate(seeds):
                writer.writerow(
                    {
                        "dataset": entry["dataset"],
                        "coordinate_key": entry["key"],
                        "region": entry["region"],
                        "sigcd": entry["sigcd"],
                        "point": entry["point"],
                        "lat": entry["lat"],
                        "lon": entry["lon"],
                        "policy": "SHIN_ADAPTED_BEST_OF_16",
                        "rule": best_rule,
                        "episode": ep,
                        "seed": int(episode_seed),
                        "reward": float(best_values[ep, 0]),
                        "pdr": float(best_values[ep, 1]),
                        "reward_woG": float(best_values[ep, 2]),
                        "pdr_woG": float(best_values[ep, 3]),
                        "time": float(best_values[ep, 4]),
                    }
                )
            if entry_idx % 50 == 0:
                print(f"[aggregate] {entry_idx}/{len(entries)}", flush=True)

    os.replace(episode_tmp, episode_path)
    write_csv_atomic(out_dir / "shin_full_summary.csv", full_rows)
    write_csv_atomic(out_dir / "shin_best_summary.csv", best_rows)
    print(
        f"[aggregate] full={len(full_rows)} best={len(best_rows)} "
        f"best_episodes={len(entries) * n_eps}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", default=str(TRAIN_MANIFEST))
    parser.add_argument("--eval_manifest", default=str(EVAL_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--n_eps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--checkpoint_every", type=int, default=4)
    parser.add_argument(
        "--phase",
        choices=["all", "shin", "aggregate"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_eps <= 0 or args.workers <= 0 or args.checkpoint_every <= 0:
        raise ValueError("n_eps/workers/checkpoint_every는 양수여야 함")
    train_path = Path(args.train_manifest).resolve()
    eval_path = Path(args.eval_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = validate_inputs(
        train_path,
        eval_path,
        args.n_eps,
        not args.no_strict,
    )
    if args.limit:
        selected = []
        for dataset in ("train1000", "eval250"):
            selected.extend(
                [entry for entry in entries if entry["dataset"] == dataset][
                    : args.limit
                ]
            )
        entries = selected
    rules = shin_rule_names()
    hashes = source_hashes()
    source_bundle = source_bundle_sha256(hashes)

    meta = {
        "protocol": "v10_shin_adapted16_totalsamples1000",
        "status": "running",
        "created_at_unix": time.time(),
        "pid": os.getpid(),
        "git_sha": git_sha(),
        "source_hashes": hashes,
        "source_bundle_sha256": source_bundle,
        "evaluation_seed_start": args.seed,
        "evaluation_seed_end": args.seed + args.n_eps - 1,
        "n_episodes_per_policy_per_coordinate": args.n_eps,
        "n_coordinates": len(entries),
        "n_train_coordinates": sum(
            entry["dataset"] == "train1000" for entry in entries
        ),
        "n_eval_coordinates": sum(
            entry["dataset"] == "eval250" for entry in entries
        ),
        "n_heuristic_rules": len(rules),
        "heuristic_episode_count": len(entries) * len(rules) * args.n_eps,
        "rules": rules,
        "rule_provenance": {
            "Threshold": "Jacobson 계열, Shin–Lee 식 (9)",
            "2Step": "Jacobson 계열, Shin–Lee 식 (10)",
            "PIH": "Mills 계열을 Shin–Lee가 수정한 식 (11)",
            "Integrated": "Shin–Lee 제안 규칙, 식 (7)·(8) 및 Figure 6",
        },
        "adaptation": (
            "Tier2/3, helipad, capacity hard mask, AMB/UAV mode, "
            "actual treatment rate and handover time"
        ),
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "eval_manifest": str(eval_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "scenario_bundle_sha256": scenario_bundle_sha256(entries),
        "environment": {
            "MCI_CAP_GATE": "occ",
            "MCI_OBS_VARIANT": "essential+load+valid",
            "MCI_H_PAD": "47",
            "MCI_REWARD_MODE": "woG",
        },
        "simulation_seed_scheme": (
            f"each coordinate and rule uses {args.seed}.."
            f"{args.seed + args.n_eps - 1}"
        ),
        "policy_rng_seed_scheme": (
            "SeedSequence([episode_seed, 0x5348494E]); "
            "separate generator from simulation RNG"
        ),
        "best_criterion": (
            "minimum mean PDR_woG per coordinate over 16 rules; "
            "post-hoc Best-of-16 oracle baseline"
        ),
        "metric_order_in_work_npz": list(METRIC_NAMES),
        "workers": args.workers,
        "checkpoint_every_rules": args.checkpoint_every,
    }
    meta_path = out_dir / "protocol_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[shin16] coords={len(entries)} rules={len(rules)} "
        f"n_eps={args.n_eps} seed={args.seed} workers={args.workers} "
        f"phase={args.phase}",
        flush=True,
    )
    print(f"[shin16] source_bundle={source_bundle}", flush=True)

    if args.phase in ("all", "shin"):
        jobs = []
        reused = 0
        for entry in entries:
            path = work_path(out_dir, entry)
            if valid_checkpoint(
                path,
                args.n_eps,
                args.seed,
                rules,
                source_bundle,
            ):
                reused += 1
                continue
            jobs.append(
                (
                    entry,
                    str(path),
                    args.n_eps,
                    args.seed,
                    args.checkpoint_every,
                    source_bundle,
                )
            )
        print(f"[shin] completed_reused={reused}", flush=True)
        run_pool(jobs, args.workers, out_dir / "failed_jobs.json")

    if args.phase in ("all", "aggregate"):
        aggregate(
            entries,
            out_dir,
            args.n_eps,
            args.seed,
            rules,
            source_bundle,
        )

    meta["status"] = "complete"
    meta["completed_at_unix"] = time.time()
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[shin16] 완료: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
