"""v10 논문용 전국 기준선 — 1,250좌표에서 Full64와 T4를 각 1,000 episode 평가한다.

대상:
  * 학습분포: random4 1,000좌표
  * 일반화: 시군구 대표점 250좌표

프로토콜:
  * 동적 시뮬레이션 seed = RL seed와 같은 0에서 시작해 0..999
  * HEUR64: 모든 좌표에서 64개 full-factorial 규칙을 전수 평가
  * HEUR64 Best: 좌표별 평균 PDR_woG가 가장 낮은 규칙을 full 결과에서 발췌
  * LB-T4: 해당 좌표의 HEUR64 Best에 병원별 발송상한 T=4를 적용
  * 환경: occ 게이트, essential+load+valid, H_PAD=47

산출:
  results/scoreboard/v10/full1000/
    heuristic_full_summary.csv       1,250×64 행
    heuristic_best_summary.csv       좌표별 Best 1,250행
    t4_summary.csv                    좌표별 T4 1,250행
    baseline_summary.csv              HEUR Best/T4 나란히 1,250행
    baseline_episodes.csv.gz          HEUR Best/T4 × 좌표 × 1,000 episode
    protocol_meta.json                hash·seed·환경·정합성
    work/{heur,t4}/...npz             좌표별 재개 체크포인트

실행:
  python src/rl_src/v10_full_baselines.py --workers 112 --n_eps 1000 --seed 0

Full64 좌표 작업은 길기 때문에 규칙 8개마다 원자적 체크포인트를 갱신한다. 같은 명령을
다시 실행하면 완료 좌표·규칙을 검증해 건너뛴다.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
DEFAULT_OUT = REPO / "results/scoreboard/v10/full1000"
KEY_RE = re.compile(r"^(?P<region>.+)_(?P<sigcd>\d{5})(?:_p(?P<point>[0-3]))?$")
METRIC_NAMES = ("reward", "pdr", "reward_woG", "pdr_woG", "time")
SOURCE_PATHS = (
    REPO / "src/rl_src/loadbalance_heuristic.py",
    REPO / "src/sim_src/RuleManager.py",
    REPO / "src/sim_src/EventManager.py",
    REPO / "src/sim_src/ScenarioManager.py",
    Path(__file__).resolve(),
)


def all_rule_names() -> list[str]:
    out = []
    for priority in ("START", "ReSTART"):
        for hospital in ("RedOnly", "YellowNearest"):
            for red in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                for yellow in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                    out.append(f"{priority}, {hospital}, Red {red}, Yellow {yellow}")
    if len(out) != 64:
        raise AssertionError("Full-factorial 규칙 수가 64가 아님")
    return out


def parse_manifest_entry(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and "path" in entry:
        return entry["path"]
    raise ValueError(f"지원하지 않는 매니페스트 엔트리: {entry!r}")


def key_parts(key: str) -> tuple[str, str, str]:
    match = KEY_RE.match(key)
    if not match:
        raise ValueError(f"시군구 키 형식 오류: {key!r}")
    point = f"p{match.group('point')}" if match.group("point") is not None else "representative"
    return match.group("region"), match.group("sigcd"), point


def coord_parts(config_path: str) -> tuple[float, float]:
    raw = Path(config_path).resolve().parent.name.strip("()")
    lat, lon = raw.split(",")
    return float(lat), float(lon)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    """커밋되지 않은 시뮬 수정까지 실험 provenance에 봉인한다."""
    return {str(path.relative_to(REPO)): sha256_file(path) for path in SOURCE_PATHS}


def source_bundle_sha256(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path, digest in sorted(hashes.items()):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def scenario_bundle_sha256(entries: list[dict]) -> str:
    h = hashlib.sha256()
    for entry in sorted(entries, key=lambda x: (x["dataset"], x["key"])):
        path = Path(entry["config"]).resolve()
        h.update(entry["dataset"].encode("utf-8"))
        h.update(b"\0")
        h.update(entry["key"].encode("utf-8"))
        h.update(b"\0")
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def validate_inputs(train_path: Path, eval_path: Path, n_eps: int, strict: bool) -> list[dict]:
    train = json.load(open(train_path, encoding="utf-8"))
    evaluation = json.load(open(eval_path, encoding="utf-8"))
    if len(train) != 1000 or len(evaluation) != 250:
        raise ValueError(f"매니페스트 크기 오류: train={len(train)}, eval={len(evaluation)}")

    groups: dict[str, set[int]] = {}
    entries = []
    coords_by_dataset = {}
    for dataset, manifest in (("train1000", train), ("eval250", evaluation)):
        coords = set()
        for key, raw_entry in manifest.items():
            region, sigcd, point = key_parts(key)
            if dataset == "train1000":
                if point == "representative":
                    raise ValueError(f"train1000 키에 p0~p3 없음: {key}")
                groups.setdefault(sigcd, set()).add(int(point[1:]))
            elif point != "representative":
                raise ValueError(f"eval250 키에 p 접미사 존재: {key}")
            config = str(Path(parse_manifest_entry(raw_entry)).resolve())
            if not os.path.exists(config):
                raise FileNotFoundError(config)
            lat, lon = coord_parts(config)
            coord = (lat, lon)
            if coord in coords:
                raise ValueError(f"{dataset} 좌표 중복: {coord}")
            coords.add(coord)
            cfg = yaml.safe_load(open(config, encoding="utf-8"))
            total_samples = int(cfg["run_setting"]["totalSamples"])
            yaml_seed = int(cfg["run_setting"]["random_seed"])
            if strict and total_samples != n_eps:
                raise ValueError(f"{key}: YAML totalSamples={total_samples}, 요청 n_eps={n_eps}")
            if strict and yaml_seed != 0:
                raise ValueError(f"{key}: YAML random_seed={yaml_seed}, RL seed 0과 불일치")
            entries.append(
                {
                    "dataset": dataset,
                    "key": key,
                    "region": region,
                    "sigcd": sigcd,
                    "point": point,
                    "lat": lat,
                    "lon": lon,
                    "config": config,
                    "yaml_totalSamples": total_samples,
                    "yaml_random_seed": yaml_seed,
                }
            )
        coords_by_dataset[dataset] = coords

    bad = {sigcd: sorted(points) for sigcd, points in groups.items() if points != {0, 1, 2, 3}}
    if len(groups) != 250 or bad:
        raise ValueError(f"random4 구조 오류: groups={len(groups)}, bad={list(bad.items())[:5]}")
    overlap = coords_by_dataset["train1000"] & coords_by_dataset["eval250"]
    if overlap:
        raise ValueError(f"train/eval 좌표 중복 {len(overlap)}개: {sorted(overlap)[:5]}")
    return entries


def work_path(out_dir: Path, phase: str, entry: dict) -> Path:
    return out_dir / "work" / phase / entry["dataset"] / f"{entry['key']}.npz"


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def valid_heur_checkpoint(path: Path, n_eps: int, seed: int, rules: list[str]) -> bool:
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
                and np.isfinite(data["values"]).all()
            )
    except Exception:
        return False


def valid_t4_checkpoint(path: Path, n_eps: int, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["values"].shape == (n_eps, len(METRIC_NAMES))
                and np.array_equal(data["seeds"], np.arange(seed, seed + n_eps))
                and np.isfinite(data["values"]).all()
                and str(data["best_rule"].item()) != ""
            )
    except Exception:
        return False


def rollout(factory, policy, episode_seed: int) -> tuple[float, float, float, float, float]:
    env = factory(seed=episode_seed)
    obs, _ = env.reset(seed=episode_seed)
    done = False
    reward = 0.0
    reward_wog = 0.0
    last_time = 0.0
    while not done:
        mask = env.action_masks()
        action = policy(obs, mask, env.unwrapped)
        obs, r, terminated, truncated, info = env.step(action)
        reward += float(r)
        reward_wog += float(info.get("r_woG", 0.0))
        last_time = float(info.get("time", 0.0))
        done = terminated or truncated
    preventable = float(env.unwrapped.preventable)
    preventable_wog = float(env.unwrapped.preventable_woG)
    pdr = 1.0 - reward / preventable if preventable > 0 else 0.0
    pdr_wog = 1.0 - reward_wog / preventable_wog if preventable_wog > 0 else 0.0
    return reward, pdr, reward_wog, pdr_wog, last_time


def heuristic_worker(job):
    entry, path_str, n_eps, seed, checkpoint_every = job
    path = Path(path_str)
    rules = all_rule_names()
    try:
        import torch

        torch.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from distill_policy import make_heuristic_policy
        from viper_distill import _suppress_stdout, make_feature_env

        seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
        values = np.full((len(rules), n_eps, len(METRIC_NAMES)), np.nan, dtype=np.float32)
        done = np.zeros(len(rules), dtype=bool)
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if (
                    old["values"].shape == values.shape
                    and np.array_equal(old["seeds"], seeds)
                    and old["rule_names"].tolist() == rules
                ):
                    values[:] = old["values"]
                    done[:] = old["done"]

        factory = make_feature_env(entry["config"], None)
        completed_since_save = 0
        with _suppress_stdout():
            for rule_idx, rule_name in enumerate(rules):
                if done[rule_idx] and np.isfinite(values[rule_idx]).all():
                    continue
                policy = make_heuristic_policy(rule_name)
                for ep, episode_seed in enumerate(seeds):
                    values[rule_idx, ep] = rollout(factory, policy, int(episode_seed))
                done[rule_idx] = True
                completed_since_save += 1
                if completed_since_save >= checkpoint_every:
                    atomic_savez(
                        path,
                        values=values,
                        done=done,
                        seeds=seeds,
                        rule_names=np.asarray(rules),
                    )
                    completed_since_save = 0
        atomic_savez(path, values=values, done=done, seeds=seeds, rule_names=np.asarray(rules))
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
            "err": (str(exc) + traceback.format_exc())[:2000],
        }


def t4_worker(job):
    entry, heur_path_str, t4_path_str, n_eps, seed = job
    heur_path = Path(heur_path_str)
    t4_path = Path(t4_path_str)
    rules = all_rule_names()
    try:
        import torch

        torch.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from loadbalance_heuristic import make_cap_policy
        from viper_distill import _suppress_stdout, make_feature_env

        with np.load(heur_path, allow_pickle=False) as data:
            heur_values = np.asarray(data["values"])
            if heur_values.shape != (64, n_eps, len(METRIC_NAMES)) or not data["done"].all():
                raise ValueError(f"HEUR 체크포인트 불완전: {heur_path}")
            best_idx = int(np.argmin(heur_values[:, :, 3].mean(axis=1)))
        best_rule = rules[best_idx]
        seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
        values = np.zeros((n_eps, len(METRIC_NAMES)), dtype=np.float32)
        factory = make_feature_env(entry["config"], None)
        policy = make_cap_policy(best_rule, 4)
        with _suppress_stdout():
            for ep, episode_seed in enumerate(seeds):
                values[ep] = rollout(factory, policy, int(episode_seed))
        atomic_savez(
            t4_path,
            values=values,
            seeds=seeds,
            best_rule=np.asarray(best_rule),
            heur_best_idx=np.asarray(best_idx, dtype=np.int64),
        )
        return {
            "ok": True,
            "dataset": entry["dataset"],
            "key": entry["key"],
            "best_rule": best_rule,
            "pdr_wog": float(values[:, 3].mean()),
        }
    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "dataset": entry["dataset"],
            "key": entry["key"],
            "err": (str(exc) + traceback.format_exc())[:2000],
        }


def run_pool(label: str, worker_fn, jobs: list, workers: int) -> None:
    if not jobs:
        print(f"[{label}] 완료 체크포인트 재사용 — 실행 job=0", flush=True)
        return
    print(f"[{label}] jobs={len(jobs)} workers={min(workers, len(jobs))}", flush=True)
    failed = []
    t0 = time.time()
    with Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for idx, result in enumerate(pool.imap_unordered(worker_fn, jobs), 1):
            if result["ok"]:
                metric = result.get("best_pdr_wog", result.get("pdr_wog", float("nan")))
                print(
                    f"  [{idx}/{len(jobs)}] {result['dataset']}:{result['key']} "
                    f"PDR_woG={metric:.5f} ({(time.time() - t0) / 60:.1f}분)",
                    flush=True,
                )
            else:
                failed.append(result)
                print(
                    f"  [{idx}/{len(jobs)}] FAIL {result['dataset']}:{result['key']} "
                    f"{result['err'][:300]}",
                    flush=True,
                )
    if failed:
        fail_path = REPO / "results/scoreboard/v10/full1000/failed_jobs.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"[{label}] 실패 {len(failed)}개 — {fail_path}")
    print(f"[{label}] 완료 wall={(time.time() - t0) / 3600:.2f}시간", flush=True)


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
        s = stats(values[:, metric_idx])
        out[f"{prefix}_{metric}_mean"] = s["mean"]
        out[f"{prefix}_{metric}_std"] = s["std"]
        out[f"{prefix}_{metric}_ci95"] = s["ci95"]
    return out


def base_row(entry: dict, n_eps: int, seed: int) -> dict:
    return {
        "dataset": entry["dataset"],
        "coordinate_key": entry["key"],
        "region": entry["region"],
        "sigcd": entry["sigcd"],
        "point": entry["point"],
        "lat": entry["lat"],
        "lon": entry["lon"],
        "n_episodes": n_eps,
        "seed_start": seed,
        "seed_end": seed + n_eps - 1,
    }


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"빈 CSV 산출 시도: {path}")
    temp = Path(str(path) + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def aggregate(entries: list[dict], out_dir: Path, n_eps: int, seed: int, rules: list[str]) -> None:
    full_rows = []
    best_rows = []
    t4_rows = []
    baseline_rows = []
    episode_path = out_dir / "baseline_episodes.csv.gz"
    episode_tmp = Path(str(episode_path) + ".tmp")
    episode_tmp.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_tmp, "wt", newline="", encoding="utf-8") as f:
        episode_fields = [
            "dataset", "coordinate_key", "region", "sigcd", "point", "lat", "lon",
            "policy", "episode", "seed", "reward", "pdr", "reward_woG", "pdr_woG", "time",
        ]
        episode_writer = csv.DictWriter(f, fieldnames=episode_fields)
        episode_writer.writeheader()

        for entry_idx, entry in enumerate(entries, 1):
            heur_path = work_path(out_dir, "heur", entry)
            t4_path = work_path(out_dir, "t4", entry)
            if not valid_heur_checkpoint(heur_path, n_eps, seed, rules):
                raise RuntimeError(f"HEUR 산출 불완전: {heur_path}")
            if not valid_t4_checkpoint(t4_path, n_eps, seed):
                raise RuntimeError(f"T4 산출 불완전: {t4_path}")

            with np.load(heur_path, allow_pickle=False) as heur:
                heur_values = np.asarray(heur["values"], dtype=np.float64)
            means = heur_values[:, :, 3].mean(axis=1)
            best_idx = int(np.argmin(means))
            best_rule = rules[best_idx]
            best_values = heur_values[best_idx]
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
                row.update(metric_summary("heur", heur_values[rule_idx]))
                full_rows.append(row)

            best_row = base_row(entry, n_eps, seed)
            best_row.update({"best_rule_index": best_idx, "best_rule": best_rule})
            best_row.update(metric_summary("heur_best", best_values))
            best_rows.append(best_row)

            with np.load(t4_path, allow_pickle=False) as t4:
                t4_values = np.asarray(t4["values"], dtype=np.float64)
                t4_rule = str(t4["best_rule"].item())
            if t4_rule != best_rule:
                raise RuntimeError(f"T4 base rule 불일치 {entry['key']}: {t4_rule} != {best_rule}")
            t4_row = base_row(entry, n_eps, seed)
            t4_row.update({"base_rule_index": best_idx, "base_rule": best_rule, "T": 4})
            t4_row.update(metric_summary("t4", t4_values))
            t4_rows.append(t4_row)

            combined = base_row(entry, n_eps, seed)
            combined.update({"heur_best_rule_index": best_idx, "heur_best_rule": best_rule, "T": 4})
            combined.update(metric_summary("heur_best", best_values))
            combined.update(metric_summary("t4", t4_values))
            combined["PDR_woG_improvement_T4_vs_HEUR"] = float(
                best_values[:, 3].mean() - t4_values[:, 3].mean()
            )
            baseline_rows.append(combined)

            seeds = np.arange(seed, seed + n_eps, dtype=np.int64)
            for policy_name, values in (("HEUR64_BEST", best_values), ("LB_T4", t4_values)):
                for ep in range(n_eps):
                    episode_writer.writerow(
                        {
                            "dataset": entry["dataset"],
                            "coordinate_key": entry["key"],
                            "region": entry["region"],
                            "sigcd": entry["sigcd"],
                            "point": entry["point"],
                            "lat": entry["lat"],
                            "lon": entry["lon"],
                            "policy": policy_name,
                            "episode": ep,
                            "seed": int(seeds[ep]),
                            "reward": float(values[ep, 0]),
                            "pdr": float(values[ep, 1]),
                            "reward_woG": float(values[ep, 2]),
                            "pdr_woG": float(values[ep, 3]),
                            "time": float(values[ep, 4]),
                        }
                    )
            if entry_idx % 50 == 0:
                print(f"[aggregate] {entry_idx}/{len(entries)}", flush=True)

    os.replace(episode_tmp, episode_path)
    write_csv_atomic(out_dir / "heuristic_full_summary.csv", full_rows)
    write_csv_atomic(out_dir / "heuristic_best_summary.csv", best_rows)
    write_csv_atomic(out_dir / "t4_summary.csv", t4_rows)
    write_csv_atomic(out_dir / "baseline_summary.csv", baseline_rows)
    print(
        f"[aggregate] full={len(full_rows)} best={len(best_rows)} t4={len(t4_rows)} "
        f"episodes={len(entries) * 2 * n_eps}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default=str(TRAIN_MANIFEST))
    p.add_argument("--eval_manifest", default=str(EVAL_MANIFEST))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--n_eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=112)
    p.add_argument("--checkpoint_every", type=int, default=8)
    p.add_argument("--phase", choices=["all", "heur", "t4", "aggregate"], default="all")
    p.add_argument("--limit", type=int, default=0, help="스모크용 데이터셋별 앞 N개 좌표")
    p.add_argument("--no_strict", action="store_true", help="YAML totalSamples/seed 일치 검사 해제")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_eps <= 0 or args.workers <= 0:
        raise ValueError("n_eps/workers는 양수여야 함")
    train_path = Path(args.train_manifest).resolve()
    eval_path = Path(args.eval_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = validate_inputs(train_path, eval_path, args.n_eps, not args.no_strict)
    if args.limit:
        selected = []
        for dataset in ("train1000", "eval250"):
            selected.extend([e for e in entries if e["dataset"] == dataset][: args.limit])
        entries = selected
    rules = all_rule_names()

    hashes = source_hashes()
    meta = {
        "protocol": "v10_full64_t4_totalsamples1000",
        "status": "running",
        "created_at_unix": time.time(),
        "git_sha": git_sha(),
        "source_hashes": hashes,
        "source_bundle_sha256": source_bundle_sha256(hashes),
        "rl_training_seed": 0,
        "evaluation_seed_start": args.seed,
        "evaluation_seed_end": args.seed + args.n_eps - 1,
        "n_episodes_per_policy_per_coordinate": args.n_eps,
        "n_coordinates": len(entries),
        "n_train_coordinates": sum(e["dataset"] == "train1000" for e in entries),
        "n_eval_coordinates": sum(e["dataset"] == "eval250" for e in entries),
        "n_heuristic_rules": 64,
        "heuristic_episode_count": len(entries) * 64 * args.n_eps,
        "t4_episode_count": len(entries) * args.n_eps,
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
        "best_criterion": "minimum mean PDR_woG per coordinate over 64 rules",
        "t4_definition": "T=4 applied to each coordinate's HEUR64 Best base rule",
        "metric_order_in_work_npz": list(METRIC_NAMES),
        "workers": args.workers,
        "checkpoint_every_rules": args.checkpoint_every,
    }
    meta_path = out_dir / "protocol_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[v10-baseline] coords={len(entries)} n_eps={args.n_eps} seed={args.seed} "
        f"workers={args.workers} phase={args.phase}",
        flush=True,
    )

    if args.phase in ("all", "heur"):
        heur_jobs = []
        reused = 0
        for entry in entries:
            path = work_path(out_dir, "heur", entry)
            if valid_heur_checkpoint(path, args.n_eps, args.seed, rules):
                reused += 1
                continue
            heur_jobs.append((entry, str(path), args.n_eps, args.seed, args.checkpoint_every))
        print(f"[heur] completed_reused={reused}", flush=True)
        run_pool("heur", heuristic_worker, heur_jobs, args.workers)

    if args.phase in ("all", "t4"):
        t4_jobs = []
        reused = 0
        for entry in entries:
            heur_path = work_path(out_dir, "heur", entry)
            if not valid_heur_checkpoint(heur_path, args.n_eps, args.seed, rules):
                raise RuntimeError(f"T4 전에 HEUR Full64 완료 필요: {heur_path}")
            path = work_path(out_dir, "t4", entry)
            if valid_t4_checkpoint(path, args.n_eps, args.seed):
                reused += 1
                continue
            t4_jobs.append((entry, str(heur_path), str(path), args.n_eps, args.seed))
        print(f"[t4] completed_reused={reused}", flush=True)
        run_pool("t4", t4_worker, t4_jobs, args.workers)

    if args.phase in ("all", "aggregate"):
        aggregate(entries, out_dir, args.n_eps, args.seed, rules)

    meta["status"] = "complete"
    meta["completed_at_unix"] = time.time()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v10-baseline] 완료: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
