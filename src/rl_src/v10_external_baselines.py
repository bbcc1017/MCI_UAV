# -*- coding: utf-8 -*-
"""외부 테스트용 사전고정 휴리스틱 기준선.

v10 train1000의 Full64 결과를 지역 구분 없이 평균해 가장 낮은 규칙 하나를 먼저 고정하고,
새 외부 좌표에서는 그 규칙과 동일 규칙+T4만 공통 seed로 평가한다. 외부 좌표에서 다시
64개 중 최저를 고르는 Best-of-64 oracle이 아니므로 배포 가능한 누수 없는 기준선이다.
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
import pandas as pd

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "scenarios/manifests/distill_external_test250_osrm_manifest.json"
DEFAULT_FULL = REPO / "results/scoreboard/v10/full1000/heuristic_full_summary.csv"
COLS = [
    "region", "method", "rule", "episode", "seed", "pdr_woG", "reward_woG",
    "sim_time", "n_decisions", "ms_per_decision",
]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_global_rule(path: Path) -> tuple[int, str, pd.DataFrame]:
    data = pd.read_csv(path)
    train = data[data["dataset"] == "train1000"]
    if len(train) != 1000 * 64:
        raise RuntimeError(f"train1000 Full64 행수 {len(train)} != 64000")
    rank = (
        train.groupby(["rule_index", "rule"], as_index=False)["heur_pdr_woG_mean"]
        .mean()
        .sort_values(["heur_pdr_woG_mean", "rule_index"], kind="stable")
        .reset_index(drop=True)
    )
    row = rank.iloc[0]
    return int(row["rule_index"]), str(row["rule"]), rank


def _rollout(factory, policy, seed: int):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    reward = 0.0
    n_dec = 0
    policy_sec = 0.0
    info = {}
    while not done:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action = policy(obs, mask, env.unwrapped)
        policy_sec += time.perf_counter() - t0
        n_dec += 1
        obs, _, term, trunc, info = env.step(action)
        reward += float(info.get("r_woG", 0.0))
        done = term or trunc
    preventable = float(env.unwrapped.preventable_woG)
    return {
        "pdr_woG": 1.0 - reward / preventable if preventable > 0 else 0.0,
        "reward_woG": reward,
        "sim_time": float(info.get("time", np.nan)),
        "n_decisions": n_dec,
        "ms_per_decision": policy_sec * 1000.0 / max(n_dec, 1),
    }


def _worker(job):
    region, cfg, rule, n_eps, seed0 = job
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from distill_policy import make_heuristic_policy
        from loadbalance_heuristic import make_cap_policy
        from viper_distill import _suppress_stdout, make_feature_env

        factory = make_feature_env(cfg, None)
        policies = {
            "HEUR64_GLOBAL_TRAIN_BEST": make_heuristic_policy(rule),
            "GLOBAL_TRAIN_BEST_T4": make_cap_policy(rule, 4),
        }
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                for method, policy in policies.items():
                    rows.append({
                        "region": region,
                        "method": method,
                        "rule": rule,
                        "episode": ep,
                        "seed": seed,
                        **_rollout(factory, policy, seed),
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
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--full64_summary", default=str(DEFAULT_FULL))
    ap.add_argument("--n_eps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    full_path = Path(args.full64_summary).resolve()
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    rule_idx, rule, ranking = select_global_rule(full_path)
    print(
        f"[external-baseline] regions={len(manifest)} rule#{rule_idx}={rule} "
        f"train_mean={ranking.iloc[0]['heur_pdr_woG_mean']:.6f}",
        flush=True,
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        old = list(csv.DictReader(open(out, encoding="utf-8")))
        by = {}
        for row in old:
            by.setdefault(row["region"], set()).add((row["method"], int(row["seed"])))
        expected = {
            (method, args.seed0 + ep)
            for method in ("HEUR64_GLOBAL_TRAIN_BEST", "GLOBAL_TRAIN_BEST_T4")
            for ep in range(args.n_eps)
        }
        done = {region for region, got in by.items() if got == expected}
        if set(by) - done:
            raise RuntimeError("부분 기록 지역 존재")
    jobs = [
        (region, cfg, rule, args.n_eps, args.seed0)
        for region, cfg in manifest.items() if region not in done
    ]
    new_file = not out.exists()
    fout = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        writer.writeheader()
    t0 = time.time()
    with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, jobs), 1):
            if not result["ok"]:
                fout.close()
                raise RuntimeError(f"{result['region']} 실패: {result['err']}")
            writer.writerows(result["rows"])
            fout.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(jobs)}] wall={(time.time()-t0)/60:.1f}분", flush=True)
    fout.close()
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    if len(rows) != len(manifest) * 2 * args.n_eps:
        raise RuntimeError("최종 행수 불일치")
    meta = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "selection_source": str(full_path),
        "selection_source_sha256": _sha256(full_path),
        "selection_dataset": "train1000 only",
        "global_rule_index": rule_idx,
        "global_rule": rule,
        "global_rule_train1000_mean_pdr_wog": float(ranking.iloc[0]["heur_pdr_woG_mean"]),
        "external_best_of_64_used": False,
        "n_eps": args.n_eps,
        "seed_start": args.seed0,
        "seed_end": args.seed0 + args.n_eps - 1,
        "output": str(out),
        "output_sha256": _sha256(out),
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[external-baseline] 완료 rows={len(rows)} → {out}", flush=True)


if __name__ == "__main__":
    main()
