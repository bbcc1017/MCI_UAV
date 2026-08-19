# -*- coding: utf-8 -*-
"""v17 순수 PPO 교사 closed-loop 재평가 — 트리와 완전히 동일한 배관.

``v10_tree_eval.py`` 의 rollout·seed·CSV 규약을 그대로 쓰고 정책만 PPO 로 바꾼다.
기존 v11 PPO cube(2026-07-28 생성)는 sim 정정 커밋 ``b01efd3``(2026-08-12, 병원 후보
순회를 수단별 실제 ETA 순으로 정정) **이전** 산출물이라, 정정 이후에 돌린 트리·LB 결과와
직접 비교할 수 없다. 이 스크립트는 같은 코드베이스에서 PPO 를 다시 돌려 그 혼용을 없앤다.
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


REPO = Path(__file__).resolve().parents[2]
EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
COLS = [
    "region", "policy", "info_level", "complexity", "episode", "seed",
    "reward_woG", "pdr_woG", "sim_time", "n_decisions", "ms_per_decision",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rollout(factory, policy, seed: int):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done, reward, n_dec, policy_sec = False, 0.0, 0, 0.0
    info = {}
    while not done:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action = policy(obs, mask, env.unwrapped)
        policy_sec += time.perf_counter() - t0
        n_dec += 1
        obs, _, term, trunc, info = env.step(action)
        reward += info.get("r_woG", 0.0)
        done = term or trunc
    preventable = env.unwrapped.preventable_woG
    pdr = 1.0 - reward / preventable if preventable > 0 else 0.0
    return reward, pdr, float(info.get("time", np.nan)), n_dec, policy_sec * 1000 / max(n_dec, 1)


def worker(job):
    region, cfg, model_dir, n_eps, seed0 = job
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
        import pad_vecnorm  # noqa: F401
        from evaluate import ppo_policy
        from viper_distill import _suppress_stdout, load_vecnorm, make_feature_env

        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        norm = load_vecnorm(os.path.join(model_dir, "vecnormalize.pkl"))
        policy = ppo_policy(model)
        factory = make_feature_env(cfg, norm)
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                reward, pdr, sim_time, n_dec, ms = rollout(factory, policy, seed)
                rows.append({
                    "region": region,
                    "policy": "PPO_POINTER_V10",
                    "info_level": "PPO",
                    "complexity": "-",
                    "episode": ep,
                    "seed": seed,
                    "reward_woG": reward,
                    "pdr_woG": pdr,
                    "sim_time": sim_time,
                    "n_decisions": n_dec,
                    "ms_per_decision": ms,
                })
        return {"ok": True, "region": region, "rows": rows}
    except Exception as exc:
        import traceback

        return {"ok": False, "region": region, "err": (str(exc) + traceback.format_exc())[:1500]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(EVAL_MANIFEST))
    p.add_argument("--model_dir",
                   default=str(REPO / "results/rl/redesign/v10_random4_1000_pointer_s0"))
    p.add_argument("--regions", default="")
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    if manifest_path == EVAL_MANIFEST.resolve():
        if len(manifest) != 250 or any(k.endswith(("_p0", "_p1", "_p2", "_p3")) for k in manifest):
            raise ValueError("대표점250 manifest 구조 오류")
    keys = [k for k in args.regions.split(",") if k in manifest] if args.regions else list(manifest)
    cases = ["PPO_POINTER_V10"]
    model_dir = str(Path(args.model_dir).resolve())

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done_regions = set()
    if out.exists():
        existing = {}
        with open(out, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.setdefault(row["region"], set()).add(
                    (row["policy"], int(row["episode"]), int(row["seed"]))
                )
        expected_n = len(cases) * args.n_eps
        done_regions = {k for k, v in existing.items() if len(v) == expected_n}
        incomplete = set(existing) - done_regions
        if incomplete:
            raise RuntimeError(f"부분 기록 지역 발견(수동 정리 필요): {sorted(incomplete)[:3]}")
    jobs = [
        (key, manifest[key], model_dir, args.n_eps, args.seed0)
        for key in keys if key not in done_regions
    ]
    print(
        f"[ppo-eval] regions={len(keys)} remaining={len(jobs)} cases={len(cases)} "
        f"n_eps={args.n_eps} seed={args.seed0}..{args.seed0+args.n_eps-1} "
        f"workers={min(args.workers,max(len(jobs),1))}",
        flush=True,
    )
    new_file = not out.exists()
    fout = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        writer.writeheader()
        fout.flush()
    t0, n_rows = time.time(), 0
    if jobs:
        with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
            for i, result in enumerate(pool.imap_unordered(worker, jobs), 1):
                if not result["ok"]:
                    fout.close()
                    raise RuntimeError(f"{result['region']} 평가 실패: {result['err']}")
                writer.writerows(result["rows"])
                fout.flush()
                n_rows += len(result["rows"])
                avg = float(np.mean([x["pdr_woG"] for x in result["rows"]]))
                print(
                    f"  [{i}/{len(jobs)}] {result['region']} rows={len(result['rows'])} "
                    f"case-avg={avg:.4f} total={n_rows} wall={time.time()-t0:.0f}s",
                    flush=True,
                )
    fout.close()

    # 전체 완전성 검증
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        key = (row["region"], row["policy"], int(row["episode"]), int(row["seed"]))
        if key in seen:
            raise RuntimeError(f"평가 중복: {key}")
        seen.add(key)
        pdr = float(row["pdr_woG"])
        if not np.isfinite(pdr) or not 0 <= pdr <= 1:
            raise RuntimeError(f"PDR 오류: {key}={pdr}")
    expected = len(keys) * len(cases) * args.n_eps
    if len(rows) != expected:
        raise RuntimeError(f"평가 행수 불일치 {len(rows)} != {expected}")

    meta = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_dir": model_dir,
        "model_sha256": sha256_file(Path(model_dir) / "final_model.zip"),
        "note": "sim 정정 b01efd3 이후 코드베이스에서 재실행 (v11 cube 는 정정 이전)",
        "cases": cases,
        "n_regions": len(keys),
        "n_eps_per_region": args.n_eps,
        "seed_start": args.seed0,
        "seed_end": args.seed0 + args.n_eps - 1,
        "environment": {
            "MCI_CAP_GATE": "occ",
            "MCI_OBS_VARIANT": "essential+load+valid",
            "MCI_H_PAD": "47",
        },
        "n_rows": len(rows),
        "output": str(out),
        "output_sha256": sha256_file(out),
    }
    Path(str(out) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ppo-eval] 완료 rows={len(rows)} wall={(time.time()-t0)/60:.1f}분 → {out}", flush=True)


if __name__ == "__main__":
    main()
