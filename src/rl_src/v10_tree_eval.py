# -*- coding: utf-8 -*-
"""v10 증류 트리 closed-loop paired 평가.

대표점 250개에서 모든 트리를 동일 episode seed로 재시뮬레이션한다. 출력은
``region, policy, episode, seed`` long-format이며 지역 단위로만 append해 안전하게 재개한다.
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

from tree_distill_policy import load_tree_package

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
    region, cfg, tree_paths, n_eps, seed0 = job
    try:
        import torch as th

        th.set_num_threads(1)
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from tree_distill_policy import make_rank_tree_policy
        from viper_distill import _suppress_stdout, make_feature_env

        packages = [load_tree_package(path) for path in tree_paths]
        policies = [make_rank_tree_policy(p, h_pad=47) for p in packages]
        factory = make_feature_env(cfg, None)
        rows = []
        with _suppress_stdout():
            for ep in range(n_eps):
                seed = seed0 + ep
                for package, policy in zip(packages, policies):
                    reward, pdr, sim_time, n_dec, ms = rollout(factory, policy, seed)
                    rows.append({
                        "region": region,
                        "policy": f"{package['info_level']}_{package['complexity']}",
                        "info_level": package["info_level"],
                        "complexity": package["complexity"],
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
    p.add_argument("--tree_dir", required=True)
    p.add_argument("--cases", default="", help="쉼표구분 I*_C*; 비우면 fit_summary 전부")
    p.add_argument("--regions", default="")
    p.add_argument("--n_eps", type=int, default=30)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    if manifest_path == EVAL_MANIFEST.resolve():
        if len(manifest) != 250 or any(k.endswith(("_p0", "_p1", "_p2", "_p3")) for k in manifest):
            raise ValueError("대표점250 manifest 구조 오류")
    keys = [k for k in args.regions.split(",") if k in manifest] if args.regions else list(manifest)
    tree_dir = Path(args.tree_dir).resolve()
    if args.cases:
        cases = [x for x in args.cases.split(",") if x]
    else:
        with open(tree_dir / "fit_summary.csv", encoding="utf-8-sig") as f:
            cases = [r["policy"] for r in csv.DictReader(f)]
    tree_paths = [str(tree_dir / f"{case}.pkl") for case in cases]
    missing = [x for x in tree_paths if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError(f"트리 누락: {missing[:3]}")
    for path in tree_paths:
        package = load_tree_package(path)
        expected = f"{package['info_level']}_{package['complexity']}"
        if Path(path).stem != expected:
            raise ValueError(f"트리 파일명/메타 불일치: {path} != {expected}")

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
        (key, manifest[key], tree_paths, args.n_eps, args.seed0)
        for key in keys if key not in done_regions
    ]
    print(
        f"[tree-eval] regions={len(keys)} remaining={len(jobs)} cases={len(cases)} "
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
        "tree_dir": str(tree_dir),
        "tree_hashes": {Path(x).stem: sha256_file(x) for x in tree_paths},
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
    print(f"[tree-eval] 완료 rows={len(rows)} wall={(time.time()-t0)/60:.1f}분 → {out}", flush=True)


if __name__ == "__main__":
    main()
