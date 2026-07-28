"""v10 휴리스틱 규칙 선택 — 학습 random4 1,000좌표에서만 시군구별 64룰을 적합한다.

대표점 250개가 최종 평가셋으로 바뀌었으므로 기존
``results/sigungu_heuristic_best.csv``를 v10 scoreboard에 쓰면 평가좌표 누수가 된다.
이 스크립트는 각 시군구의 random4(p0~p3)에서 64개 규칙을 같은 seed로 평가하고,
4개 좌표 평균 PDR_woG가 가장 낮은 규칙을 고정한다.

선택용 seed(기본 7000)는 최종 성능평가 seed(11000)와 분리한다. 출력 ``*_best.csv``는
``paired_eval_ladder.py --match sigcd --heur_csv ...``에 바로 넣을 수 있다.

예:
  python src/rl_src/fit_v10_heuristic_rules.py \
    --manifest scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json \
    --n_eps 30 --workers 32 \
    --out_prefix results/scoreboard/v10/heuristic_train1000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
DEFAULT_EVAL_MANIFEST = REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
KEY_RE = re.compile(r"^(?P<region>.+)_(?P<sigcd>\d{5})_p(?P<point>[0-3])$")


def all_rule_names() -> list[str]:
    """RuleManager full-factorial과 순서까지 동일한 64개 규칙명."""
    out = []
    for priority in ("START", "ReSTART"):
        for hospital in ("RedOnly", "YellowNearest"):
            for red in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                for yellow in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                    out.append(f"{priority}, {hospital}, Red {red}, Yellow {yellow}")
    assert len(out) == 64
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scenario_bundle_sha256(manifest: dict[str, str]) -> str:
    """키·경로·YAML 내용을 함께 묶은 데이터셋 지문."""
    h = hashlib.sha256()
    for key in sorted(manifest):
        path = Path(manifest[key]).resolve()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def coord_from_config(path: str) -> str:
    return Path(path).resolve().parent.name.strip()


def validate_random4(manifest: dict[str, str], eval_manifest: dict[str, str]) -> dict[str, dict]:
    """1,000=250×p0~p3 및 대표점과 좌표 무중복을 강제한다."""
    if len(manifest) != 1000:
        raise ValueError(f"학습 매니페스트는 정확히 1,000개여야 함: {len(manifest)}")
    groups: dict[str, dict] = {}
    for key, path in manifest.items():
        m = KEY_RE.match(key)
        if not m:
            raise ValueError(f"random4 키 형식 오류: {key!r}")
        sigcd = m.group("sigcd")
        g = groups.setdefault(sigcd, {"region": m.group("region"), "points": set(), "paths": []})
        if g["region"] != m.group("region"):
            raise ValueError(f"동일 sigcd의 지역명 불일치: {sigcd}")
        g["points"].add(int(m.group("point")))
        g["paths"].append(path)
    bad = {s: sorted(g["points"]) for s, g in groups.items() if g["points"] != {0, 1, 2, 3}}
    if len(groups) != 250 or bad:
        raise ValueError(f"random4 그룹 오류: groups={len(groups)}, bad={list(bad.items())[:5]}")

    train_coords = {coord_from_config(p) for p in manifest.values()}
    eval_coords = {coord_from_config(p) for p in eval_manifest.values()}
    overlap = train_coords & eval_coords
    if overlap:
        raise ValueError(f"학습/평가 좌표 중복 {len(overlap)}개: {sorted(overlap)[:5]}")
    if len(train_coords) != 1000:
        raise ValueError(f"학습 좌표 중복: 고유좌표={len(train_coords)}")
    return groups


def rollout_woG(factory, policy_fn, seed: int) -> tuple[float, float]:
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    reward_woG = 0.0
    while not done:
        mask = env.action_masks()
        action = policy_fn(obs, mask, env.unwrapped)
        obs, _, terminated, truncated, info = env.step(action)
        reward_woG += info.get("r_woG", 0.0)
        done = terminated or truncated
    preventable = env.unwrapped.preventable_woG
    pdr = 1.0 - reward_woG / preventable if preventable > 0 else 0.0
    return reward_woG, pdr


def worker(job):
    key, config_path, rules, n_eps, seed = job
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

        factory = make_feature_env(config_path, None)
        policies = [(name, make_heuristic_policy(name)) for name in rules]
        pdr = np.zeros((len(rules), n_eps), dtype=np.float64)
        wog = np.zeros_like(pdr)
        with _suppress_stdout():
            for ep in range(n_eps):
                episode_seed = seed + ep
                for i, (_, policy) in enumerate(policies):
                    wog[i, ep], pdr[i, ep] = rollout_woG(factory, policy, episode_seed)
        return {
            "ok": True,
            "key": key,
            "PDR": pdr.mean(axis=1).tolist(),
            "woG": wog.mean(axis=1).tolist(),
        }
    except Exception as exc:
        import traceback

        return {"ok": False, "key": key, "err": (str(exc) + traceback.format_exc())[:1000]}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--eval_manifest", default=str(DEFAULT_EVAL_MANIFEST))
    p.add_argument("--n_eps", type=int, default=30, help="규칙 선택용 좌표당 episode 수")
    p.add_argument("--seed", type=int, default=7000, help="선택 전용 seed; 평가 seed 11000과 분리")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="스모크용 앞 N개 좌표")
    p.add_argument("--out_prefix", default=str(REPO / "results/scoreboard/v10/heuristic_train1000"))
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    eval_manifest_path = Path(args.eval_manifest).resolve()
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    eval_manifest = json.load(open(eval_manifest_path, encoding="utf-8"))
    groups = validate_random4(manifest, eval_manifest)
    rules = all_rule_names()

    items = list(manifest.items())
    if args.limit:
        items = items[: args.limit]
    jobs = [(key, path, rules, args.n_eps, args.seed) for key, path in items]
    print(
        f"[heur-fit] points={len(jobs)} rules=64 n_eps={args.n_eps} "
        f"seed={args.seed} workers={min(args.workers, len(jobs))}",
        flush=True,
    )

    results = []
    t0 = time.time()
    with Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            results.append(result)
            if result["ok"]:
                best = int(np.argmin(result["PDR"]))
                print(
                    f"  [{i}/{len(jobs)}] {result['key']} best={rules[best]} "
                    f"PDR={result['PDR'][best]:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            else:
                print(f"  [{i}/{len(jobs)}] FAIL {result['key']}: {result['err'][:200]}", flush=True)

    failed = [r for r in results if not r["ok"]]
    if failed:
        raise RuntimeError(f"휴리스틱 적합 실패 {len(failed)}/{len(results)}개")
    if len(results) != len(jobs):
        raise RuntimeError(f"결과 수 불일치: {len(results)} != {len(jobs)}")

    prefix = Path(args.out_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    full_path = Path(str(prefix) + "_full.csv")
    best_path = Path(str(prefix) + "_best.csv")
    meta_path = Path(str(prefix) + "_meta.json")

    full_rows = []
    by_sig_rule: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    key_to_result = {r["key"]: r for r in results}
    for key, _ in items:
        match = KEY_RE.match(key)
        assert match is not None
        result = key_to_result[key]
        for i, rule in enumerate(rules):
            row = {
                "point_key": key,
                "region": match.group("region"),
                "sigcd": match.group("sigcd"),
                "point": f"p{match.group('point')}",
                "rule": rule,
                "n_fit_eps": args.n_eps,
                "fit_seed": args.seed,
                "PDR_woG": result["PDR"][i],
                "reward_woG": result["woG"][i],
            }
            full_rows.append(row)
            by_sig_rule[(match.group("sigcd"), rule)].append((result["PDR"][i], result["woG"][i]))

    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0]))
        writer.writeheader()
        writer.writerows(full_rows)

    best_rows = []
    fitted_sigcds = sorted({sigcd for sigcd, _ in by_sig_rule})
    for sigcd in fitted_sigcds:
        if not args.limit and len({k for k in manifest if f"_{sigcd}_p" in k}) != 4:
            raise RuntimeError(f"{sigcd}: random4 네 좌표가 모두 적합되지 않음")
        candidates = []
        for rule in rules:
            vals = by_sig_rule.get((sigcd, rule), [])
            if not vals:
                continue
            candidates.append(
                (float(np.mean([v[0] for v in vals])), rule, float(np.mean([v[1] for v in vals])), len(vals))
            )
        fit_pdr, best_rule, fit_wog, n_points = min(candidates, key=lambda x: (x[0], x[1]))
        region = groups[sigcd]["region"]
        best_rows.append(
            {
                "region": region,
                "sigcd": sigcd,
                "best_rule": best_rule,
                "fit_PDR_woG": fit_pdr,
                "fit_reward_woG": fit_wog,
                "n_points": n_points,
                "n_fit_eps_per_point": args.n_eps,
                "fit_seed": args.seed,
                "selection_manifest": str(manifest_path),
            }
        )
    with open(best_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)

    meta = {
        "protocol": "v10_heuristic_fit_train1000_only",
        "created_at_unix": time.time(),
        "git_sha": git_sha(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scenario_bundle_sha256": scenario_bundle_sha256(dict(items)),
        "excluded_eval_manifest": str(eval_manifest_path),
        "excluded_eval_manifest_sha256": sha256_file(eval_manifest_path),
        "coordinate_overlap": 0,
        "n_points": len(items),
        "n_districts": len(best_rows),
        "n_rules": len(rules),
        "n_fit_eps_per_point": args.n_eps,
        "fit_seed": args.seed,
        "evaluation_seed_reserved": 11000,
        "environment": {
            "MCI_CAP_GATE": "occ",
            "MCI_OBS_VARIANT": "essential+load+valid",
            "MCI_H_PAD": "47",
            "MCI_REWARD_MODE": "woG",
        },
        "outputs": {"best": str(best_path), "full": str(full_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[heur-fit] 저장 best={best_path} full={full_path} meta={meta_path}", flush=True)
    print(f"[heur-fit] wall={(time.time() - t0) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
