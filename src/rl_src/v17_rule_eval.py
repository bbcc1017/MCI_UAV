# -*- coding: utf-8 -*-
"""v17 규칙 기준선 closed-loop 재평가 — 트리·PPO 와 완전히 동일한 배관.

``v10_tree_eval.py`` 의 rollout·seed·CSV 규약을 그대로 쓰고 정책만 규칙으로 바꾼다.
기존 v17 전수평가 산출물(2026-08-12 04:48 HEUR, 13:19 LB)은 sim 정정 커밋
``b01efd3``(같은 날 15:04, 병원 후보 순회를 수단별 실제 ETA 순으로 정정) **이전** 이라
정정 이후 코드로 돌린 트리·PPO 와 직접 비교할 수 없다.

규칙 **선정**(전국 단일 조합)은 정정 이전 train1000 전수평가 결과를 그대로 승계하고,
여기서는 대표점250 에서의 **측정만** 다시 한다. 재선정에는 train1000 전수 재실행이
필요하므로 이번 주 범위 밖이며, 승계 사실을 각주로 남긴다.

정책 스펙: ``이름=cap3:<규칙>`` | ``이름=heur:<규칙>`` | ``이름=agn`` | ``all_heur64``
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


def build_rule_policies(specs):
    """(이름, 정책fn) 목록. 규칙 이름은 sim_src.RuleManager 의 정본 문자열."""
    from distill_policy import make_heuristic_policy
    from lb3_policy import make_agnostic_lb_policy
    from loadbalance_heuristic import make_cap_policy
    from fit_v10_heuristic_rules import all_rule_names
    from v17_field_rules import make_field_card_policy

    out = []
    for spec in specs:
        if spec == "all_heur64":
            for rule in all_rule_names():
                out.append((f"HEUR64|{rule}", make_heuristic_policy(rule)))
            continue
        name, body = spec.split("=", 1)
        if body.startswith("card:"):
            # card:lam,red_km,yhold[,dist_mode[,load_term]]
            # 4·5번째 토큰 미지정 시 기본값 = 구 동작 비트동일.
            toks = body[5:].split(",")
            lam, rkm, yh = (float(x) for x in toks[:3])
            dm = toks[3] if len(toks) > 3 else "raw"
            lt = toks[4] if len(toks) > 4 else "load"
            out.append((name, make_field_card_policy(lam, rkm, yh, dist_mode=dm, load_term=lt)))
            continue
        if body == "agn":
            out.append((name, make_agnostic_lb_policy(T=3)))
        else:
            kind, rule = body.split(":", 1)
            if rule not in all_rule_names():
                raise ValueError(f"미지 규칙: {rule}")
            out.append((name, make_cap_policy(rule, 3) if kind == "cap3"
                        else make_heuristic_policy(rule)))
    return out


def worker(job):
    region, cfg, specs, n_eps, seed0 = job
    try:
        os.environ.update(
            MCI_CAP_GATE="occ",
            MCI_OBS_VARIANT="essential+load+valid",
            MCI_H_PAD="47",
            MCI_REWARD_MODE="woG",
        )
        from viper_distill import _suppress_stdout, make_feature_env

        rows = []
        with _suppress_stdout():
            policies = build_rule_policies(specs)
            factory = make_feature_env(cfg, None)
            for ep in range(n_eps):
                seed = seed0 + ep
                for name, policy in policies:
                    reward, pdr, sim_time, n_dec, ms = rollout(factory, policy, seed)
                    rows.append({
                        "region": region, "policy": name, "info_level": "RULE",
                        "complexity": "-", "episode": ep, "seed": seed,
                        "reward_woG": reward, "pdr_woG": pdr, "sim_time": sim_time,
                        "n_decisions": n_dec, "ms_per_decision": ms,
                    })
        return {"ok": True, "region": region, "rows": rows}
    except Exception as exc:
        import traceback

        return {"ok": False, "region": region, "err": (str(exc) + traceback.format_exc())[:1500]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(EVAL_MANIFEST))
    p.add_argument("--policies", required=True, help="세미콜론(;) 구분 스펙 — 규칙명에 쉼표가 들어감")
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
    specs = [x for x in args.policies.split(";") if x]
    sys.path.insert(0, str(REPO / "src/sim_src"))
    cases = [n for n, _ in build_rule_policies(specs)]
    model_dir = ";".join(specs)

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
        (key, manifest[key], specs, args.n_eps, args.seed0)
        for key in keys if key not in done_regions
    ]
    print(
        f"[rule-eval] regions={len(keys)} remaining={len(jobs)} cases={len(cases)} "
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
        "policy_specs": specs,
        "selection_inherited_from": "results/scoreboard/v17/lbt3_common30_scoreboard_selection_audit.json",
        "note": "sim 정정 b01efd3 이후 재측정. 규칙 선정은 정정 이전 train1000 결과 승계",
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
    print(f"[rule-eval] 완료 rows={len(rows)} wall={(time.time()-t0)/60:.1f}분 → {out}", flush=True)


if __name__ == "__main__":
    main()
