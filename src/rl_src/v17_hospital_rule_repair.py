"""v17 LB 병원규칙 복원 부분 재실험 준비·감사 도구.

기존 v17 산출물은 보존한다. 새 결과 폴더에는 다음만 다시 계산하도록 체크포인트를
준비한다.

* LB3: RedOnly 32개 슬라이스만 미완료로 표시
* T4: 좌표별 HEUR Best가 RedOnly인 좌표만 미완료

HEUR64·Shin aligned·LB AGNOSTIC·YellowNearest LB/T4는 동결 산출물을 재사용한다.
재사용 파일은 source와 별개 경로에 하드링크하거나 새 source bundle로 원자 복사하며,
``repair_meta.json``에 원본 lineage를 기록한다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from v10_full_baselines import (
    EVAL_MANIFEST,
    REPO,
    TRAIN_MANIFEST,
    all_rule_names,
    atomic_savez,
    sha256_file,
    valid_heur_checkpoint,
    valid_t4_checkpoint,
    validate_inputs,
    work_path as v10_work_path,
)
import v16_baseline_alignment as v16


OLD_HEUR = REPO / "results/scoreboard/v17/heur64_eta_aligned_full1000"
OLD_LB = REPO / "results/scoreboard/v17/lb3_shin_eta_aligned_full1000"
NEW_HEUR = REPO / "results/scoreboard/v17/heur64_t4_hospital_rule_fix_full1000"
NEW_LB = REPO / "results/scoreboard/v17/lb3_shin_hospital_rule_fix_full1000"


def _load_meta(path: Path) -> dict:
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise RuntimeError(f"완료되지 않은 원본 meta: {path}")
    return meta


def _link_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    os.link(source, target)


def _copy_checkpoint_with_bundle(source: Path, target: Path, bundle: str) -> None:
    with np.load(source, allow_pickle=False) as data:
        v16.save_checkpoint(
            target,
            np.asarray(data["values"]),
            np.asarray(data["done"]),
            np.asarray(data["seeds"]),
            data["policy_names"].tolist(),
            bundle,
        )


def prepare(n_eps: int, seed: int) -> dict:
    entries = validate_inputs(TRAIN_MANIFEST, EVAL_MANIFEST, n_eps, True)
    rules = all_rule_names()
    red_indices = np.asarray(
        [idx + 1 for idx, rule in enumerate(rules) if ", RedOnly," in rule], dtype=int
    )
    if len(red_indices) != 32:
        raise AssertionError(f"RedOnly LB 슬라이스가 32개가 아님: {len(red_indices)}")

    old_hmeta = _load_meta(OLD_HEUR / "protocol_meta.json")
    old_lmeta = _load_meta(OLD_LB / "protocol_meta.json")
    old_lbundle = str(old_lmeta["source_bundle_sha256"])
    new_lbundle = v16.source_bundle_sha256(v16.source_hashes())
    lb_names = v16.lb_policy_names()
    shin_names = v16.shin_aligned_names()

    counts = {
        "heur_linked": 0,
        "t4_reused_yellownearest": 0,
        "t4_redonly_to_run": 0,
        "lb_prepared": 0,
        "lb_already_complete": 0,
        "shin_reused": 0,
    }
    for entry in entries:
        # HEUR64는 LB 코드 변경과 무관하므로 원시 체크포인트를 그대로 재사용한다.
        old_hp = v10_work_path(OLD_HEUR, "heur", entry)
        if not valid_heur_checkpoint(old_hp, n_eps, seed, rules):
            raise RuntimeError(f"원본 HEUR 불완전: {old_hp}")
        new_hp = v10_work_path(NEW_HEUR, "heur", entry)
        _link_once(old_hp, new_hp)
        counts["heur_linked"] += 1

        # T4는 RedOnly base 좌표만 다시 계산한다. YellowNearest는 현재 로직과 의미가 같다.
        old_tp = v10_work_path(OLD_HEUR, "t4", entry)
        if not valid_t4_checkpoint(old_tp, n_eps, seed):
            raise RuntimeError(f"원본 T4 불완전: {old_tp}")
        with np.load(old_tp, allow_pickle=False) as data:
            best_rule = str(data["best_rule"].item())
        new_tp = v10_work_path(NEW_HEUR, "t4", entry)
        if ", YellowNearest," in best_rule:
            _link_once(old_tp, new_tp)
            counts["t4_reused_yellownearest"] += 1
        elif ", RedOnly," in best_rule:
            # 재개 중 다시 prepare해도 이미 새로 계산한 파일은 보존한다.
            if new_tp.exists() and os.path.samefile(old_tp, new_tp):
                new_tp.unlink()
            if not valid_t4_checkpoint(new_tp, n_eps, seed):
                counts["t4_redonly_to_run"] += 1
        else:
            raise RuntimeError(f"알 수 없는 T4 base rule: {best_rule}")

        # LB는 기존 값을 복사하되 RedOnly 32개만 NaN/미완료로 만들어 부분 재개한다.
        old_lp = v16.work_path(OLD_LB, "lb3", entry)
        if not v16.checkpoint_valid(old_lp, lb_names, n_eps, seed, old_lbundle):
            raise RuntimeError(f"원본 LB 불완전: {old_lp}")
        new_lp = v16.work_path(NEW_LB, "lb3", entry)
        if v16.checkpoint_valid(new_lp, lb_names, n_eps, seed, new_lbundle):
            counts["lb_already_complete"] += 1
        elif not new_lp.exists():
            with np.load(old_lp, allow_pickle=False) as data:
                values = np.asarray(data["values"]).copy()
                done = np.asarray(data["done"]).copy()
                seeds = np.asarray(data["seeds"]).copy()
            values[red_indices] = np.nan
            done[red_indices] = False
            v16.save_checkpoint(new_lp, values, done, seeds, lb_names, new_lbundle)
            counts["lb_prepared"] += 1

        # Shin은 loadbalance 코드와 독립이다. 값은 그대로 두고 새 폴더의 lineage로 복사한다.
        old_sp = v16.work_path(OLD_LB, "shin_align", entry)
        if not v16.checkpoint_valid(old_sp, shin_names, n_eps, seed, old_lbundle):
            raise RuntimeError(f"원본 Shin 불완전: {old_sp}")
        new_sp = v16.work_path(NEW_LB, "shin_align", entry)
        if not v16.checkpoint_valid(new_sp, shin_names, n_eps, seed, new_lbundle):
            _copy_checkpoint_with_bundle(old_sp, new_sp, new_lbundle)
        counts["shin_reused"] += 1

    repair_meta = {
        "protocol": "v17_hospital_rule_repair",
        "status": "prepared",
        "prepared_at_unix": time.time(),
        "n_coordinates": len(entries),
        "n_episodes": n_eps,
        "seed_start": seed,
        "seed_end": seed + n_eps - 1,
        "old_heur_dir": str(OLD_HEUR),
        "old_lb_dir": str(OLD_LB),
        "new_heur_dir": str(NEW_HEUR),
        "new_lb_dir": str(NEW_LB),
        "old_heur_source_bundle": old_hmeta.get("source_bundle_sha256"),
        "old_lb_source_bundle": old_lbundle,
        "new_lb_source_bundle": new_lbundle,
        "loadbalance_sha256": sha256_file(REPO / "src/rl_src/loadbalance_heuristic.py"),
        "recomputed": {
            "lb_redonly_rule_count": 32,
            "lb_episode_count": len(entries) * 32 * n_eps,
            "t4_redonly_coordinate_count": counts["t4_redonly_to_run"],
            "t4_episode_count": counts["t4_redonly_to_run"] * n_eps,
        },
        "reused": {
            "heur64_all": True,
            "lb_agnostic": True,
            "lb_yellownearest_rule_count": 32,
            "shin_aligned_all": True,
            "t4_yellownearest_coordinate_count": counts["t4_reused_yellownearest"],
        },
        "counts": counts,
    }
    for out in (NEW_HEUR, NEW_LB):
        out.mkdir(parents=True, exist_ok=True)
        (out / "repair_meta.json").write_text(
            json.dumps(repair_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return repair_meta


def audit(n_eps: int, seed: int) -> dict:
    entries = validate_inputs(TRAIN_MANIFEST, EVAL_MANIFEST, n_eps, True)
    rules = all_rule_names()
    red_indices = np.asarray([idx + 1 for idx, r in enumerate(rules) if ", RedOnly," in r])
    yellow_indices = np.asarray([idx + 1 for idx, r in enumerate(rules) if ", YellowNearest," in r])
    old_lmeta = _load_meta(OLD_LB / "protocol_meta.json")
    old_bundle = str(old_lmeta["source_bundle_sha256"])
    new_bundle = v16.source_bundle_sha256(v16.source_hashes())
    lb_names = v16.lb_policy_names()
    shin_names = v16.shin_aligned_names()
    errors = []
    red_changed = 0
    yellow_drift = 0
    agnostic_drift = 0
    t4_red_changed = 0
    t4_yellow_drift = 0

    for entry in entries:
        old_lp = v16.work_path(OLD_LB, "lb3", entry)
        new_lp = v16.work_path(NEW_LB, "lb3", entry)
        if not v16.checkpoint_valid(new_lp, lb_names, n_eps, seed, new_bundle):
            errors.append(f"LB 불완전 {entry['dataset']}:{entry['key']}")
            continue
        new_sp = v16.work_path(NEW_LB, "shin_align", entry)
        if not v16.checkpoint_valid(new_sp, shin_names, n_eps, seed, new_bundle):
            errors.append(f"Shin 불완전 {entry['dataset']}:{entry['key']}")
        with np.load(old_lp, allow_pickle=False) as old, np.load(new_lp, allow_pickle=False) as new:
            ov = np.asarray(old["values"])
            nv = np.asarray(new["values"])
        agnostic_drift += int(not np.array_equal(ov[0], nv[0]))
        yellow_drift += sum(not np.array_equal(ov[i], nv[i]) for i in yellow_indices)
        red_changed += sum(not np.array_equal(ov[i], nv[i]) for i in red_indices)

        old_tp = v10_work_path(OLD_HEUR, "t4", entry)
        new_tp = v10_work_path(NEW_HEUR, "t4", entry)
        if not valid_t4_checkpoint(new_tp, n_eps, seed):
            errors.append(f"T4 불완전 {entry['dataset']}:{entry['key']}")
            continue
        with np.load(old_tp, allow_pickle=False) as old, np.load(new_tp, allow_pickle=False) as new:
            best_rule = str(old["best_rule"].item())
            same = np.array_equal(old["values"], new["values"])
        if ", RedOnly," in best_rule:
            t4_red_changed += int(not same)
        else:
            t4_yellow_drift += int(not same)

    if agnostic_drift:
        errors.append(f"AGNOSTIC drift={agnostic_drift}")
    if yellow_drift:
        errors.append(f"YellowNearest LB drift={yellow_drift}")
    if t4_yellow_drift:
        errors.append(f"YellowNearest T4 drift={t4_yellow_drift}")
    if red_changed == 0:
        errors.append("수정된 RedOnly LB가 한 건도 변하지 않음")
    if t4_red_changed == 0:
        errors.append("수정된 RedOnly T4가 한 건도 변하지 않음")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "lb_redonly_changed_coordinate_rules": red_changed,
        "lb_yellownearest_drift": yellow_drift,
        "lb_agnostic_drift": agnostic_drift,
        "t4_redonly_changed_coordinates": t4_red_changed,
        "t4_yellownearest_drift": t4_yellow_drift,
        "old_bundle": old_bundle,
        "new_bundle": new_bundle,
    }
    for out in (NEW_HEUR, NEW_LB):
        (out / "repair_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("prepare", "audit"), required=True)
    p.add_argument("--n_eps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    result = prepare(args.n_eps, args.seed) if args.phase == "prepare" else audit(args.n_eps, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
