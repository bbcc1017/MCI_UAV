#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shin 휴리스틱 16개를 실제 시뮬레이터에서 한 episode씩 검증한다.

검증 범위:
  1) 기존 기본 RuleManager가 여전히 64개인지
  2) opt-in 설정에서 64+16=80개인지
  3) 16개 정책의 모든 이송 행동이 joint action mask를 만족하는지
  4) 각 episode가 truncation 없이 종료되고 PDR_woG가 유한한지
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
SIM_SRC = REPO / "src/sim_src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from MCIEnvironment_gymnasium import MCIEnvironment_gym  # noqa: E402
from RuleManager import RuleManager  # noqa: E402
from ScenarioManager import ScenarioManager  # noqa: E402
from ShinHeuristics import SHIN_METHODS, SHIN_MODE_RULES, ShinHeuristicRule  # noqa: E402


DEFAULT_MANIFEST = (
    REPO / "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
)


def default_config() -> Path:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    entry = next(iter(manifest.values()))
    path = entry["path"] if isinstance(entry, dict) else entry
    return Path(path)


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg, seed):
    rng = np.random.default_rng(seed)
    scenario = ScenarioManager(cfg, rng=rng).scenario
    env = MCIEnvironment_gym(
        scenario=scenario,
        rng=rng,
        rule_test=True,
        eval_mode=True,
    )
    return scenario, env


def encode(action, h):
    c, d, m = map(int, action)
    return c * ((h + 1) * 2) + d * 2 + m


def run_one(cfg, method, mode_rule, seed, max_steps):
    scenario, env = make_env(cfg, seed)
    rule = ShinHeuristicRule(method, mode_rule)
    rule.set_seed(np.random.default_rng(seed))
    rule.init_with_scenario(scenario)
    obs, _ = env.reset(seed=seed)
    reward_wog = 0.0
    actions = 0

    terminated = truncated = False
    info = {"time": 0.0}
    while not (terminated or truncated):
        action = rule.select(obs)
        c, d, m = map(int, action)
        if d == 0:
            if action != [-1, 0, -1]:
                raise AssertionError(f"{rule.rule_name}: 비표준 STAY {action}")
        else:
            if c not in (0, 1) or m not in (0, 1) or not (1 <= d <= env.H):
                raise AssertionError(f"{rule.rule_name}: 행동 범위 오류 {action}")
            mask = env.action_masks_joint()
            flat = encode(action, env.H)
            if flat >= len(mask) or not mask[flat]:
                raise AssertionError(
                    f"{rule.rule_name}: mask 위반 action={action}, meta={rule.last_meta}"
                )
        obs, _, terminated, truncated, info = env.step(action)
        reward_wog += float(info.get("r_woG", 0.0))
        actions += 1
        if actions > max_steps:
            raise AssertionError(f"{rule.rule_name}: smoke step>{max_steps}")

    if truncated:
        raise AssertionError(f"{rule.rule_name}: episode truncated")
    pdr_wog = (
        1.0 - reward_wog / env.preventable_woG
        if env.preventable_woG > 0
        else 0.0
    )
    if not np.isfinite(pdr_wog):
        raise AssertionError(f"{rule.rule_name}: PDR_woG 비유한 {pdr_wog}")
    return {
        "rule": rule.rule_name,
        "steps": actions,
        "time": float(info["time"]),
        "pdr_wog": float(pdr_wog),
    }


def exercise_formula_branches(cfg, seed):
    """R/Y 동시 대기 상태를 합성해 네 핵심 선택식을 직접 실행한다.

    실제 episode에서는 구조·자원 여건에 따라 한 중증도만 대기한 채 의사결정이
    반복될 수 있다. 그 경우 episode smoke만으로는 Threshold/2Step/Integrated
    고유 분기가 실행됐다고 보장할 수 없으므로, 실제 시나리오의 병원·차량
    파라미터를 유지한 상태에서 대기열과 병원 부하만 통제한다.
    """
    scenario, env = make_env(cfg, seed)
    obs, _ = env.reset(seed=seed)
    obs = copy.deepcopy(obs)

    # 두 등급이 동시에 한 명 이상 대기하고 모든 병원이 비어 있는 결정시점.
    obs["p_wait"][0][0] = [0]
    obs["p_wait"][1][0] = [1]
    obs["h_states"][:, 1:] = 0
    obs["p_sent"][:] = 0
    if len(obs["amb_states"]):
        obs["amb_states"][:] = 0
    if len(obs["uav_states"]):
        obs["uav_states"][:] = 0

    props = scenario["EntityManager"].en_properties
    amb_num = int(props["ambulance"].get("amb_num", 0))
    uav_num = int(props["uav"].get("uav_num", 0))
    if amb_num > 0:
        mode = 0
        mode_rule = "OnlyAMB"
        obs["amb_wait"][0] = [0]
        obs["uav_wait"][0] = []
    elif uav_num > 0:
        mode = 1
        mode_rule = "OnlyUAV"
        obs["uav_wait"][0] = [0]
        obs["amb_wait"][0] = []
    else:
        raise AssertionError("수식 분기 smoke에 사용할 AMB/UAV가 없음")

    required_meta = {
        "Threshold": "threshold",
        "2Step": "two_step_scores",
        "PIH": "pih_log_scores",
        "Integrated": "integrated_region",
    }
    diagnostics = {}
    for method, meta_key in required_meta.items():
        rule = ShinHeuristicRule(method, mode_rule)
        rule.set_seed(np.random.default_rng(seed))
        rule.init_with_scenario(scenario)
        action = rule.select(copy.deepcopy(obs))
        c, d, selected_mode = map(int, action)
        if d <= 0 or selected_mode != mode:
            raise AssertionError(
                f"{rule.rule_name}: 수식 분기에서 행동 생성 실패 "
                f"action={action}, meta={rule.last_meta}"
            )
        h = d - 1
        if h not in rule._eligible(obs, mode, c):
            raise AssertionError(
                f"{rule.rule_name}: 합성 상태 제약 위반 action={action}"
            )
        if meta_key not in rule.last_meta:
            raise AssertionError(
                f"{rule.rule_name}: 핵심 분기 미실행 "
                f"expected={meta_key}, meta={rule.last_meta}"
            )
        diagnostics[method] = {
            "action": action,
            meta_key: rule.last_meta[meta_key],
        }
    return diagnostics


def exercise_mode_branches(cfg, seed):
    """4개 공통 mode 규칙의 우선·대체·단독 운용을 통제 상태에서 검증한다."""
    scenario, env = make_env(cfg, seed)
    obs, _ = env.reset(seed=seed)
    obs = copy.deepcopy(obs)
    obs["p_wait"][0][0] = []
    obs["p_wait"][1][0] = [1]
    obs["h_states"][:, 1:] = 0
    obs["p_sent"][:] = 0
    if len(obs["amb_states"]):
        obs["amb_states"][:] = 0
    if len(obs["uav_states"]):
        obs["uav_states"][:] = 0

    probe = ShinHeuristicRule("PIH", "Both_AMBFirst")
    probe.init_with_scenario(scenario)
    if probe.fleet_size[0] <= 0 or probe.fleet_size[1] <= 0:
        raise AssertionError("mode smoke는 AMB와 UAV가 모두 있는 시나리오가 필요함")

    cases = [
        ("OnlyAMB", True, True, 0),
        ("OnlyUAV", True, True, 1),
        ("Both_AMBFirst", True, True, 0),
        ("Both_UAVFirst", True, True, 1),
        ("Both_AMBFirst", False, True, 1),
        ("Both_UAVFirst", True, False, 0),
        ("OnlyAMB", False, True, None),
        ("OnlyUAV", True, False, None),
    ]
    diagnostics = []
    for mode_rule, amb_ready, uav_ready, expected_mode in cases:
        case_obs = copy.deepcopy(obs)
        case_obs["amb_wait"][0] = [0] if amb_ready else []
        case_obs["uav_wait"][0] = [0] if uav_ready else []
        rule = ShinHeuristicRule("PIH", mode_rule)
        rule.set_seed(np.random.default_rng(seed))
        rule.init_with_scenario(scenario)
        action = rule.select(case_obs)
        if expected_mode is None:
            if action != [-1, 0, -1]:
                raise AssertionError(
                    f"{mode_rule}: 단독 mode가 대체수단을 사용함 action={action}"
                )
        else:
            c, d, selected_mode = map(int, action)
            if d <= 0 or selected_mode != expected_mode:
                raise AssertionError(
                    f"{mode_rule}: mode 우선순위 오류 expected={expected_mode}, "
                    f"action={action}, meta={rule.last_meta}"
                )
            if d - 1 not in rule._eligible(case_obs, selected_mode, c):
                raise AssertionError(f"{mode_rule}: mode smoke 목적지 제약 위반 {action}")
        diagnostics.append(
            {
                "rule": mode_rule,
                "ready": f"A{int(amb_ready)}U{int(uav_ready)}",
                "action": action,
            }
        )
    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, default=default_config())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument(
        "--verbose_sim",
        action="store_true",
        help="시뮬레이터의 이벤트별 stdout을 그대로 표시",
    )
    args = parser.parse_args()
    cfg = load_config(args.config_path)

    # RuleManager 개수 회귀: 기존 시나리오에는 새 키가 없어도 64개여야 한다.
    scenario, _ = make_env(cfg, args.seed)
    base_info = dict(cfg["rule_info"])
    base_info.pop("include_shin", None)
    base_info.pop("include_standard", None)
    base_info.pop("shin_methods", None)
    base_info.pop("shin_mode_rules", None)
    base_rules = RuleManager(
        base_info, scenario=scenario, rng=np.random.default_rng(args.seed)
    ).rules
    expected_legacy_names = [
        f"{priority}, {hospital}, Red {red_mode}, Yellow {yellow_mode}"
        for priority in ("START", "ReSTART")
        for hospital in ("RedOnly", "YellowNearest")
        for red_mode in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB")
        for yellow_mode in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB")
    ]
    if [rule.rule_name for rule in base_rules] != expected_legacy_names:
        raise AssertionError("기존 Full64 개수·순서·이름 회귀 실패")

    shin_info = dict(base_info)
    shin_info.update(
        {
            "include_standard": True,
            "include_shin": True,
            "shin_methods": list(SHIN_METHODS),
            "shin_mode_rules": list(SHIN_MODE_RULES),
        }
    )
    all_rules = RuleManager(
        shin_info, scenario=scenario, rng=np.random.default_rng(args.seed)
    ).rules
    expected_shin_names = [
        f"Shin {method}, Mode {mode}"
        for method in SHIN_METHODS
        for mode in SHIN_MODE_RULES
    ]
    if (
        [rule.rule_name for rule in all_rules[:64]] != expected_legacy_names
        or [rule.rule_name for rule in all_rules[64:]] != expected_shin_names
    ):
        raise AssertionError("Full64+Shin16 개수·순서·이름 배선 실패")
    shin_only_info = dict(shin_info)
    shin_only_info["include_standard"] = False
    shin_only_rules = RuleManager(
        shin_only_info, scenario=scenario, rng=np.random.default_rng(args.seed)
    ).rules
    if [rule.rule_name for rule in shin_only_rules] != expected_shin_names:
        raise AssertionError("Shin16 단독 실행 배선 실패")

    results = []
    sink = (
        contextlib.nullcontext()
        if args.verbose_sim
        else contextlib.redirect_stdout(io.StringIO())
    )
    with sink:
        formula_diagnostics = exercise_formula_branches(cfg, args.seed)
        mode_diagnostics = exercise_mode_branches(cfg, args.seed)
        for method in SHIN_METHODS:
            for mode_rule in SHIN_MODE_RULES:
                results.append(
                    run_one(cfg, method, mode_rule, args.seed, args.max_steps)
                )

    print(f"config={args.config_path}")
    print("rule_count: legacy=64, shin_only=16, with_shin=80")
    print("formula_branches:")
    for method, diagnostic in formula_diagnostics.items():
        print(f"  {method:10s} {diagnostic}")
    print("mode_branches:")
    for diagnostic in mode_diagnostics:
        print(f"  {diagnostic}")
    print(f"{'rule':43s} {'steps':>6s} {'time':>9s} {'PDR_woG':>10s}")
    for row in results:
        print(
            f"{row['rule']:43s} {row['steps']:6d} "
            f"{row['time']:9.2f} {row['pdr_wog']:10.6f}"
        )
    print("SMOKE_OK: Shin16 mask-valid, finite, terminated")


if __name__ == "__main__":
    main()
