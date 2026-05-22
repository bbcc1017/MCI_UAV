"""학습된 RL 모델 + 휴리스틱 best 를 17개 광역 좌표에서 일괄 평가.

좌표마다:
  1) 시나리오 생성 (학습 시와 동일한 incident_size/amb_count/uav_count 필수)
  2) main.py 호출 → 휴리스틱 시뮬, results_*.txt / results_*_stat.txt 저장
  3) results_*_stat.txt 파싱 → 휴리스틱 best 추출
  4) PPO / DQN / REINFORCE 학습 모델 평가 (같은 ep 수)
  5) 결과 행 누적

마지막:
  - results/cross_location_eval.csv  (17행 × 알고리즘 metrics)
  - results/cross_location_eval.png  (mean_R bar + Δ vs heuristic 라인 차트)

학습 시 obs/action shape 와 일치해야 모델 load 가능:
  incident_size, amb_count, uav_count 가 학습 시점과 같아야 함 (기본 100 / 30 / 25).

시나리오 생성기는 MCI_ADV 의 ScenarioGenerator (AMB+UAV 혼합) 를 사용.
기본은 OSRM 모드 → 폴더 suffix 가 `_osrm` (kakao 모드면 `_dep_<YYYYMMDDHHMM>`).
"""
import argparse
import os
import subprocess
import sys

# Windows + Anaconda 에서 numpy(MKL)/torch(MKL)/matplotlib 가 libiomp5md.dll 을
# 중복 로드해 OMP Error #15 가 발생. 무해한 workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
SCE_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sce_src"))
if SCE_DIR not in sys.path:
    sys.path.insert(0, SCE_DIR)
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sim_src"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

import numpy as np
import pandas as pd

from make_csv_yaml_dynamic import ScenarioGenerator
from evaluate import eval_policy, ppo_policy, dqn_policy, reinforce_policy, make_eval_env


LOCATIONS = [
    ("서울", "서울특별시청",         37.5666, 126.9784),
    ("부산", "부산광역시청",         35.1798, 129.0750),
    ("대구", "대구광역시청(산격)",    35.8894, 128.6087),
    ("인천", "인천광역시청",         37.4563, 126.7052),
    ("광주", "광주광역시청",         35.1601, 126.8515),
    ("대전", "대전광역시청",         36.3505, 127.3845),
    ("울산", "울산광역시청",         35.5398, 129.3114),
    ("세종", "세종특별자치시청",      36.4800, 127.2890),
    ("경기", "경기도청(광교)",       37.2893, 127.0535),
    ("강원", "강원특별자치도청",      37.8845, 127.7297),
    ("충북", "충청북도청",           36.6359, 127.4913),
    ("충남", "충청남도청",           36.6588, 126.8315),
    ("전북", "전북특별자치도청",      35.8203, 127.1088),
    ("전남", "전라남도청",           34.8160, 126.4623),
    ("경북", "경상북도청",           36.5759, 128.7067),
    ("경남", "경상남도청",           35.2277, 128.6811),
    ("제주", "제주특별자치도청",      33.4890, 126.4983),
]


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ppo_path", required=True)
    p.add_argument("--dqn_path", required=True)
    p.add_argument("--reinforce_path", required=True)
    p.add_argument("--incident_size", type=int, default=100,
                   help="학습 시 사용한 값과 동일해야 함")
    p.add_argument("--amb_count", type=int, default=30,
                   help="학습 시 사용한 값과 동일해야 함")
    p.add_argument("--uav_count", type=int, default=25)
    p.add_argument("--amb_velocity", type=int, default=40)
    p.add_argument("--uav_velocity", type=int, default=80)
    p.add_argument("--amb_handover_time", type=float, default=10.0)
    p.add_argument("--uav_handover_time", type=float, default=15.0)
    p.add_argument("--n_episodes", type=int, default=1000,
                   help="좌표 한 곳당 평가 ep 수 (휴리스틱/RL 동일). main.py totalSamples 와 일치")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_path", default=".")
    p.add_argument("--out_csv", default="results/cross_location_eval.csv")
    p.add_argument("--plot_out", default="results/cross_location_eval.png")
    p.add_argument("--python", default=sys.executable,
                   help="main.py 호출용 python 인터프리터")
    p.add_argument("--plot_only", action="store_true",
                   help="평가 건너뛰고 기존 --out_csv 만 읽어 PNG 재생성")
    p.add_argument("--exp_prefix", default="crosseval_mix",
                   help="시나리오 experiment_id prefix. 폴더 suffix 는 도로 공급자에 따라 자동 결정 "
                        "(osrm → _osrm, kakao+departure_time → _dep_<ts>)")
    p.add_argument("--is_use_time", type=str2bool, default=False,
                   help="True: Kakao API duration 사용 / False: OSRM 거리·속도 기반")
    p.add_argument("--osrm_url", type=str, default=None,
                   help="OSRM 백엔드 URL. 미지정 시 ENV(MCI_OSRM_URL) 또는 공용 서버")
    p.add_argument("--kakao_api_key", type=str,
                   default=os.environ.get("KAKAO_API_KEY"),
                   help="--is_use_time True 모드 필수. 미지정 시 ENV(KAKAO_API_KEY)")
    p.add_argument("--departure_time", type=str, default=None,
                   help="kakao 모드 출발시간 (YYYYMMDDHHMM)")
    p.add_argument("--fixed_hos_num", type=int, default=None,
                   help="모든 좌표에서 hos_num 강제 고정 (학습 시 obs 차원과 동일해야 함)")
    p.add_argument("--regions", nargs="+", default=None,
                   help="평가할 지역 short_name 리스트 (기본: 17개 전체). 예: --regions 광주 강원 전남 경북")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 평가: AMB 결정을 룰에 위임 (RL 의 mode==0 출력 시 룰 결정으로 swap). "
                        "학습 시 사용한 룰과 동일해야 정합")
    p.add_argument("--skip_heuristic", action="store_true",
                   help="휴리스틱 main.py 실행을 건너뛰고 기존 stat.txt 를 그대로 사용. "
                        "--heuristic_exp_prefix 와 함께 사용")
    p.add_argument("--heuristic_exp_prefix", default=None,
                   help="--skip_heuristic 시 기존 휴리스틱 결과를 찾을 prefix. "
                        "신규 생성기와 동일한 suffix(_osrm 등) 가 자동 부여됨")
    p.add_argument("--heuristic_results_dir", default="./results",
                   help="--skip_heuristic 시 휴리스틱 stat 을 찾을 루트 디렉토리. "
                        "예: results/korea_center/heuristic_sim")
    p.add_argument("--scenario_search_dir", default=None,
                   help="--reuse_scenario 시 기존 시나리오 config 를 찾을 루트. "
                        "예: scenarios/korea_center/eval")
    p.add_argument("--reuse_scenario", action="store_true",
                   help="시나리오 생성 건너뛰고 --scenario_search_dir 아래의 기존 config_*.yaml 을 사용. "
                        "exp_id 는 --exp_prefix + --departure_time 으로 결정")
    return p.parse_args()


def _build_generator(base_path, exp_prefix, short_name, *,
                     is_use_time, osrm_url, kakao_api_key, departure_time,
                     fixed_hos_num=None):
    return ScenarioGenerator(
        base_path=base_path,
        experiment_id=f"{exp_prefix}_{short_name}",
        kakao_api_key=kakao_api_key,
        departure_time=departure_time,
        osrm_url=osrm_url,
        is_use_time=is_use_time,
        fixed_hos_num=fixed_hos_num,
    )


def gen_scenario_for_region(short_name, lat, lon, *,
                            incident_size, amb_count, uav_count,
                            amb_velocity, uav_velocity,
                            amb_handover_time, uav_handover_time,
                            total_samples, base_path, exp_prefix,
                            is_use_time, osrm_url, kakao_api_key,
                            departure_time, fixed_hos_num=None):
    gen = _build_generator(base_path, exp_prefix, short_name,
                           is_use_time=is_use_time, osrm_url=osrm_url,
                           kakao_api_key=kakao_api_key,
                           departure_time=departure_time,
                           fixed_hos_num=fixed_hos_num)
    return gen.generate_scenario(
        latitude=lat, longitude=lon,
        incident_size=incident_size, amb_count=amb_count, uav_count=uav_count,
        amb_velocity=amb_velocity, uav_velocity=uav_velocity,
        total_samples=total_samples, random_seed=0,
        is_use_time=is_use_time,
        amb_handover_time=amb_handover_time,
        uav_handover_time=uav_handover_time,
    )


def run_main_sim(config_path: str, python_exe: str, base_path: str):
    """main.py 를 subprocess 로 호출. results_*.txt / results_*_stat.txt 생성."""
    cmd = [python_exe, os.path.join("src", "sim_src", "main.py"),
           "--config_path", config_path, "--no_log"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, env=env, cwd=base_path, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_stat_file(stat_path: str, n_rules: int):
    """main.py 의 results_*_stat.txt 를 파싱하여 룰별 metrics 반환.

    파일 구조: 5 block (rew, time, pdr, rew_woG, pdr_woG) × n_rules rows.
    각 row: '<rule_name>  mean  std  ci_half'  (delimiter = 두 칸 공백)
    """
    with open(stat_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    if len(lines) != 5 * n_rules:
        raise ValueError(f"{stat_path}: expected {5*n_rules} lines, got {len(lines)}")

    block_names = ["R", "time", "PDR", "R_woG", "PDR_woG"]
    rules = []
    for i in range(n_rules):
        parts = lines[i].rsplit(None, 3)
        rules.append({"rule_name": parts[0]})

    for b_idx, b_name in enumerate(block_names):
        for i in range(n_rules):
            parts = lines[b_idx * n_rules + i].rsplit(None, 3)
            rules[i][f"{b_name}_mean"] = float(parts[1])
            rules[i][f"{b_name}_std"]  = float(parts[2])
            rules[i][f"{b_name}_ci"]   = float(parts[3])
    return rules


def heuristic_best_from_yaml_dir(config_path: str, output_path_override: str = None):
    """config yaml 의 output_path/exp_indicator 로부터 stat.txt 위치 추정 후 파싱.

    output_path_override 가 주어지면 yaml 의 output_path 대신 사용 (--skip_heuristic 시
    기존 다른 prefix 의 stat.txt 를 가져올 때 활용).
    """
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_path = output_path_override or cfg["run_setting"]["output_path"]
    exp_indicator = cfg["run_setting"]["exp_indicator"]
    # main.py 의 output_path 는 cwd 기준 상대경로
    stat_path = os.path.join(output_path, exp_indicator,
                              f"results_{exp_indicator}_stat.txt")
    # rule 개수 알아내기
    rule_info = cfg["rule_info"]
    n_rules = len(rule_info["priority_rule"]) * len(rule_info["hos_select_rule"]) \
              * len(rule_info["red_mode_rule"]) * len(rule_info["yellow_mode_rule"])
    rules = parse_stat_file(stat_path, n_rules)
    best = max(rules, key=lambda r: r["R_mean"])
    return best, rules, stat_path


def _experiment_id_for(base, exp_prefix, short_name, args):
    """ScenarioGenerator 가 부여하는 experiment_id (suffix 포함) 를 미리 계산."""
    gen = _build_generator(base, exp_prefix, short_name,
                           is_use_time=args.is_use_time, osrm_url=args.osrm_url,
                           kakao_api_key=args.kakao_api_key,
                           departure_time=args.departure_time,
                           fixed_hos_num=args.fixed_hos_num)
    return gen.experiment_id


def _find_existing_config(search_dir, exp_id):
    """search_dir/<exp_id>/(lat,lon)/config_(lat,lon).yaml 글로브로 찾기. 없으면 None."""
    import glob
    pat = os.path.join(search_dir, exp_id, "*", "config_*.yaml")
    hits = glob.glob(pat)
    if not hits:
        return None
    return hits[0]


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    if args.plot_only:
        df = pd.read_csv(args.out_csv, encoding="utf-8-sig")
        print(f"Loaded CSV: {args.out_csv}  rows={len(df)}")
        plot_results(df, args.plot_out)
        return

    if args.skip_heuristic and not args.heuristic_exp_prefix:
        raise SystemExit("--skip_heuristic 사용 시 --heuristic_exp_prefix 를 함께 지정해야 합니다 "
                         "(예: --heuristic_exp_prefix crosseval_mix)")

    if args.is_use_time and not args.kakao_api_key:
        raise SystemExit("--is_use_time True 사용 시 --kakao_api_key 가 필요합니다.")

    print("Loading trained models...")
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN
    from reinforce_agent import ReinforceAgent
    ppo_model = MaskablePPO.load(args.ppo_path)
    dqn_model = DQN.load(args.dqn_path)
    rein_agent = ReinforceAgent.load(args.reinforce_path)
    print(f"  PPO       : {args.ppo_path}")
    print(f"  DQN       : {args.dqn_path}")
    print(f"  REINFORCE : {args.reinforce_path}")

    locations = LOCATIONS if not args.regions else [l for l in LOCATIONS if l[0] in args.regions]
    if args.regions and len(locations) != len(args.regions):
        unknown = set(args.regions) - {l[0] for l in LOCATIONS}
        if unknown:
            print(f"⚠️ 알 수 없는 지역: {unknown}")
    print(f"\nEvaluating {len(locations)} locations (n_episodes={args.n_episodes}, seed={args.seed})")

    rows = []
    for i, (short_name, full_name, lat, lon) in enumerate(locations, 1):
        print(f"\n[{i:2d}/{len(locations)}] {short_name} {full_name}  ({lat},{lon})")

        config_path = None
        if args.reuse_scenario:
            if not args.scenario_search_dir:
                raise SystemExit("--reuse_scenario 사용 시 --scenario_search_dir 필요")
            exp_id = _experiment_id_for(base, args.exp_prefix, short_name, args)
            config_path = _find_existing_config(args.scenario_search_dir, exp_id)
            if config_path is None:
                print(f"  ! 기존 시나리오 미발견: {args.scenario_search_dir}/{exp_id}/*/config_*.yaml")
                continue
            print(f"  Reusing scenario: {config_path}")

        if config_path is None:
            try:
                config_path = gen_scenario_for_region(
                    short_name, lat, lon,
                    incident_size=args.incident_size,
                    amb_count=args.amb_count,
                    uav_count=args.uav_count,
                    amb_velocity=args.amb_velocity,
                    uav_velocity=args.uav_velocity,
                    amb_handover_time=args.amb_handover_time,
                    uav_handover_time=args.uav_handover_time,
                    total_samples=args.n_episodes,
                    base_path=base,
                    exp_prefix=args.exp_prefix,
                    is_use_time=args.is_use_time,
                    osrm_url=args.osrm_url,
                    kakao_api_key=args.kakao_api_key,
                    departure_time=args.departure_time,
                    fixed_hos_num=args.fixed_hos_num,
                )
            except Exception as e:
                print(f"  ! 시나리오 생성 실패: {e}")
                continue

        # 1) Heuristic stat 확보
        if args.skip_heuristic:
            print(f"  Heuristic skipped — reusing existing stat (prefix={args.heuristic_exp_prefix})")
            old_exp_id = _experiment_id_for(
                base, args.heuristic_exp_prefix, short_name, args)
            old_output_path = os.path.join(args.heuristic_results_dir, old_exp_id)
            try:
                heur_best, _, stat_path = heuristic_best_from_yaml_dir(
                    config_path, output_path_override=old_output_path)
                print(f"    loaded -> {stat_path}")
                print(f"    best  : {heur_best['rule_name']}  R={heur_best['R_mean']:.3f}  "
                      f"R_woG={heur_best['R_woG_mean']:.3f}  "
                      f"PDR={heur_best['PDR_mean']:.3f}  time={heur_best['time_mean']:.0f}")
            except Exception as e:
                print(f"  ! 기존 휴리스틱 stat 로드 실패: {e}")
                continue
        else:
            print("  Heuristic simulation (main.py) ...")
            try:
                run_main_sim(config_path, args.python, base)
                heur_best, _, stat_path = heuristic_best_from_yaml_dir(config_path)
                print(f"    saved -> {stat_path}")
                print(f"    best  : {heur_best['rule_name']}  R={heur_best['R_mean']:.3f}  "
                      f"R_woG={heur_best['R_woG_mean']:.3f}  "
                      f"PDR={heur_best['PDR_mean']:.3f}  time={heur_best['time_mean']:.0f}")
            except Exception as e:
                print(f"  ! main.py 실패: {e}")
                continue

        # 2) RL 평가 (in-memory)
        rl_env_factory = make_eval_env(config_path)

        def _do_eval(name, model, kind):
            if args.hybrid_amb_rule:
                from hybrid_eval import hybrid_eval
                from RuleManager import Universal_Rule
                rule_for_eval = Universal_Rule(*args.hybrid_amb_rule)
                m = hybrid_eval(rl_env_factory, model, kind, rule_for_eval,
                                args.n_episodes, args.seed, mode_split="strict")
                tag = f"{name}[hybrid]"
            else:
                if kind == "ppo":
                    policy = ppo_policy(model)
                elif kind == "dqn":
                    policy = dqn_policy(model)
                else:
                    policy = reinforce_policy(model)
                m = eval_policy(rl_env_factory, policy, args.n_episodes, args.seed)
                tag = name
            print(f"    {tag:<18} R={m['mean_R']:.3f}  R_woG={m['mean_R_woG']:.3f}  "
                  f"PDR={m['mean_PDR']:.3f}  PDR_woG={m['mean_PDR_woG']:.3f}  "
                  f"time={m['mean_time']:.0f}")
            return m

        m_ppo  = _do_eval("PPO",       ppo_model,  "ppo")
        m_dqn  = _do_eval("DQN",       dqn_model,  "dqn")
        m_rein = _do_eval("REINFORCE", rein_agent, "reinforce")

        rows.append({
            "region": short_name, "name": full_name, "lat": lat, "lon": lon,
            "heuristic_rule":      heur_best["rule_name"],
            "heuristic_R":         heur_best["R_mean"],
            "heuristic_R_woG":     heur_best["R_woG_mean"],
            "heuristic_PDR":       heur_best["PDR_mean"],
            "heuristic_PDR_woG":   heur_best["PDR_woG_mean"],
            "heuristic_time":      heur_best["time_mean"],
            "PPO_R": m_ppo["mean_R"], "PPO_R_woG": m_ppo["mean_R_woG"],
            "PPO_PDR": m_ppo["mean_PDR"], "PPO_PDR_woG": m_ppo["mean_PDR_woG"],
            "PPO_time": m_ppo["mean_time"],
            "DQN_R": m_dqn["mean_R"], "DQN_R_woG": m_dqn["mean_R_woG"],
            "DQN_PDR": m_dqn["mean_PDR"], "DQN_PDR_woG": m_dqn["mean_PDR_woG"],
            "DQN_time": m_dqn["mean_time"],
            "REINFORCE_R": m_rein["mean_R"], "REINFORCE_R_woG": m_rein["mean_R_woG"],
            "REINFORCE_PDR": m_rein["mean_PDR"], "REINFORCE_PDR_woG": m_rein["mean_PDR_woG"],
            "REINFORCE_time": m_rein["mean_time"],
            "PPO_vs_heur":           m_ppo["mean_R"]      - heur_best["R_mean"],
            "DQN_vs_heur":           m_dqn["mean_R"]      - heur_best["R_mean"],
            "REINFORCE_vs_heur":     m_rein["mean_R"]     - heur_best["R_mean"],
            "PPO_woG_vs_heur":       m_ppo["mean_R_woG"]  - heur_best["R_woG_mean"],
            "DQN_woG_vs_heur":       m_dqn["mean_R_woG"]  - heur_best["R_woG_mean"],
            "REINFORCE_woG_vs_heur": m_rein["mean_R_woG"] - heur_best["R_woG_mean"],
            "stat_path": stat_path,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved CSV: {out_csv}")

    if len(df) == 0:
        return

    summary_cols = ["region", "heuristic_R", "PPO_R", "DQN_R", "REINFORCE_R",
                    "PPO_vs_heur", "DQN_vs_heur", "REINFORCE_vs_heur"]
    print("\n=== mean reward 요약 (R) ===")
    print(df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    summary_woG = ["region", "heuristic_R_woG", "PPO_R_woG", "DQN_R_woG", "REINFORCE_R_woG",
                   "PPO_woG_vs_heur", "DQN_woG_vs_heur", "REINFORCE_woG_vs_heur"]
    print("\n=== mean reward 요약 (R_woG, Green 제외) ===")
    print(df[summary_woG].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print("\n=== 휴리스틱 best 를 이긴 횟수 ===")
    print(f"  PPO       : R={(df['PPO_vs_heur']       > 0).sum():2d}/{len(df)}  "
          f"R_woG={(df['PPO_woG_vs_heur']       > 0).sum():2d}/{len(df)}")
    print(f"  DQN       : R={(df['DQN_vs_heur']       > 0).sum():2d}/{len(df)}  "
          f"R_woG={(df['DQN_woG_vs_heur']       > 0).sum():2d}/{len(df)}")
    print(f"  REINFORCE : R={(df['REINFORCE_vs_heur'] > 0).sum():2d}/{len(df)}  "
          f"R_woG={(df['REINFORCE_woG_vs_heur'] > 0).sum():2d}/{len(df)}")

    plot_results(df, args.plot_out)


def plot_results(df, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 한글 폰트 — 플랫폼별 후보 중 설치된 것을 선택 (Linux 서버: Noto Sans CJK KR)
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _kf in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic",
                "Noto Sans CJK JP", "AppleGothic"):
        if _kf in _installed:
            plt.rcParams["font.family"] = _kf
            break
    plt.rcParams["axes.unicode_minus"] = False

    regions = df["region"].tolist()
    x = np.arange(len(regions))
    width = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    def _bar(ax, col_h, col_p, col_d, col_r, ylabel, title):
        ax.bar(x - 1.5*width, df[col_h], width, label="Heuristic best", color="#888")
        ax.bar(x - 0.5*width, df[col_p], width, label="PPO",            color="#1f77b4")
        ax.bar(x + 0.5*width, df[col_d], width, label="DQN",            color="#ff7f0e")
        ax.bar(x + 1.5*width, df[col_r], width, label="REINFORCE",      color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="lower right", ncol=4); ax.grid(axis="y", alpha=0.3)

    def _delta(ax, col_p, col_d, col_r, ylabel, title):
        ax.axhline(0, color="#888", linewidth=1)
        ax.plot(x, df[col_p], "o-", label="PPO - Heur",       color="#1f77b4")
        ax.plot(x, df[col_d], "s-", label="DQN - Heur",       color="#ff7f0e")
        ax.plot(x, df[col_r], "^-", label="REINFORCE - Heur", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(regions, rotation=0)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="best"); ax.grid(axis="y", alpha=0.3)

    _bar(axes[0,0], "heuristic_R", "PPO_R", "DQN_R", "REINFORCE_R",
         "mean reward", "Mean reward (R, Green 포함)")
    _bar(axes[0,1], "heuristic_R_woG", "PPO_R_woG", "DQN_R_woG", "REINFORCE_R_woG",
         "mean reward woG", "Mean reward (R_woG, Green 제외)")
    _delta(axes[1,0], "PPO_vs_heur", "DQN_vs_heur", "REINFORCE_vs_heur",
           "Δ vs heuristic (R)", "RL 우위 폭 (R)")
    _delta(axes[1,1], "PPO_woG_vs_heur", "DQN_woG_vs_heur", "REINFORCE_woG_vs_heur",
           "Δ vs heuristic (R_woG)", "RL 우위 폭 (R_woG)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
