"""학습된 RL 모델 + 휴리스틱 best 를 17개 광역 좌표에서 일괄 평가.

좌표마다:
  1) 시나리오 생성 (학습 시와 동일한 incident_size/uav_count 필수)
  2) main.py 호출 → 휴리스틱 1000ep 시뮬, results_*.txt / results_*_stat.txt 저장
  3) results_*_stat.txt 파싱 → 휴리스틱 best 추출
  4) PPO / DQN / REINFORCE 학습 모델 평가 (같은 ep 수)
  5) 결과 행 누적

마지막:
  - results/cross_location_eval.csv  (17행 × 알고리즘 metrics)
  - results/cross_location_eval.png  (mean_R bar + Δ vs heuristic 라인 차트)

학습 시 obs/action shape 와 일치해야 모델 load 가능:
  incident_size, uav_count 가 학습 시점과 같아야 함 (기본 100 / 25).
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

from make_uav_scenario import UAVScenarioGenerator
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ppo_path", required=True)
    p.add_argument("--dqn_path", required=True)
    p.add_argument("--reinforce_path", required=True)
    p.add_argument("--incident_size", type=int, default=100,
                   help="학습 시 사용한 값과 동일해야 함")
    p.add_argument("--uav_count", type=int, default=25)
    p.add_argument("--n_episodes", type=int, default=1000,
                   help="좌표 한 곳당 평가 ep 수 (휴리스틱/RL 동일). main.py totalSamples 와 일치")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_path", default=".")
    p.add_argument("--hospital_data", default=None)
    p.add_argument("--out_csv", default="results/cross_location_eval.csv")
    p.add_argument("--plot_out", default="results/cross_location_eval.png")
    p.add_argument("--python", default=sys.executable,
                   help="main.py 호출용 python 인터프리터")
    p.add_argument("--plot_only", action="store_true",
                   help="평가 건너뛰고 기존 --out_csv 만 읽어 PNG 재생성")
    p.add_argument("--exp_prefix", default="crosseval",
                   help="시나리오 experiment_id prefix. 결과 폴더: exp_<prefix>_<region>_uav "
                        "(예: crosseval_1M -> exp_crosseval_1M_서울_uav)")
    p.add_argument("--skip_heuristic", action="store_true",
                   help="휴리스틱 main.py 실행을 건너뛰고 기존 stat.txt 를 그대로 사용. "
                        "--heuristic_exp_prefix 와 함께 사용")
    p.add_argument("--heuristic_exp_prefix", default=None,
                   help="--skip_heuristic 시 기존 휴리스틱 결과를 찾을 prefix. "
                        "예: crosseval -> results/exp_crosseval_<region>_uav/.../*_stat.txt")
    return p.parse_args()


def gen_scenario_for_region(short_name, lat, lon, incident_size, uav_count,
                             total_samples, base_path, hospital_data,
                             exp_prefix: str = "crosseval") -> str:
    gen = UAVScenarioGenerator(base_path, f"{exp_prefix}_{short_name}", hospital_data)
    return gen.generate(
        latitude=lat, longitude=lon,
        incident_size=incident_size, uav_count=uav_count,
        uav_velocity=80, uav_handover_time=15.0,
        total_samples=total_samples, random_seed=0, rule_test=True,
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


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)
    hospital_data = args.hospital_data or os.path.join(base, "scenarios", "엑셀 결합 데이터.xlsx")

    if args.plot_only:
        df = pd.read_csv(args.out_csv, encoding="utf-8-sig")
        print(f"Loaded CSV: {args.out_csv}  rows={len(df)}")
        plot_results(df, args.plot_out)
        return

    if args.skip_heuristic and not args.heuristic_exp_prefix:
        raise SystemExit("--skip_heuristic 사용 시 --heuristic_exp_prefix 를 함께 지정해야 합니다 "
                         "(예: --heuristic_exp_prefix crosseval)")

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

    print(f"\nEvaluating {len(LOCATIONS)} locations (n_episodes={args.n_episodes}, seed={args.seed})")

    rows = []
    for i, (short_name, full_name, lat, lon) in enumerate(LOCATIONS, 1):
        print(f"\n[{i:2d}/{len(LOCATIONS)}] {short_name} {full_name}  ({lat},{lon})")

        try:
            config_path = gen_scenario_for_region(
                short_name, lat, lon,
                args.incident_size, args.uav_count, args.n_episodes,
                base, hospital_data, exp_prefix=args.exp_prefix,
            )
        except Exception as e:
            print(f"  ! 시나리오 생성 실패: {e}")
            continue

        # 1) Heuristic stat 확보
        if args.skip_heuristic:
            print(f"  Heuristic skipped — reusing existing stat (prefix={args.heuristic_exp_prefix})")
            old_exp_id = f"exp_{args.heuristic_exp_prefix}_{short_name}_uav"
            old_output_path = f"./results/{old_exp_id}"
            try:
                heur_best, _, stat_path = heuristic_best_from_yaml_dir(
                    config_path, output_path_override=old_output_path)
                print(f"    loaded -> {stat_path}")
                print(f"    best  : {heur_best['rule_name']}  R={heur_best['R_mean']:.3f}  "
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
                      f"PDR={heur_best['PDR_mean']:.3f}  time={heur_best['time_mean']:.0f}")
            except Exception as e:
                print(f"  ! main.py 실패: {e}")
                continue

        # 2) RL 평가 (in-memory)
        rl_env_factory = make_eval_env(config_path)

        ppo_r, ppo_pdr, ppo_t, ppo_ci = eval_policy(
            rl_env_factory, ppo_policy(ppo_model), args.n_episodes, args.seed)
        print(f"    PPO        R={ppo_r:.3f}  PDR={ppo_pdr:.3f}  time={ppo_t:.0f}")

        dqn_r, dqn_pdr, dqn_t, dqn_ci = eval_policy(
            rl_env_factory, dqn_policy(dqn_model), args.n_episodes, args.seed)
        print(f"    DQN        R={dqn_r:.3f}  PDR={dqn_pdr:.3f}  time={dqn_t:.0f}")

        rein_r, rein_pdr, rein_t, rein_ci = eval_policy(
            rl_env_factory, reinforce_policy(rein_agent), args.n_episodes, args.seed)
        print(f"    REINFORCE  R={rein_r:.3f}  PDR={rein_pdr:.3f}  time={rein_t:.0f}")

        rows.append({
            "region": short_name, "name": full_name, "lat": lat, "lon": lon,
            "heuristic_rule": heur_best["rule_name"],
            "heuristic_R":    heur_best["R_mean"],
            "heuristic_PDR":  heur_best["PDR_mean"],
            "heuristic_time": heur_best["time_mean"],
            "PPO_R": ppo_r, "PPO_PDR": ppo_pdr, "PPO_time": ppo_t,
            "DQN_R": dqn_r, "DQN_PDR": dqn_pdr, "DQN_time": dqn_t,
            "REINFORCE_R": rein_r, "REINFORCE_PDR": rein_pdr, "REINFORCE_time": rein_t,
            "PPO_vs_heur":       ppo_r  - heur_best["R_mean"],
            "DQN_vs_heur":       dqn_r  - heur_best["R_mean"],
            "REINFORCE_vs_heur": rein_r - heur_best["R_mean"],
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
    print("\n=== mean reward 요약 ===")
    print(df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print("\n=== 휴리스틱 best 를 이긴 횟수 ===")
    print(f"  PPO       : {(df['PPO_vs_heur']       > 0).sum():2d}/{len(df)}")
    print(f"  DQN       : {(df['DQN_vs_heur']       > 0).sum():2d}/{len(df)}")
    print(f"  REINFORCE : {(df['REINFORCE_vs_heur'] > 0).sum():2d}/{len(df)}")

    plot_results(df, args.plot_out)


def plot_results(df, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import platform
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    regions = df["region"].tolist()
    x = np.arange(len(regions))
    width = 0.2

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    ax.bar(x - 1.5 * width, df["heuristic_R"], width, label="Heuristic best", color="#888")
    ax.bar(x - 0.5 * width, df["PPO_R"],       width, label="PPO",            color="#1f77b4")
    ax.bar(x + 0.5 * width, df["DQN_R"],       width, label="DQN",            color="#ff7f0e")
    ax.bar(x + 1.5 * width, df["REINFORCE_R"], width, label="REINFORCE",      color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=0)
    ax.set_ylabel("mean reward (1000 ep)")
    ax.set_title("Cross-location evaluation: mean reward per region")
    ax.legend(loc="lower right", ncol=4)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.axhline(0, color="#888", linewidth=1)
    ax.plot(x, df["PPO_vs_heur"],       "o-", label="PPO - Heur",       color="#1f77b4")
    ax.plot(x, df["DQN_vs_heur"],       "s-", label="DQN - Heur",       color="#ff7f0e")
    ax.plot(x, df["REINFORCE_vs_heur"], "^-", label="REINFORCE - Heur", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=0)
    ax.set_ylabel("Δ vs heuristic best")
    ax.set_title("RL 우위 폭 (양수 = 휴리스틱 초과)")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
