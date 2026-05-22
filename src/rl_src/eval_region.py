"""단일 지역 대각(diagonal) 평가 워커 — Plan 1.

해당 지역에서 학습된 PPO/DQN/REINFORCE 모델 3종을 그 지역 자신의 시나리오에서
평가하고, 같은 시나리오의 휴리스틱 best 와 비교한다.
run_grid_eval.py 가 17개 지역 워커를 병렬로 띄우고 결과를 취합한다.

결과는 --out_json 에 1개 region 행(dict)으로 기록.

reuse:
  evaluate.eval_policy / ppo_policy / dqn_policy / reinforce_policy / make_eval_env
  cross_location_eval.run_main_sim / heuristic_best_from_yaml_dir
"""
import argparse
import json
import os
import sys

# sim_src 의 디버그 print(이벤트마다) spam 을 디스크에 쌓지 않도록 프로세스
# stdout(fd 1) 을 /dev/null 로 돌린다. 워커 자신의 진행 메시지는 stderr 로 낸다.
# (sim_src 는 수정하지 않는다 — 사용자 결정)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 1)


def log(msg):
    """워커 진행 메시지 — stderr 로 출력 (run_grid_eval 가 .err 로 캡처)."""
    print(msg, file=sys.stderr, flush=True)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "sim_src"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from evaluate import (eval_policy, ppo_policy, dqn_policy, reinforce_policy,
                      make_eval_env)
from cross_location_eval import run_main_sim, heuristic_best_from_yaml_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_root", default="results/rl/plan1",
                   help="기본: <model_root>/<region>/{ppo,dqn,reinforce}/final_model.* 구조")
    p.add_argument("--national", action="store_true",
                   help="전국 단일 정책 평가 — 지역 구분 없이 "
                        "<model_root>/{ppo,dqn,reinforce}/ 모델을 모든 지역에 사용")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--base_path", default=".")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out_json", required=True)
    p.add_argument("--skip_heuristic", action="store_true",
                   help="휴리스틱 main.py 재실행 없이 기존 stat.txt 사용")
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)
    region = args.region

    # 1) 휴리스틱 stat 확보
    if not args.skip_heuristic:
        log(f"[{region}] 휴리스틱 main.py 실행 ...")
        run_main_sim(args.config_path, args.python, base)
    heur_best, _, stat_path = heuristic_best_from_yaml_dir(args.config_path)
    log(f"[{region}] heuristic best={heur_best['rule_name']}  "
        f"R={heur_best['R_mean']:.3f}  R_woG={heur_best['R_woG_mean']:.3f}")

    # 2) 모델 로드 — 전국 정책이면 공통 모델, 아니면 지역 전용
    mdir = args.model_root if args.national else os.path.join(args.model_root, region)
    ppo_path = os.path.join(mdir, "ppo", "final_model.zip")
    dqn_path = os.path.join(mdir, "dqn", "final_model.zip")
    rein_path = os.path.join(mdir, "reinforce", "final_model.pt")
    for pth in (ppo_path, dqn_path, rein_path):
        if not os.path.exists(pth):
            raise SystemExit(f"[{region}] 모델 미발견: {pth}")

    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN
    from reinforce_agent import ReinforceAgent
    ppo_model = MaskablePPO.load(ppo_path)
    dqn_model = DQN.load(dqn_path)
    rein_agent = ReinforceAgent.load(rein_path)

    # 3) RL 평가 (자기 시나리오)
    factory = make_eval_env(args.config_path)
    m_ppo = eval_policy(factory, ppo_policy(ppo_model), args.n_episodes, args.seed)
    m_dqn = eval_policy(factory, dqn_policy(dqn_model), args.n_episodes, args.seed)
    m_rein = eval_policy(factory, reinforce_policy(rein_agent), args.n_episodes, args.seed)
    for tag, m in (("PPO", m_ppo), ("DQN", m_dqn), ("REINFORCE", m_rein)):
        log(f"[{region}] {tag:10s} R={m['mean_R']:.3f}  R_woG={m['mean_R_woG']:.3f}  "
            f"PDR={m['mean_PDR']:.3f}  time={m['mean_time']:.0f}")

    # 4) 결과 행 기록 (cross_location_eval.plot_results 와 동일 컬럼명)
    row = {
        "region": region,
        "heuristic_rule":    heur_best["rule_name"],
        "heuristic_R":       heur_best["R_mean"],
        "heuristic_R_woG":   heur_best["R_woG_mean"],
        "heuristic_PDR":     heur_best["PDR_mean"],
        "heuristic_PDR_woG": heur_best["PDR_woG_mean"],
        "heuristic_time":    heur_best["time_mean"],
        "PPO_R": m_ppo["mean_R"], "PPO_R_woG": m_ppo["mean_R_woG"],
        "PPO_PDR": m_ppo["mean_PDR"], "PPO_PDR_woG": m_ppo["mean_PDR_woG"],
        "PPO_time": m_ppo["mean_time"],
        "DQN_R": m_dqn["mean_R"], "DQN_R_woG": m_dqn["mean_R_woG"],
        "DQN_PDR": m_dqn["mean_PDR"], "DQN_PDR_woG": m_dqn["mean_PDR_woG"],
        "DQN_time": m_dqn["mean_time"],
        "REINFORCE_R": m_rein["mean_R"], "REINFORCE_R_woG": m_rein["mean_R_woG"],
        "REINFORCE_PDR": m_rein["mean_PDR"], "REINFORCE_PDR_woG": m_rein["mean_PDR_woG"],
        "REINFORCE_time": m_rein["mean_time"],
        "PPO_vs_heur":       m_ppo["mean_R"]  - heur_best["R_mean"],
        "DQN_vs_heur":       m_dqn["mean_R"]  - heur_best["R_mean"],
        "REINFORCE_vs_heur": m_rein["mean_R"] - heur_best["R_mean"],
        "PPO_woG_vs_heur":       m_ppo["mean_R_woG"]  - heur_best["R_woG_mean"],
        "DQN_woG_vs_heur":       m_dqn["mean_R_woG"]  - heur_best["R_woG_mean"],
        "REINFORCE_woG_vs_heur": m_rein["mean_R_woG"] - heur_best["R_woG_mean"],
        "stat_path": stat_path,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    log(f"[{region}] 완료 → {args.out_json}")


if __name__ == "__main__":
    main()
