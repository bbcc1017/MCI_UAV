"""DQN / PPO / REINFORCE 동시 학습 launcher.

예:
    python src/rl_src/run_all_parallel.py \
        --config_path "scenarios/.../config.yaml" \
        --total_timesteps 200000 --seed 0 \
        --log_root results/rl --suffix p100

생성 폴더:
    results/rl/dqn_p100/
    results/rl/ppo_p100/
    results/rl/reinforce_p100/
    results/rl/run_logs/  (stdout/stderr 캡처)
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_root", default="results/rl")
    p.add_argument("--suffix", default="run",
                   help="log_dir 접미사 (e.g. 'p100' -> 'dqn_p100')")
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo", "reinforce"],
                   choices=["dqn", "ppo", "reinforce"],
                   help="실행할 알고리즘 (기본: 셋 다)")
    p.add_argument("--n_envs_ppo", type=int, default=4)
    p.add_argument("--python", default=sys.executable,
                   help="파이썬 인터프리터 (기본: 현재)")
    p.add_argument("--hybrid_amb_rule", nargs=4, default=None,
                   metavar=("PRIORITY", "HOS_SELECT", "RED_MODE", "YELLOW_MODE"),
                   help="2안 학습: AMB 결정을 룰에 위임 (모든 알고리즘에 일괄 전달)")
    return p.parse_args()


def build_cmd(args, algo: str):
    log_dir = os.path.join(args.log_root, f"{algo}_{args.suffix}")
    base = [
        args.python,
        os.path.join(THIS_DIR, f"train_{algo}.py"),
        "--config_path", args.config_path,
        "--total_timesteps", str(args.total_timesteps),
        "--seed", str(args.seed),
        "--log_dir", log_dir,
    ]
    if algo == "ppo":
        base += ["--n_envs", str(args.n_envs_ppo)]
    if args.hybrid_amb_rule:
        base += ["--hybrid_amb_rule"] + list(args.hybrid_amb_rule)
    return base, log_dir


def main():
    args = parse_args()

    run_log_dir = os.path.join(args.log_root, "run_logs")
    os.makedirs(run_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    procs = []
    print(f"[launcher] {len(args.algos)}개 알고리즘 동시 학습 시작")
    print(f"[launcher] timestep={args.total_timesteps}  seed={args.seed}  suffix={args.suffix}")
    print()

    t0 = time.time()
    for algo in args.algos:
        cmd, log_dir = build_cmd(args, algo)
        out_path = os.path.join(run_log_dir, f"{algo}_{args.suffix}_{ts}.out")
        err_path = os.path.join(run_log_dir, f"{algo}_{args.suffix}_{ts}.err")
        out_f = open(out_path, "w", encoding="utf-8")
        err_f = open(err_path, "w", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        procs.append({
            "algo": algo, "proc": proc, "log_dir": log_dir,
            "out_path": out_path, "err_path": err_path,
            "out_f": out_f, "err_f": err_f, "t_start": time.time(),
        })
        print(f"  [{algo:10s}]  pid={proc.pid:6d}  log_dir={log_dir}")
        print(f"             stdout -> {out_path}")

    print(f"\n[launcher] 모두 띄움. wait() 진입...\n")

    for p in procs:
        rc = p["proc"].wait()
        elapsed = time.time() - p["t_start"]
        p["out_f"].close()
        p["err_f"].close()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  [{p['algo']:10s}]  {status:20s}  {int(elapsed)}s   final_model={p['log_dir']}")

    total_elapsed = time.time() - t0
    print(f"\n[launcher] 전체 완료. wall-clock = {int(total_elapsed)}s")
    print(f"[launcher] TensorBoard:  python -m tensorboard.main --logdir {args.log_root}")


if __name__ == "__main__":
    main()
