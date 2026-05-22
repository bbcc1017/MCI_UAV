"""17지역 × 3알고리즘 병렬 학습 그리드 런처 (Plan 1).

scenarios/plan1_manifest.json (region -> config_path) 을 읽어 각 지역마다
DQN / PPO / REINFORCE 를 subprocess 로 띄운다.

- 코어 예산 세마포어로 동시 실행 잡 수를 제한 (max_cores).
- CUDA_VISIBLE_DEVICES="" 로 CPU 강제 → GPU 1장 경합 회피.
- train_*.py 는 수정하지 않고 그대로 호출 (run_all_parallel.py 패턴 재사용).

sim_src 의 디버그 print(이벤트마다 출력) 가 매우 많아 학습 stdout 을 파일로
받으면 잡당 수 GB 가 쌓인다. → 학습 stdout 은 /dev/null 로 버리고(모니터링은
TensorBoard), stderr(실제 에러·트레이스백) 만 .err 파일로 캡처한다.

출력:
  results/rl/plan1/<region>/{dqn,ppo,reinforce}/final_model.{zip,pt}
  results/rl/plan1/<region>/{dqn,ppo,reinforce}/tb/   (TensorBoard)
  results/rl/plan1/run_logs/<region>_<algo>_<ts>.err  (stderr 만)

예:
  python src/rl_src/run_grid_parallel.py                    # 17지역 전체
  python src/rl_src/run_grid_parallel.py --regions 서울 부산  # 일부만
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="scenarios/plan1_manifest.json")
    p.add_argument("--regions", nargs="+", default=None,
                   help="부분 학습 (기본: manifest 의 전 지역)")
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo", "reinforce"],
                   choices=["dqn", "ppo", "reinforce"])
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_envs_ppo", type=int, default=4,
                   help="PPO SubprocVecEnv 환경 수 (= PPO 잡의 코어 점유량)")
    p.add_argument("--log_root", default="results/rl/plan1")
    p.add_argument("--max_cores", type=int, default=105,
                   help="동시 실행 코어 예산 (128코어 - 타 사용자 여유분)")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--poll_interval", type=float, default=3.0)
    p.add_argument("--reduced_obs", action="store_true",
                   help="피드백 3: 환자/차량 obs 를 집계 통계로 압축 (MCI_REDUCED_OBS=1)")
    return p.parse_args()


def build_cmd(python, algo, config_path, total_timesteps, seed, log_dir, n_envs_ppo):
    cmd = [python, os.path.join(THIS_DIR, f"train_{algo}.py"),
           "--config_path", config_path,
           "--total_timesteps", str(total_timesteps),
           "--seed", str(seed),
           "--log_dir", log_dir]
    if algo == "ppo":
        cmd += ["--n_envs", str(n_envs_ppo), "--vec", "subproc"]
    return cmd


def main():
    args = parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    regions = list(manifest.keys()) if not args.regions else args.regions

    run_log_dir = os.path.join(args.log_root, "run_logs")
    os.makedirs(run_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 잡 목록 구성
    jobs = []
    for region in regions:
        config_path = manifest.get(region)
        if not config_path or not os.path.exists(config_path):
            print(f"⚠️ {region}: config 미발견, 건너뜀 ({config_path})")
            continue
        for algo in args.algos:
            # PPO(subproc): 메인 1 + env 워커 n_envs → n_envs+1 코어. 그 외 1.
            cores = (args.n_envs_ppo + 1) if algo == "ppo" else 1
            jobs.append({
                "region": region, "algo": algo,
                "cores": cores,
                "config": config_path,
                "log_dir": os.path.join(args.log_root, region, algo),
            })

    if not jobs:
        raise SystemExit("실행할 잡이 없습니다. manifest/regions 를 확인하세요.")

    print(f"[grid] {len(jobs)}개 잡 (지역 {len(regions)} × 알고리즘 {len(args.algos)})  "
          f"max_cores={args.max_cores}  timesteps={args.total_timesteps}  seed={args.seed}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU 강제
    # numpy(MKL)/torch 가 프로세스마다 코어 수만큼 OpenMP 스레드를 띄워
    # 스레드 폭발(load explosion)이 난다 → 잡당 1스레드로 고정.
    # 병렬성은 잡(프로세스) 수준에서만 확보한다.
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        env[_k] = "1"
    if args.reduced_obs:
        env["MCI_REDUCED_OBS"] = "1"
        print("[grid] reduced_obs=ON — 환자/차량 obs 집계 압축")

    pending, running, done = list(jobs), [], []
    used_cores = 0
    t0 = time.time()

    while pending or running:
        # 코어 예산 내에서 가능한 만큼 launch
        progressed = True
        while progressed:
            progressed = False
            for j in list(pending):
                if used_cores + j["cores"] <= args.max_cores:
                    os.makedirs(j["log_dir"], exist_ok=True)
                    cmd = build_cmd(args.python, j["algo"], j["config"],
                                    args.total_timesteps, args.seed,
                                    j["log_dir"], args.n_envs_ppo)
                    err_p = os.path.join(run_log_dir, f"{j['region']}_{j['algo']}_{ts}.err")
                    ef = open(err_p, "w", encoding="utf-8")
                    # stdout(디버그 print spam) 은 버리고 stderr 만 캡처
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                            stderr=ef, env=env)
                    j.update(proc=proc, err_f=ef, err_p=err_p, t_start=time.time())
                    running.append(j)
                    pending.remove(j)
                    used_cores += j["cores"]
                    progressed = True
                    print(f"  ▶ {j['region']:5s} {j['algo']:10s} pid={proc.pid:7d} "
                          f"cores={j['cores']}  (used={used_cores}/{args.max_cores}, "
                          f"pending={len(pending)})")

        time.sleep(args.poll_interval)

        # 완료 잡 회수
        for j in list(running):
            rc = j["proc"].poll()
            if rc is not None:
                j["err_f"].close()
                used_cores -= j["cores"]
                running.remove(j)
                done.append((j, rc))
                el = int(time.time() - j["t_start"])
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  ■ {j['region']:5s} {j['algo']:10s} {status:12s} {el:5d}s  "
                      f"(used={used_cores}/{args.max_cores}, "
                      f"남은잡={len(pending)+len(running)})")

    fails = [(j, rc) for j, rc in done if rc != 0]
    print(f"\n[grid] 완료. {len(done)}잡  실패 {len(fails)}  "
          f"wall-clock={int(time.time()-t0)}s")
    for j, rc in fails:
        print(f"  ✗ {j['region']} {j['algo']} rc={rc}  err 로그: {j['err_p']}")
    print(f"[grid] TensorBoard:  python -m tensorboard.main --logdir {args.log_root}")


if __name__ == "__main__":
    main()
