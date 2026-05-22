"""17지역 병렬 대각 평가 런처 (Plan 1).

각 지역의 학습 모델 3종을 그 지역 시나리오에서 평가하는 eval_region.py 워커를
17개 병렬 subprocess 로 띄우고, 결과 JSON 행을 모아 CSV + PNG 로 취합한다.

- CUDA_VISIBLE_DEVICES="" 로 CPU 강제.
- 각 워커는 main.py(휴리스틱) 1개 + RL 평가 → 지역당 1~2 코어. 17지역 동시 실행.

eval_region.py 워커는 sim 디버그 print spam 을 자체적으로 /dev/null 로 돌리고
진행 메시지만 stderr 로 낸다. → 여기서는 stderr 만 .err 로 캡처.

출력:
  results/plan1_regional_eval.csv
  results/plan1_regional_eval.png
  results/rl/plan1/eval_logs/<region>_<ts>.err, <region>.json

예:
  python src/rl_src/run_grid_eval.py
  python src/rl_src/run_grid_eval.py --regions 서울 부산 --n_episodes 300
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from cross_location_eval import plot_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="scenarios/plan1_manifest.json")
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--model_root", default="results/rl/plan1")
    p.add_argument("--national", action="store_true",
                   help="전국 단일 정책 평가 — <model_root>/{ppo,dqn,reinforce}/ "
                        "모델을 17지역 공통 사용")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--base_path", default=".")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out_csv", default="results/plan1_regional_eval.csv")
    p.add_argument("--plot_out", default="results/plan1_regional_eval.png")
    p.add_argument("--max_parallel", type=int, default=17,
                   help="동시 실행 지역 워커 수")
    p.add_argument("--skip_heuristic", action="store_true")
    p.add_argument("--reduced_obs", action="store_true",
                   help="피드백 3: reduced obs 모델 평가 (MCI_REDUCED_OBS=1). "
                        "학습 시 사용했으면 반드시 켜야 obs 차원 정합")
    p.add_argument("--poll_interval", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    regions = list(manifest.keys()) if not args.regions else args.regions

    eval_log_dir = os.path.join(args.model_root, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = []
    for region in regions:
        config_path = manifest.get(region)
        if not config_path or not os.path.exists(config_path):
            print(f"⚠️ {region}: config 미발견, 건너뜀")
            continue
        jobs.append({
            "region": region, "config": config_path,
            "out_json": os.path.join(eval_log_dir, f"{region}.json"),
        })

    if not jobs:
        raise SystemExit("평가할 지역이 없습니다.")

    print(f"[grid-eval] {len(jobs)}개 지역 병렬 평가  "
          f"n_episodes={args.n_episodes}  max_parallel={args.max_parallel}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # 잡당 1스레드 고정 (numpy/torch OpenMP 스레드 폭발 방지)
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        env[_k] = "1"
    if args.reduced_obs:
        env["MCI_REDUCED_OBS"] = "1"

    pending, running, done = list(jobs), [], []
    t0 = time.time()

    while pending or running:
        while pending and len(running) < args.max_parallel:
            j = pending.pop(0)
            cmd = [args.python, os.path.join(THIS_DIR, "eval_region.py"),
                   "--region", j["region"],
                   "--config_path", j["config"],
                   "--model_root", args.model_root,
                   "--n_episodes", str(args.n_episodes),
                   "--seed", str(args.seed),
                   "--base_path", base,
                   "--python", args.python,
                   "--out_json", j["out_json"]]
            if args.skip_heuristic:
                cmd.append("--skip_heuristic")
            if args.national:
                cmd.append("--national")
            err_p = os.path.join(eval_log_dir, f"{j['region']}_{ts}.err")
            ef = open(err_p, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=ef, env=env, cwd=base)
            j.update(proc=proc, err_f=ef, err_p=err_p, t_start=time.time())
            running.append(j)
            print(f"  ▶ {j['region']:5s} pid={proc.pid:7d}  (running={len(running)}, pending={len(pending)})")

        time.sleep(args.poll_interval)

        for j in list(running):
            rc = j["proc"].poll()
            if rc is not None:
                j["err_f"].close()
                running.remove(j)
                done.append((j, rc))
                el = int(time.time() - j["t_start"])
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  ■ {j['region']:5s} {status:12s} {el:5d}s  "
                      f"(남은={len(pending)+len(running)})")

    fails = [(j, rc) for j, rc in done if rc != 0]
    for j, rc in fails:
        print(f"  ✗ {j['region']} rc={rc}  err 로그: {j['err_p']}")

    # 결과 취합
    rows = []
    for j, rc in done:
        if rc == 0 and os.path.exists(j["out_json"]):
            with open(j["out_json"], encoding="utf-8") as f:
                rows.append(json.load(f))
    if not rows:
        raise SystemExit("취합할 결과 행이 없습니다.")

    df = pd.DataFrame(rows)
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[grid-eval] CSV 저장: {out_csv}  ({len(df)}행)")

    summary = ["region", "heuristic_R", "PPO_R", "DQN_R", "REINFORCE_R",
               "PPO_vs_heur", "DQN_vs_heur", "REINFORCE_vs_heur"]
    print("\n=== mean reward 요약 (R) ===")
    print(df[summary].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\n=== 휴리스틱 best 를 이긴 횟수 ===")
    for algo in ("PPO", "DQN", "REINFORCE"):
        r = (df[f"{algo}_vs_heur"] > 0).sum()
        rw = (df[f"{algo}_woG_vs_heur"] > 0).sum()
        print(f"  {algo:10s} R={r:2d}/{len(df)}  R_woG={rw:2d}/{len(df)}")

    plot_results(df, args.plot_out)
    print(f"\n[grid-eval] wall-clock={int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
