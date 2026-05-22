"""plan1nat 전국 단일 정책의 hold-out 일반화 평가 (피드백 3).

plan1nat_f3 평가는 학습 좌표(시청/도청)에서 그대로 봤다. 이 스크립트는 학습된
전국 정책을 학습 좌표가 아닌 지역 내부 랜덤 지점(plan1nat_eval_manifest.json,
지역당 5점)에서 평가한다 — 좌표 일반화 시험.

각 (지역, 점) 시나리오마다 eval_region.py 워커를 띄운다(무수정 재사용):
  - 휴리스틱 main.py 64룰 실행 → best 추출
  - 전국 PPO/DQN/REINFORCE 모델(--national) 평가
85개 워커를 병렬로 돌리고, 점별 행을 모아 두 가지로 취합한다:
  - results/<out_prefix>_points.csv : 85행 (지역·점별 raw)
  - results/<out_prefix>.csv        : 17행 (지역별 5점 평균)
  - results/<out_prefix>.png        : 지역평균 막대/Δ 차트

reduced obs(plan1nat_f3) 모델이므로 MCI_REDUCED_OBS=1 을 워커에 주입한다.

예:
  python src/rl_src/eval_random_points.py
  python src/rl_src/eval_random_points.py --regions 서울 부산 --n_episodes 300
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

# 점별 raw / 지역평균 양쪽에 쓰는 수치 컬럼
_METRIC_COLS = [
    "heuristic_R", "heuristic_R_woG", "heuristic_PDR", "heuristic_PDR_woG",
    "heuristic_time",
    "PPO_R", "PPO_R_woG", "PPO_PDR", "PPO_PDR_woG", "PPO_time",
    "DQN_R", "DQN_R_woG", "DQN_PDR", "DQN_PDR_woG", "DQN_time",
    "REINFORCE_R", "REINFORCE_R_woG", "REINFORCE_PDR", "REINFORCE_PDR_woG",
    "REINFORCE_time",
    "PPO_vs_heur", "DQN_vs_heur", "REINFORCE_vs_heur",
    "PPO_woG_vs_heur", "DQN_woG_vs_heur", "REINFORCE_woG_vs_heur",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="scenarios/plan1nat_eval_manifest.json")
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--model_root", default="results/rl/plan1nat_f3/national")
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--base_path", default=".")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out_prefix", default="results/plan1nat_f3_randeval")
    p.add_argument("--max_parallel", type=int, default=24,
                   help="동시 실행 워커 수 (워커당 휴리스틱 64k에피소드 CPU 바운드)")
    p.add_argument("--reduced_obs", action="store_true", default=True,
                   help="plan1nat_f3 = reduced obs 학습 → 기본 켜짐")
    p.add_argument("--full_obs", dest="reduced_obs", action="store_false",
                   help="reduced obs 끄기 (구 plan1nat 모델 평가용)")
    p.add_argument("--poll_interval", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    regions = list(manifest.keys()) if not args.regions else args.regions

    eval_log_dir = os.path.join(args.model_root, "randeval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = []
    for region in regions:
        # manifest 는 {region: {slot_str: cfg}} (gen_eval_points.py) — slot 정렬
        entry = manifest.get(region, {})
        items = (sorted(((int(k), v) for k, v in entry.items()))
                 if isinstance(entry, dict) else list(enumerate(entry)))
        for idx, cfg in items:
            if not cfg or not os.path.exists(cfg):
                print(f"⚠️ {region}#{idx}: config 미발견, 건너뜀 — {cfg}")
                continue
            jobs.append({
                "region": region, "idx": idx, "config": cfg,
                "tag": f"{region}#{idx}",
                "out_json": os.path.join(eval_log_dir, f"{region}_{idx}.json"),
            })
    if not jobs:
        raise SystemExit("평가할 시나리오가 없습니다 — manifest 확인.")

    print(f"[randeval] {len(jobs)}개 시나리오 병렬 평가  "
          f"n_episodes={args.n_episodes}  max_parallel={args.max_parallel}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""
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
                   "--national",
                   "--n_episodes", str(args.n_episodes),
                   "--seed", str(args.seed),
                   "--base_path", base,
                   "--python", args.python,
                   "--out_json", j["out_json"]]
            err_p = os.path.join(eval_log_dir, f"{j['region']}_{j['idx']}_{ts}.err")
            ef = open(err_p, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=ef, env=env, cwd=base)
            j.update(proc=proc, err_f=ef, err_p=err_p, t_start=time.time())
            running.append(j)
            print(f"  ▶ {j['tag']:9s} pid={proc.pid:7d}  "
                  f"(running={len(running)}, pending={len(pending)})")

        time.sleep(args.poll_interval)
        for j in list(running):
            rc = j["proc"].poll()
            if rc is not None:
                j["err_f"].close()
                running.remove(j)
                done.append((j, rc))
                el = int(time.time() - j["t_start"])
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  ■ {j['tag']:9s} {status:12s} {el:5d}s  "
                      f"(남은={len(pending)+len(running)})")

    fails = [(j, rc) for j, rc in done if rc != 0]
    for j, rc in fails:
        print(f"  ✗ {j['tag']} rc={rc}  err: {j['err_p']}")

    rows = []
    for j, rc in done:
        if rc == 0 and os.path.exists(j["out_json"]):
            with open(j["out_json"], encoding="utf-8") as f:
                row = json.load(f)
            row["point_idx"] = j["idx"]
            rows.append(row)
    if not rows:
        raise SystemExit("취합할 결과 행이 없습니다.")

    pts = pd.DataFrame(rows).sort_values(["region", "point_idx"])
    out_pts = os.path.abspath(args.out_prefix + "_points.csv")
    os.makedirs(os.path.dirname(out_pts), exist_ok=True)
    pts.to_csv(out_pts, index=False, encoding="utf-8-sig")
    print(f"\n[randeval] 점별 CSV: {out_pts}  ({len(pts)}행)")

    # 지역별 5점 평균
    agg = pts.groupby("region", as_index=False)[_METRIC_COLS].mean()
    agg["heuristic_rule"] = "(5점 평균)"
    agg["n_points"] = pts.groupby("region")["point_idx"].count().values
    out_csv = os.path.abspath(args.out_prefix + ".csv")
    agg.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[randeval] 지역평균 CSV: {out_csv}  ({len(agg)}행)")

    summary = ["region", "n_points", "heuristic_R", "PPO_R", "DQN_R",
               "REINFORCE_R", "PPO_vs_heur", "DQN_vs_heur", "REINFORCE_vs_heur"]
    print("\n=== 지역평균 mean reward 요약 (R) ===")
    print(agg[summary].to_string(index=False,
                                 float_format=lambda x: f"{x:7.3f}"))
    print("\n=== 휴리스틱 best 를 이긴 지역 수 (5점 평균 기준) ===")
    for algo in ("PPO", "DQN", "REINFORCE"):
        r = (agg[f"{algo}_vs_heur"] > 0).sum()
        rw = (agg[f"{algo}_woG_vs_heur"] > 0).sum()
        print(f"  {algo:10s} R={r:2d}/{len(agg)}  R_woG={rw:2d}/{len(agg)}  "
              f"평균 Δ={agg[f'{algo}_vs_heur'].mean():+.3f}")
    print("\n=== 점 단위(85점) 승률 ===")
    for algo in ("PPO", "DQN", "REINFORCE"):
        r = (pts[f"{algo}_vs_heur"] > 0).sum()
        print(f"  {algo:10s} {r:2d}/{len(pts)}")

    plot_results(agg, args.out_prefix + ".png")
    print(f"\n[randeval] wall-clock={int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
