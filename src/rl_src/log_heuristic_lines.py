"""지역별 휴리스틱 best rule reward 를 TensorBoard 평탄선으로 기록.

이미 산출된 평가 CSV(예: results/plan1_regional_eval.csv — region / heuristic_R /
heuristic_R_woG 컬럼)를 읽어, 각 지역 RL 학습 폴더 안에 heuristic_best run 을
만든다. 휴리스틱 시뮬 재실행 없이 CSV 값만 사용한다.

→ tensorboard --logdir results/rl/plan1/<region> 하면 그 지역의
  dqn/ppo/reinforce 학습 곡선과 휴리스틱 best 평탄선이 같은 화면에 겹쳐 보인다.

예:
  python src/rl_src/log_heuristic_lines.py \
      --eval_csv results/plan1_regional_eval.csv \
      --rl_root results/rl/plan1 --max_step 200000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from log_heuristic_baseline import write_flat_line


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", default="results/plan1_regional_eval.csv",
                   help="region / heuristic_R / heuristic_R_woG 컬럼을 가진 평가 CSV")
    p.add_argument("--rl_root", default="results/rl/plan1",
                   help="<rl_root>/<region>/ 아래에 heuristic_best run 생성")
    p.add_argument("--max_step", type=int, default=200000,
                   help="평탄선 x축 끝점 (RL 학습 total_timesteps 와 맞춤)")
    p.add_argument("--tag", default="rollout/ep_rew_mean",
                   help="SB3 학습 곡선과 동일 태그")
    p.add_argument("--metric", choices=["R", "woG", "both"], default="R",
                   help="R: heuristic_R / woG: heuristic_R_woG / both: 둘 다")
    p.add_argument("--national", action="store_true",
                   help="전국 정책 모드 — 지역별이 아니라 17지역 heuristic_R 평균을 "
                        "<rl_root>/heuristic_best 단일 평탄선으로 기록")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.eval_csv, encoding="utf-8-sig")

    metrics = []
    if args.metric in ("R", "both"):
        metrics.append(("heuristic_R", "heuristic_best"))
    if args.metric in ("woG", "both"):
        metrics.append(("heuristic_R_woG", "heuristic_best_woG"))

    # 전국 정책: 학습 곡선이 1개(17지역 혼합) → 17지역 휴리스틱 best 평균을 단일선으로
    if args.national:
        for col, run in metrics:
            mean_v = float(df[col].mean())
            run_path = write_flat_line(args.rl_root, os.path.join(run, "tb"),
                                       mean_v, args.max_step, args.tag)
            print(f"  {col} 17지역 평균={mean_v:.3f}  "
                  f"(min={df[col].min():.2f}, max={df[col].max():.2f}) → {run_path}")
        print(f"\n보기: tensorboard --logdir {args.rl_root}")
        return

    n = 0
    for _, row in df.iterrows():
        region = str(row["region"])
        region_dir = os.path.join(args.rl_root, region)
        if not os.path.isdir(region_dir):
            print(f"  ! {region}: {region_dir} 없음 — 건너뜀")
            continue
        msg = [f"  {region:5s}"]
        for col, run in metrics:
            value = float(row[col])
            run_path = write_flat_line(region_dir, os.path.join(run, "tb"),
                                       value, args.max_step, args.tag)
            msg.append(f"{col}={value:.2f}")
        print("  ".join(msg))
        n += 1

    print(f"\n{n}개 지역 휴리스틱 평탄선 기록 완료.")
    print(f"보기: tensorboard --logdir {args.rl_root}/<지역>  "
          f"(예: {args.rl_root}/서울)")


if __name__ == "__main__":
    main()
