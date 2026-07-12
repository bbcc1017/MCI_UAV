"""v5 공정비교 하네스 — zoo 학습곡선 추출·플롯(TB 이벤트 → CSV + PNG 2장).

`results/rl/zoo/<algo>_s<seed>/tb/`(train_zoo 저장 관례)에서 `rollout/ep_rew_mean` 스칼라를
긁어 CSV 로 만들고, env-step 축·wall-clock 축 곡선 2장을 그린다(샘플효율/계산효율 분리 보고,
계획 §3.1-7). `--extra_dirs` 로 챔피언 PPO 런 등 zoo 밖 tb 디렉터리를 추가할 수 있다.

run 식별: 런 디렉터리의 meta.json(algo/seed) 우선 → 디렉터리명 `<algo>_s<seed>` 정규식 →
폴백(디렉터리명=algo, seed 0). ⚠️PPO(champion)의 ep_rew_mean 은 VecNormalize norm_reward
정규화 보상 단위라 zoo(pdrwog raw)와 절대 스케일 비교 불가 — 곡선은 알고리즘 내 추이용,
교차 비교는 paired 판정 CSV 로(보고서 관례).

CSV 컬럼: algo,seed,step,wall_time,ep_rew_mean (wall_time=그 런 첫 이벤트 기준 상대 초).
플롯: 알고별 색, 시드별 반투명 실선. `--hline` 에 LB-T4 수평선(ep_rew_mean=1−PDR 단위 등
호출자 책임). NanumGothic(레포 관례), circled digit(①②) 라벨 금지.

예:
  python src/rl_src/learning_curve_zoo.py                       # results/rl/zoo 스캔
  python src/rl_src/learning_curve_zoo.py --hline 0.88 \\
      --extra_dirs results/rl/redesign/v4_plr2_s0,results/rl/redesign/v4_plr2_s1
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
TAG = "rollout/ep_rew_mean"
_RUN_RE = re.compile(r"^(.+)_s(\d+)$")


def _run_identity(run_dir: str):
    """(algo, seed) — meta.json 우선, 디렉터리명 <algo>_s<seed> 차선, 폴백 (basename, 0)."""
    base = os.path.basename(os.path.normpath(run_dir))
    meta_p = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_p):
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            return str(meta.get("algo", base)), int(meta.get("seed", 0))
        except Exception:
            pass
    m = _RUN_RE.match(base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def read_run_scalars(run_dir: str, tag: str = TAG):
    """run_dir/tb/ 이하 전 이벤트 파일에서 tag 스칼라 수집 → [(wall_time, step, value)] 정렬.

    SB3 는 tb/<tb_log_name>_1/ 서브디렉터리, 자작(reinforce_vec)은 tb/ 직하에 쓰므로
    os.walk 로 양쪽을 커버. 같은 step 중복(재시작) 은 나중 wall_time 이 이기게 정렬만 한다.
    """
    from tensorboard.backend.event_processing import event_accumulator
    tb_root = os.path.join(run_dir, "tb")
    if not os.path.isdir(tb_root):
        return []
    rows = []
    for root, _dirs, files in os.walk(tb_root):
        for fn in files:
            if not fn.startswith("events.out.tfevents"):
                continue
            ea = event_accumulator.EventAccumulator(
                os.path.join(root, fn), size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
            if tag in ea.Tags().get("scalars", []):
                for ev in ea.Scalars(tag):
                    rows.append((float(ev.wall_time), int(ev.step), float(ev.value)))
    rows.sort()
    return rows


def collect_runs(scan_dir: str, extra_dirs):
    """스캔 대상 런 디렉터리 열거 → [(algo, seed, run_dir, rows)] (rows 비면 제외)."""
    run_dirs = []
    if os.path.isdir(scan_dir):
        for name in sorted(os.listdir(scan_dir)):
            d = os.path.join(scan_dir, name)
            if os.path.isdir(os.path.join(d, "tb")):
                run_dirs.append(d)
    for d in extra_dirs:
        if not os.path.isabs(d):
            d = os.path.join(REPO, d)
        if os.path.isdir(os.path.join(d, "tb")):
            run_dirs.append(d)
        else:
            print(f"[curve] tb 없음, 건너뜀: {d}", flush=True)

    runs = []
    for d in run_dirs:
        rows = read_run_scalars(d)
        if not rows:
            print(f"[curve] '{TAG}' 스칼라 없음, 건너뜀: {d}", flush=True)
            continue
        algo, seed = _run_identity(d)
        runs.append((algo, seed, d, rows))
        print(f"[curve] {algo} s{seed}: {len(rows)}점 ({os.path.relpath(d, REPO)})", flush=True)
    return runs


def write_csv(runs, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "seed", "step", "wall_time", "ep_rew_mean"])
        for algo, seed, _d, rows in runs:
            t0 = rows[0][0]
            for wt, step, val in rows:
                w.writerow([algo, seed, step, f"{wt - t0:.1f}", f"{val:.6f}"])
    print(f"[curve] CSV 저장: {out_csv}", flush=True)


def plot_curves(runs, out_png: str, x_axis: str, hline, title: str):
    """x_axis: 'step'(env-step) | 'wall'(상대 시간, h)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

    algos = sorted({r[0] for r in runs})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(algos)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    seen = set()
    for algo, seed, _d, rows in sorted(runs, key=lambda r: (r[0], r[1])):
        t0 = rows[0][0]
        if x_axis == "step":
            xs = [r[1] for r in rows]
        else:
            xs = [(r[0] - t0) / 3600.0 for r in rows]
        ys = [r[2] for r in rows]
        label = algo if algo not in seen else None  # 범례는 알고당 1개
        seen.add(algo)
        ax.plot(xs, ys, color=colors[algo], alpha=0.55, linewidth=1.4, label=label)
    if hline is not None:
        ax.axhline(hline, color="dimgray", linestyle="--", linewidth=1.2,
                   label=f"LB-T4 ({hline:g})")
    ax.set_xlabel("env steps" if x_axis == "step" else "wall-clock (h)")
    ax.set_ylabel("ep_rew_mean")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[curve] PNG 저장: {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir", default=os.path.join(REPO, "results/rl/zoo"),
                    help="런 디렉터리(<algo>_s<seed>/tb) 스캔 루트")
    ap.add_argument("--extra_dirs", default="",
                    help="쉼표구분 추가 런 디렉터리(예: 챔피언 results/rl/redesign/v4_plr2_s0)")
    ap.add_argument("--out_csv", default=os.path.join(REPO, "results/rl/zoo/learning_curves.csv"))
    ap.add_argument("--out_png_prefix", default=None,
                    help="PNG 경로 접두(기본 out_csv 와 같은 디렉터리 learning_curves)")
    ap.add_argument("--hline", type=float, default=None,
                    help="LB-T4 수평선 값(ep_rew_mean 단위) — 미지정 시 생략")
    A = ap.parse_args()

    extra = [d for d in A.extra_dirs.split(",") if d.strip()]
    runs = collect_runs(A.scan_dir, extra)
    if not runs:
        print("[curve] 수집된 런 없음 — scan_dir/extra_dirs 확인", flush=True)
        return
    write_csv(runs, A.out_csv)

    prefix = A.out_png_prefix or os.path.splitext(A.out_csv)[0]
    plot_curves(runs, prefix + "_steps.png", "step", A.hline,
                "v5 zoo 학습곡선 (env-step 축)")
    plot_curves(runs, prefix + "_wall.png", "wall", A.hline,
                "v5 zoo 학습곡선 (wall-clock 축)")


if __name__ == "__main__":
    main()
