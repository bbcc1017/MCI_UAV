"""학습 시간 구간 분해 — sim 가속이 학습 wall-clock 에 얼마나 반영될 수 있는지 상한을 구한다.

배경
----
env 한 스텝은 2.23× 빨라졌는데 학습 wall-clock 은 1.02× 였다(50k steps, n_envs 8, subproc,
batch 128). 원인을 추측하지 않고 쪼갠다. SB3 학습 1회 반복은 크게 두 토막이다.

    collect_rollouts : env.step × n + 정책 forward + 버퍼 기록   ← sim 이 여기 있다
    train            : PPO 경사 갱신 (n_epochs × minibatch)      ← sim 무관

`--vec dummy` 로 돌려 env.step 을 **같은 프로세스에서 직접 계측**한다(subproc 의 IPC 가
env 시간을 가려 분해가 불가능해지는 것을 피함). 비율만 필요하므로 dummy 로 충분하다.

    python src/sim_src_upgrade/bench/profile_train.py --core old  --total_timesteps 20000
    python src/sim_src_upgrade/bench/profile_train.py --core fast --total_timesteps 20000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

from sim_src_upgrade._paths import REPO, ensure_paths  # noqa: E402

DEFAULT_CFG = os.path.join(REPO, "scenarios/exp_시도/osrm/exp_서울_osrm/(37.5666,126.9784)/config_(37.5666,126.9784).yaml")

ACC = {"collect": 0.0, "train": 0.0, "env_step": 0.0, "n_env_step": 0, "n_collect": 0, "n_train": 0}


def install_timers():
    """`collect_rollouts`/`train`/래퍼 `step` 에 누적 타이머를 건다(값·동작 불변)."""
    from sb3_contrib import MaskablePPO
    from hospital_feature_wrapper import HospitalFeatureWrapper

    _cr = MaskablePPO.collect_rollouts
    _tr = MaskablePPO.train
    _st = HospitalFeatureWrapper.step

    def collect_rollouts(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return _cr(self, *a, **k)
        finally:
            ACC["collect"] += time.perf_counter() - t0
            ACC["n_collect"] += 1

    def train(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return _tr(self, *a, **k)
        finally:
            ACC["train"] += time.perf_counter() - t0
            ACC["n_train"] += 1

    def step(self, action):
        t0 = time.perf_counter()
        try:
            return _st(self, action)
        finally:
            ACC["env_step"] += time.perf_counter() - t0
            ACC["n_env_step"] += 1

    MaskablePPO.collect_rollouts = collect_rollouts
    MaskablePPO.train = train
    HospitalFeatureWrapper.step = step


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 시간 구간 분해")
    ap.add_argument("--core", choices=["old", "fast"], required=True)
    ap.add_argument("--config_path", default=DEFAULT_CFG)
    ap.add_argument("--total_timesteps", type=int, default=20_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=512)      # 프로덕션 레시피
    ap.add_argument("--batch_size", type=int, default=512)   # 프로덕션 레시피
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--force_cpu", action="store_true",
                    help="CUDA 숨김 → CPU 학습. GPU 대비 어느 구간이 달라지는지 볼 때 사용")
    ap.add_argument("--tag", default="", help="출력 파일명 접미(예: cpu)")
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(MCI_OBS_VARIANT="essential+load+valid", MCI_H_PAD="47",
                      MCI_CAP_GATE="occ")
    ensure_paths()

    log_dir = args.log_dir or os.path.join(REPO, f"results/sim_upgrade/train_prof/{args.core}")
    shutil.rmtree(log_dir, ignore_errors=True)

    import train_ppo_feature as T
    install_timers()          # 타이머는 코어 패치 전에 걸어도 무관(둘 다 클래스 속성)
    if args.core == "fast":
        from sim_src_upgrade.drivers.run_fast import apply_fast_core
        apply_fast_core(mask_only=False)

    sys.argv = ["train_ppo_feature.py",
                "--config_path", args.config_path,
                "--total_timesteps", str(args.total_timesteps),
                "--n_envs", str(args.n_envs), "--n_steps", str(args.n_steps),
                "--batch_size", str(args.batch_size), "--n_epochs", str(args.n_epochs),
                "--vec", "dummy", "--seed", str(args.seed),
                "--extractor", "pointer", "--log_dir", log_dir]
    t0 = time.perf_counter()
    T.main()
    wall = time.perf_counter() - t0

    other = wall - ACC["collect"] - ACC["train"]
    inside = ACC["collect"] - ACC["env_step"]
    rows = {
        "core": args.core, "wall_s": wall,
        "collect_s": ACC["collect"], "train_s": ACC["train"], "other_s": other,
        "env_step_s": ACC["env_step"], "collect_minus_env_s": inside,
        "n_env_step": ACC["n_env_step"], "n_collect": ACC["n_collect"], "n_train": ACC["n_train"],
    }
    print(f"\n=== [{args.core}] {args.total_timesteps} steps · n_envs {args.n_envs} · "
          f"n_steps {args.n_steps} · batch {args.batch_size} · vec=dummy ===")
    print(f"  wall                 {wall:8.1f}s")
    print(f"  ├ collect_rollouts   {ACC['collect']:8.1f}s  ({ACC['collect']/wall*100:5.1f}%)")
    print(f"  │  ├ env.step 합      {ACC['env_step']:8.1f}s  ({ACC['env_step']/wall*100:5.1f}%)"
          f"   ← sim 가속이 닿는 구간, {ACC['n_env_step']:,}회")
    print(f"  │  └ 정책forward+버퍼 {inside:8.1f}s  ({inside/wall*100:5.1f}%)")
    print(f"  ├ train (경사갱신)     {ACC['train']:8.1f}s  ({ACC['train']/wall*100:5.1f}%)  {ACC['n_train']}회")
    print(f"  └ 그밖(셋업·저장)      {other:8.1f}s  ({other/wall*100:5.1f}%)")

    out = args.out or os.path.join(REPO, f"results/sim_upgrade/train_prof_{args.core}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[저장] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
