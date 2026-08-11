"""학습 wall-clock 배속 실측 — 고정 오버헤드가 묻히는 규모에서 구/신 코어를 짝비교한다.

`verify_train_equiv.py` 는 **결과 동일성**(가중치 비트동일)을 본다. 그건 2,048스텝으로 충분하지만
그 규모에서는 고정 오버헤드(import·시나리오 빌드·forkserver 스폰·모델 저장 약 30초)가 전체를
지배해 시간 비교가 무의미하다. 그래서 배속은 여기서 따로 잰다.

⚠️ BLAS 스레드 수는 양쪽 **똑같이 고정**한다 — 스레드 수가 다르면 시간뿐 아니라 부동소수
결과까지 갈린다(`verify_train_equiv` 첫 실행이 이 함정에 걸렸다).

    python src/sim_src_upgrade/bench/bench_train.py --total_timesteps 50000 --n_envs 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

from sim_src_upgrade._paths import REPO  # noqa: E402

PYTHON = sys.executable
TRAINER = os.path.join(REPO, "src/rl_src/train_ppo_feature.py")
LAUNCHER = os.path.join(REPO, "src/sim_src_upgrade/drivers/run_fast.py")
DEFAULT_CFG = os.path.join(REPO, "scenarios/exp_시도/osrm/exp_서울_osrm/(37.5666,126.9784)/config_(37.5666,126.9784).yaml")


def run(mode: str, log_dir: str, args) -> tuple[float, str]:
    shutil.rmtree(log_dir, ignore_errors=True)
    common = ["--config_path", args.config_path,
              "--total_timesteps", str(args.total_timesteps),
              "--n_envs", str(args.n_envs), "--n_steps", str(args.n_steps),
              "--batch_size", str(args.batch_size), "--vec", "subproc",
              "--seed", str(args.seed), "--extractor", "pointer", "--log_dir", log_dir]
    cmd = ([PYTHON, TRAINER, *common] if mode == "old"
           else [PYTHON, LAUNCHER, "--target", "train_ppo_feature", "--skip_preflight", "--", *common])
    env = dict(os.environ, MCI_OBS_VARIANT="essential+load+valid", MCI_H_PAD="47",
               MCI_CAP_GATE="occ",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    for k in ("MCI_FASTCORE", "MCI_FASTCORE_MASK_ONLY"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and "fastcore_boot" not in p)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print((p.stdout or "")[-1500:]); print((p.stderr or "")[-1500:])
        raise RuntimeError(f"{mode} 학습 실패 rc={p.returncode}")
    warn = "자식 프로세스 패치 실패" in (p.stderr or "")
    return dt, ("자식패치실패경고" if warn else "")


def compare_weights(d_old: str, d_fast: str):
    """두 학습 산출 모델의 파라미터를 텐서 단위로 대조 → (동일?, 다른수, 최대차이, 총수)."""
    sys.path.insert(0, os.path.join(REPO, "src/rl_src"))
    import torch as th
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401

    def load(d):
        m = MaskablePPO.load(os.path.join(d, "final_model.zip"), device="cpu")
        return {k: v.detach().cpu().clone() for k, v in m.policy.state_dict().items()}

    a, b = load(d_old), load(d_fast)
    if set(a) != set(b):
        return False, -1, float("nan"), len(a)
    n_diff, dmax = 0, 0.0
    for k in a:
        if not th.equal(a[k], b[k]):
            n_diff += 1
            dmax = max(dmax, float((a[k] - b[k]).abs().max()))
    return n_diff == 0, n_diff, dmax, len(a)


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 wall-clock 배속 실측 + 결과 동일성 대조")
    ap.add_argument("--config_path", default=DEFAULT_CFG)
    ap.add_argument("--total_timesteps", type=int, default=50_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default=os.path.join(REPO, "results/sim_upgrade/train_bench"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    print(f"[학습 배속] {args.total_timesteps} steps × n_envs {args.n_envs} "
          f"(vec=subproc, 스레드 양쪽 1로 고정)", flush=True)

    d_old = os.path.join(args.work, "old")
    d_fast = os.path.join(args.work, "fast")
    t_old, w1 = run("old", d_old, args)
    print(f"  구 코어 {t_old:.1f}s {w1}", flush=True)
    t_fast, w2 = run("fast", d_fast, args)
    print(f"  신 코어 {t_fast:.1f}s {w2}", flush=True)

    # ★속도만 재고 결과를 안 보면 스스로를 속인다 — 가중치 대조를 벤치에 붙박이로 넣는다.
    same, n_diff, dmax, n = compare_weights(d_old, d_fast)
    print(f"\n[결과 동일성] 파라미터 텐서 {n}개 중 다름 {n_diff}개, 최대차이 {dmax:g} "
          f"→ {'PASS' if same else 'FAIL'}")
    if not same:
        print("  ⚠️ 가중치가 다르다. 배속 수치를 인용하기 전에 원인을 규명하라.\n"
              "     (양쪽 BLAS 스레드 수가 같은지 먼저 확인 — 다르면 코어와 무관하게 갈린다)")

    sp = t_old / t_fast
    print(f"\n[학습 배속] {sp:.2f}x  ({t_old:.1f}s → {t_fast:.1f}s)")
    print(f"  스텝당: {t_old/args.total_timesteps*1000:.3f} → "
          f"{t_fast/args.total_timesteps*1000:.3f} ms/step")
    print("  ※ 고정 오버헤드(약 30s)가 양쪽에 동일하게 포함 → 10M 규모에선 이보다 배속이 커진다")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "old_s": t_old, "fast_s": t_fast,
                       "speedup": sp}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
