"""G8 학습 경로 동치 게이트 — SB3 학습 산출 모델 가중치가 **비트동일**한지 본다.

학습은 `SubprocVecEnv`(기본 forkserver)를 쓰므로 런처의 몽키패치가 자식에 상속되지 않고,
`fastcore_boot/sitecustomize.py` 가 자식 부팅 시 다시 패치를 건다. 그 배선이 실제로 먹었는지,
그리고 결과가 변하지 않았는지를 **최종 모델 파라미터 텐서 단위**로 확인한다.

절차
----
1. **재현성 기준선**: 구 코어로 같은 시드 2회 학습 → 가중치가 같아야 한다.
   (같지 않으면 학습 자체가 비결정적이라 이 게이트로는 아무것도 판정할 수 없다 → 즉시 중단)
2. **동치**: 고속 코어로 같은 시드 학습 → 구 코어 가중치와 비교.

    python src/sim_src_upgrade/verify/verify_train_equiv.py --total_timesteps 2048 --n_envs 2
"""
from __future__ import annotations

import argparse
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


def train(mode: str, log_dir: str, cfg: str, steps: int, n_envs: int, seed: int, extra: list[str]) -> float:
    """mode='old'|'fast' 로 학습 1회. 반환 = wall 초."""
    shutil.rmtree(log_dir, ignore_errors=True)
    common = ["--config_path", cfg, "--total_timesteps", str(steps),
              "--n_envs", str(n_envs), "--n_steps", "256", "--batch_size", "128",
              "--vec", "subproc", "--seed", str(seed), "--extractor", "pointer",
              "--log_dir", log_dir, *extra]
    if mode == "old":
        cmd = [PYTHON, TRAINER, *common]
    else:
        cmd = [PYTHON, LAUNCHER, "--target", "train_ppo_feature", "--skip_preflight", "--", *common]
    # ★BLAS 스레드 수를 양쪽 **똑같이** 고정해야 한다.
    #   torch 의 병렬 축소는 스레드 수에 따라 누산 순서가 달라져 마지막 비트가 변한다.
    #   `run_fast.py` 는 import 시 스레드를 1로 핀하는데 원본 실행은 안 핀하므로,
    #   맞춰주지 않으면 "코어 차이"가 아니라 "스레드 차이"를 재게 된다(첫 G8 실행이 그랬다:
    #   가중치 maxΔ 1.17e-4 · 소요 32× 라는 비현실적 배속이 그 증거였다).
    env = dict(os.environ, MCI_OBS_VARIANT="essential+load+valid", MCI_H_PAD="47",
               MCI_CAP_GATE="occ", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    # 구 코어 실행에 이전 실행의 전파 설정이 새지 않도록 청소
    for k in ("MCI_FASTCORE", "MCI_FASTCORE_MASK_ONLY"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and "fastcore_boot" not in p)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print((p.stdout or "")[-2000:])
        print((p.stderr or "")[-2000:])
        raise RuntimeError(f"{mode} 학습 실패 rc={p.returncode}")
    if mode == "fast":
        # sitecustomize 가 자식에 실제로 걸렸는지 흔적 확인
        if "[fastcore] 자식 프로세스 패치 실패" in (p.stderr or ""):
            raise RuntimeError("자식 패치 실패 경고 발견 — 전파 배선 확인 필요")
    return dt


def weights(log_dir: str):
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401

    m = MaskablePPO.load(os.path.join(log_dir, "final_model.zip"), device="cpu")
    return {k: v.detach().cpu().clone() for k, v in m.policy.state_dict().items()}


def cmp_weights(a, b, label_a, label_b) -> tuple[bool, float, int]:
    import torch as th

    if set(a) != set(b):
        print(f"  파라미터 키 불일치: {set(a) ^ set(b)}")
        return False, float("nan"), 0
    dmax = 0.0
    n_diff = 0
    for k in a:
        if not th.equal(a[k], b[k]):
            n_diff += 1
            dmax = max(dmax, float((a[k] - b[k]).abs().max()))
    print(f"  {label_a} vs {label_b}: 텐서 {len(a)}개 중 다름 {n_diff}개, 최대차이 {dmax:g}")
    return n_diff == 0, dmax, len(a)


def main() -> int:
    ap = argparse.ArgumentParser(description="G8 학습 경로 동치 게이트")
    ap.add_argument("--config_path", default=DEFAULT_CFG)
    ap.add_argument("--total_timesteps", type=int, default=2048)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default=os.path.join(REPO, "results/sim_upgrade/train_equiv"))
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO, "src/rl_src"))
    os.makedirs(args.work, exist_ok=True)
    dirs = {k: os.path.join(args.work, k) for k in ("old1", "old2", "fast")}

    print(f"[G8] {args.total_timesteps} steps × n_envs {args.n_envs} (vec=subproc/forkserver), "
          f"seed {args.seed}")
    t_old1 = train("old", dirs["old1"], args.config_path, args.total_timesteps,
                   args.n_envs, args.seed, args.extra)
    t_old2 = train("old", dirs["old2"], args.config_path, args.total_timesteps,
                   args.n_envs, args.seed, args.extra)
    w1, w2 = weights(dirs["old1"]), weights(dirs["old2"])
    print("\n[1] 재현성 기준선 (구 코어 2회)")
    repro, _d, _n = cmp_weights(w1, w2, "old1", "old2")
    if not repro:
        print("[G8] 판정 불가 — 학습이 같은 시드에서도 재현되지 않는다(비결정적). "
              "가중치 대조로는 코어 동치를 판정할 수 없다.")
        return 2

    t_fast = train("fast", dirs["fast"], args.config_path, args.total_timesteps,
                   args.n_envs, args.seed, args.extra)
    wf = weights(dirs["fast"])
    print("\n[2] 코어 동치 (구 vs 신)")
    same, _d, n = cmp_weights(w1, wf, "old", "fast")

    print(f"\n소요: old {t_old1:.1f}s / {t_old2:.1f}s, fast {t_fast:.1f}s "
          f"→ 배속 {(t_old1+t_old2)/2/t_fast:.2f}x")
    print(f"[G8] {'PASS — 학습 산출 가중치 비트동일' if same else 'FAIL'} (파라미터 텐서 {n}개)")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
