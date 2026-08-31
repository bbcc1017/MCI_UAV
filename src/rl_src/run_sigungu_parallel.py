# -*- coding: utf-8 -*-
"""시군구별 지역특화 PPO 교사 250개를 슬롯 예산 안에서 병렬 학습한다 (v18 E5).

``run_grid_parallel.py`` 의 슬롯 세마포어·stderr 회수 구조를 승계하되 두 가지가 다르다.

1. **호출 대상이 ``train_ppo_feature.py`` 다.** 구 드라이버는 ``train_ppo.py``(flat obs,
   pointer/pdrwog/H_PAD 인자 없음)를 부르므로 v10 레시피를 재현할 수 없다.
2. **동시성 예산이 명목 코어가 아니라 실측 loadavg 다.** CLAUDE.md 실측으로 학습 런당
   loadavg ≈ 1.6 이다(env 워커 대부분이 파이프 대기). ``n_envs+1`` 명목 슬롯으로 잡으면
   약 6배 과대추정한다. 여기서는 ``--max_jobs`` 로 **동시 런 수**를 직접 제한하고,
   ``--loadavg_cap`` 으로 co-tenant 를 포함한 실제 부하 상한을 함께 건다.

아키텍처는 **v10 레시피 그대로**(``n_attn_blocks 1``)다. v12 에서 attention 제거가 더 좋았지만
여기서는 전국모델 ``v10_random4_1000_pointer_s0`` 와의 대각 비교가 목적이라 아키텍처를
맞춰 지역화 효과만 분리한다.

GPU 는 co-tenant 가 점유 중이므로 기본 CPU 강제다(``train_ppo_feature.py`` 는 device 를
지정하지 않아 SB3 auto = CUDA 가 된다 — 수십 개 동시 CUDA 컨텍스트는 스래싱).

사용
----
    python src/rl_src/run_sigungu_parallel.py --wave 1 --total_timesteps 10000000 \
        --max_jobs 16 --tag wave1
    python src/rl_src/run_sigungu_parallel.py --wave 2 --total_timesteps 2000000 \
        --max_jobs 48 --tag wave2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "scenarios/manifests/sigungu250/_index.json"
LOG_ROOT = REPO / "results/rl/sigungu250"
PYBIN = os.environ.get("MCI_PYBIN", "/home/ryu/anaconda3/envs/UAV/bin/python")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(INDEX))
    p.add_argument("--wave", type=int, default=0, help="0=전체, 1/2=_index.json 의 wave")
    p.add_argument("--regions", default="", help="쉼표 구분. 주면 wave 무시")
    p.add_argument("--holdout_train3", action="store_true",
                   help="p3 를 뺀 3좌표 매니페스트로 학습(Wave1 스텝예산 내부검증용)")
    p.add_argument("--total_timesteps", type=int, default=10_000_000)
    p.add_argument("--checkpoint_freq", type=int, default=5_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="w")
    p.add_argument("--log_root", default=str(LOG_ROOT))
    p.add_argument("--max_jobs", type=int, default=16, help="동시 학습 런 수")
    p.add_argument("--loadavg_cap", type=float, default=110.0,
                   help="1분 loadavg 가 이 값을 넘으면 새 런을 띄우지 않는다")
    p.add_argument("--gpu", action="store_true", help="CUDA 허용(기본은 CPU 강제)")
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def build_cmd(manifest: str, log_dir: str, a) -> list[str]:
    """v10 레시피 (results/rl/redesign/v10_random4_1000_pointer_s0/meta.json 기준)."""
    return [PYBIN, str(REPO / "src/rl_src/train_ppo_feature.py"),
            "--config_path", manifest,
            "--total_timesteps", str(a.total_timesteps),
            "--n_envs", str(a.n_envs), "--vec", "subproc",
            "--seed", str(a.seed),
            "--log_dir", log_dir,
            "--extractor", "pointer",
            "--reward_mode", "pdrwog", "--norm_reward",
            "--learning_rate", "3e-4", "--lr_anneal",
            "--target_kl", "0.03", "--n_epochs", "5",
            "--n_steps", "512", "--batch_size", "512",
            "--embed_dim", "64", "--ctx_dim", "128", "--head_hidden", "128",
            "--n_attn_blocks", "1",
            "--checkpoint_freq", str(a.checkpoint_freq)]


def main() -> None:
    a = parse_args()
    idx = json.load(open(a.index, encoding="utf-8"))
    regs = idx["regions"]
    if a.regions:
        keys = [r for r in a.regions.split(",") if r in regs]
    elif a.wave:
        keys = [r for r, v in regs.items() if v["wave"] == a.wave]
    else:
        keys = list(regs)
    keys.sort()
    if not keys:
        raise SystemExit("실행할 지역이 없다")

    if a.holdout_train3 and not idx.get("holdout_coord"):
        raise SystemExit("_index.json 에 holdout 좌표가 없다 — split 을 --holdout p3 로 다시 돌려라")

    root = Path(a.log_root) / a.tag
    (root / "run_logs").mkdir(parents=True, exist_ok=True)

    # 학습 환경 — MCI_OBS_VARIANT 는 train_ppo_feature 가 스스로 설정하지 않는다(호출자 책임).
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8",
               MCI_OBS_VARIANT="essential+load+valid", MCI_H_PAD="47", MCI_CAP_GATE="occ")
    if not a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"

    jobs = []
    for r in keys:
        man = regs[r]["manifest_train3"] if a.holdout_train3 else regs[r]["manifest"]
        log_dir = str(root / r)
        if (Path(log_dir) / "final_model.zip").exists():
            continue                                   # skip-done (재개 가능)
        jobs.append({"region": r, "manifest": man, "log_dir": log_dir})

    print(f"[sigungu] wave={a.wave or 'all'} 대상 {len(keys)} · 남은 잡 {len(jobs)} · "
          f"steps={a.total_timesteps:,} · max_jobs={a.max_jobs} · "
          f"device={'GPU' if a.gpu else 'CPU강제'} · train3={a.holdout_train3}")
    if a.dry_run:
        for j in jobs[:3]:
            print("  ", " ".join(build_cmd(j["manifest"], j["log_dir"], a)))
        return
    if not jobs:
        print("[sigungu] 전부 완료 상태")
        return

    pending, running, done, failed = list(jobs), [], [], []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.max_jobs and os.getloadavg()[0] < a.loadavg_cap:
            j = pending.pop(0)
            Path(j["log_dir"]).mkdir(parents=True, exist_ok=True)
            errp = root / "run_logs" / f"{j['region']}.err"
            fe = open(errp, "w", encoding="utf-8")
            p = subprocess.Popen(build_cmd(j["manifest"], j["log_dir"], a),
                                 cwd=str(REPO), env=env,
                                 stdout=subprocess.DEVNULL, stderr=fe)
            running.append({**j, "proc": p, "fe": fe, "t0": time.time()})
            time.sleep(1.0)                            # 동시 기동 스파이크 완화
        time.sleep(a.poll)
        for it in list(running):
            rc = it["proc"].poll()
            if rc is None:
                continue
            it["fe"].close()
            running.remove(it)
            (done if rc == 0 else failed).append(it["region"])
            if rc != 0:
                print(f"  ✗ {it['region']} rc={rc} → {root/'run_logs'/(it['region']+'.err')}")
            el = (time.time() - t0) / 3600
            print(f"  [{len(done)+len(failed)}/{len(jobs)}] {it['region']} rc={rc} "
                  f"({(time.time()-it['t0'])/60:.1f}분) 누적 {el:.2f}h "
                  f"load={os.getloadavg()[0]:.0f}", flush=True)

    summary = {"tag": a.tag, "wave": a.wave, "finished": datetime.now().isoformat(timespec="seconds"),
               "total_timesteps": a.total_timesteps, "holdout_train3": a.holdout_train3,
               "n_done": len(done), "n_failed": len(failed), "failed": failed,
               "wall_hours": round((time.time() - t0) / 3600, 3)}
    (root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[sigungu] 완료 {len(done)} · 실패 {len(failed)} · {summary['wall_hours']}h → {root}")


if __name__ == "__main__":
    main()
