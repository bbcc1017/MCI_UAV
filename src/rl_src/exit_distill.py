"""ExIt-lite 재증류 (성능트랙 S4 — 2단계) — 오라클 라벨을 champion 정책에 BC 주입.

champion MaskablePPO(포인터 head 포함 — MaskableActorCriticPolicy 계열이면 무엇이든)를
로드해 exit_labels.py 데이터셋으로 masked-NLL BC(train_ppo_bc.bc_pretrain 그대로 재사용)
후 저장한다. 이후 PPO 미세조정(→L4)은 기존 train_ppo_feature.py --resume_from <out_dir>
로 이어간다(기존 코드 무수정).

재사용 의존: train_ppo_bc.{load_bc_dataset,bc_pretrain}(evaluate_actions masked NLL),
pointer_policy/hospital_set_extractor(MaskablePPO.load 전 import — 역직렬화 필수).

설계 결정:
  - champion 이 --lr_anneal 로 학습됐으면 로드 시점 옵티마이저 lr≈0 → BC 전에
    optimizer.param_groups lr 을 --lr 로 강제(안 하면 BC 가 사실상 no-op).
  - PPO 미세조정 lr: train_ppo_feature.py 의 --resume_from 경로는 --learning_rate 인자를
    **무시**하고 zip 에 저장된 learning_rate 로 lr_schedule 을 복원(_setup_lr_schedule)
    → 여기서 저장 직전에 model.learning_rate = --finetune_lr(상수) 로 교체해 zip 에 굽는다.
    resume 학습은 상수 finetune_lr 로 진행된다(0 이면 미변경 = champion 스케줄 유지).
  - 검증 지표: BC 전/후 데이터셋 라벨 top-1 acc + 같은 obs 배치(최대 --kl_batch 샘플)에서
    KL(pre‖post) — 과도 drift 감지(스위치율 ~25% 데이터면 KL 이 수~수십 nat 로 치솟으면 의심).
  - out_dir 산출물: final_model.zip + vecnormalize.pkl(champion 것 복사 — BC 는 obs 통계를
    바꾸지 않음) + exit_distill_meta.json(라벨 수·epochs·acc 전후·KL).

예(스모크): PYTHONIOENCODING=utf-8 python src/rl_src/exit_distill.py \
    --model_dir results/rl/redesign/L3_pointer_s0 --dataset /tmp/exit_smoke.pkl \
    --epochs 1 --batch_size 512 --lr 3e-4 --device cuda --out_dir /tmp/exit_smoke_model
이후 PPO 미세조정(예):
    MCI_OBS_VARIANT=essential+load MCI_CAP_GATE=occ python src/rl_src/train_ppo_feature.py \
      --config_path <champion 학습 매니페스트> --extractor pointer --reward_mode pdrwog \
      --norm_reward --resume_from <out_dir> --total_timesteps 2000000 \
      --vec subproc --n_envs 8 --log_dir results/rl/redesign/L4_exit_s0
    (--learning_rate 는 resume 시 무시됨 — lr 은 이 스크립트의 --finetune_lr 로 이미 구움)
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


def policy_probs(policy, obs_np, mask_np, device, batch=1024):
    """마스크 적용 정책분포 probs (N, n_actions) — 배치 순회, no_grad."""
    import torch as th
    policy.set_training_mode(False)
    outs = []
    with th.no_grad():
        for i in range(0, len(obs_np), batch):
            ob = th.as_tensor(obs_np[i:i + batch], device=device)
            mk = th.as_tensor(mask_np[i:i + batch], device=device)
            dist = policy.get_distribution(ob, action_masks=mk)
            outs.append(dist.distribution.probs.cpu().numpy())
    return np.concatenate(outs, axis=0)


def kl_mean(p_pre, p_post, eps=1e-12):
    """평균 KL(pre‖post). 마스크 밖 확률은 양쪽 다 ≈0 → p_pre>eps 가드로 기여 0."""
    lp = np.log(np.maximum(p_pre, eps)) - np.log(np.maximum(p_post, eps))
    return float(np.where(p_pre > eps, p_pre * lp, 0.0).sum(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="champion 디렉터리(final_model.zip+vecnormalize.pkl) 또는 zip 경로")
    ap.add_argument("--dataset", required=True, help="exit_labels.py 출력 pickle")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4, help="BC 단계 lr(옵티마이저에 강제 주입)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--finetune_lr", type=float, default=1e-4,
                    help="저장 zip 에 구울 PPO 미세조정 상수 lr(resume 시 적용). 0=미변경")
    ap.add_argument("--kl_batch", type=int, default=4096, help="KL 측정 샘플 수 상한")
    ap.add_argument("--out_dir", required=True)
    A = ap.parse_args()

    # 역직렬화용 import (MaskablePPO.load 전 필수 — pointer/deepsets 챔피언 모두 대비)
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from sb3_contrib import MaskablePPO
    from train_ppo_bc import load_bc_dataset, bc_pretrain

    model_zip = A.model_dir if A.model_dir.endswith(".zip") \
        else os.path.join(A.model_dir, "final_model.zip")
    model = MaskablePPO.load(model_zip, device=A.device)
    policy, device = model.policy, model.device

    d = load_bc_dataset(A.dataset)
    obs_dim_m = int(model.observation_space.shape[0])
    n_act_m = int(model.action_space.n)
    if d["obs_dim"] != obs_dim_m or d["n_actions"] != n_act_m:
        raise ValueError(f"데이터셋/모델 차원 불일치 — obs {d['obs_dim']} vs {obs_dim_m}, "
                         f"act {d['n_actions']} vs {n_act_m} (champion·라벨 수집 obs 변형 확인)")
    obs_np = d["obs"].astype(np.float32)
    act_np = d["actions"].astype(np.int64)
    mask_np = d["masks"].astype(bool)
    N = len(act_np)
    sw = d.get("switch_rate", None)
    print(f"[distill] champion={model_zip} device={device}", flush=True)
    print(f"[distill] 라벨 N={N} obs_dim={d['obs_dim']} n_actions={d['n_actions']}"
          + (f" 수집시 스위치율={sw:.3f}" if sw is not None else ""), flush=True)

    # ---- BC 전 지표: full acc + KL 기준 분포(고정 샘플) ----
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(A.kl_batch, N), replace=False)
    probs_pre = policy_probs(policy, obs_np, mask_np, device)
    acc_pre = float((probs_pre.argmax(axis=1) == act_np).mean())
    p_pre_kl = probs_pre[idx]

    # ---- BC (masked NLL — train_ppo_bc.bc_pretrain 재사용) ----
    # champion 이 lr_anneal 이면 로드된 옵티마이저 lr≈0 → BC lr 강제 주입
    for g in policy.optimizer.param_groups:
        g["lr"] = A.lr
    bc_pretrain(model, d, epochs=A.epochs, batch_size=A.batch_size, device=device)

    # ---- BC 후 지표 ----
    probs_post = policy_probs(policy, obs_np, mask_np, device)
    acc_post = float((probs_post.argmax(axis=1) == act_np).mean())
    kl = kl_mean(p_pre_kl, probs_post[idx])
    print(f"[distill] 라벨 top-1 acc: {acc_pre:.3f} → {acc_post:.3f} "
          f"(Δ{acc_post - acc_pre:+.3f}) | KL(pre‖post)={kl:.4f} nat (n={len(idx)})", flush=True)

    # ---- PPO 미세조정 lr 굽기(resume 가 --learning_rate 를 무시하므로 zip 에 저장) ----
    if A.finetune_lr > 0:
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = float(A.finetune_lr)
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        print(f"[distill] finetune lr={A.finetune_lr:g} (상수) 를 zip 에 저장 — "
              f"resume 시 _setup_lr_schedule 이 이 값으로 복원", flush=True)

    # ---- 저장: 모델 + vecnorm 복사 + 메타 ----
    os.makedirs(A.out_dir, exist_ok=True)
    out_zip = os.path.join(A.out_dir, "final_model.zip")
    model.save(out_zip)
    vn_src = os.path.join(os.path.dirname(model_zip), "vecnormalize.pkl")
    if os.path.exists(vn_src):
        shutil.copy2(vn_src, os.path.join(A.out_dir, "vecnormalize.pkl"))
    else:
        print(f"[distill] ⚠️ vecnormalize.pkl 미발견({vn_src}) — resume/eval 시 정규화 불일치 주의",
              flush=True)
    meta = {"champion": model_zip, "dataset": A.dataset, "n_labels": int(N),
            "epochs": A.epochs, "batch_size": A.batch_size, "bc_lr": A.lr,
            "finetune_lr": A.finetune_lr, "acc_pre": acc_pre, "acc_post": acc_post,
            "kl_pre_post": kl, "kl_n": int(len(idx)),
            "dataset_switch_rate": sw}
    with open(os.path.join(A.out_dir, "exit_distill_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[distill] 저장: {out_zip} (+vecnormalize.pkl, exit_distill_meta.json)", flush=True)


if __name__ == "__main__":
    main()
