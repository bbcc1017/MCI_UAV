"""v6 B1 — NCRP 라벨 예측가능성 probe — "obs 로 NCRP 라벨을 맞출 수 있는가".

ncrp_labels.py 가 모은 (obs, NCRP 라벨, mask, switched) 데이터셋을 받아 **P4(ExIt-online)
착수 전 관문**을 잰다: 챔피언 아키텍처 복제본에 라벨을 짧게 BC 주입하고, held-out 에서
top-1 acc 를 **전체 / switched 한정 / non-switched 한정**으로 분리 측정한다.

왜 switched 한정 acc 가 핵심인가:
  - non-switched 샘플은 라벨=greedy=챔피언 argmax → 챔피언이 이미(BC 전에도) 정확히 맞힘
    (acc≈1). 예측가능성 신호가 없다.
  - switched 샘플은 라벨≠greedy → 챔피언 argmax 는 BC 전 acc=0(정의상). BC 후 이 acc 가
    올라가면 "플래너의 greedy 이탈이 obs 에 예측 정보로 남아 있다"는 뜻 = 반응형 정책으로
    증류 가능 → P4 유망. 안 오르면(v3 천리안 라벨 0.19 처럼) 구조적 천장.
  - chance(=1/유효행동수) 대비 배율로 '무작위보다 얼마나 나은가'를 정규화해 본다.

재사용 의존: train_ppo_bc.bc_pretrain(masked NLL — 학습 단계 그대로), exit_distill.
{policy_probs,kl_mean}(분포·drift 진단), pointer_policy/hospital_set_extractor(MaskablePPO.
load 전 import — 역직렬화 필수).

설계 결정:
  - **지역 단위 분할(--split region, 기본)**: held-out 지역을 통째로 빼서 "새 지역에서
    라벨 예측"을 잰다. 챔피언은 전국 단일 정책이고 배포도 전국이므로, 같은 지역 상태가
    train·val 에 섞여 근접중복으로 acc 가 부풀려지는 누수를 막고 일반화를 본다(deployment
    관련 질문). 지역 수 <2 면 결정 단위(--split decision)로 자동 폴백. 결정 단위는 in-dist
    예측력을 재는 완화 지표(허용).
  - lr 강제: 챔피언(v4_plr2)은 lr_anneal 로 lr≈0 저장 → BC 전 optimizer lr 을 --lr 로
    주입(안 하면 BC no-op — exit_distill 동일 함정).
  - go/no-go 임계는 출력하지 않는다(판정은 메인). 수치만 사람용 요약 + --out_json 으로.

예: PYTHONIOENCODING=utf-8 python src/rl_src/ncrp_label_probe.py \
    --labels results/rl/redesign/ncrp_labels_v6.pkl --probe_epochs 2 --device cpu \
    --out_json results/rl/redesign/ncrp_probe_v6.json
"""
import os
import argparse
import json
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _acc(probs, acts):
    """top-1 acc = mean(argmax(probs) == acts). 빈 배열이면 NaN."""
    if len(acts) == 0:
        return float("nan")
    return float((probs.argmax(axis=1) == acts).mean())


def _safe(x):
    """JSON 직렬화용: 비유한(NaN/inf) → None."""
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except (TypeError, ValueError):
        return None


def split_indices(regions, switched, val_frac, mode, seed=0):
    """train/val 인덱스 분할. mode='region' 이면 held-out 지역을 통째로(일반화 측정),
    'decision' 이면 결정 단위 무작위. 지역 <2 개면 region→decision 자동 폴백."""
    N = len(regions)
    uniq = sorted(set(regions))
    rng = np.random.default_rng(seed)
    used_mode = mode
    if mode == "region" and len(uniq) >= 2:
        n_val_reg = max(1, int(round(val_frac * len(uniq))))
        n_val_reg = min(n_val_reg, len(uniq) - 1)  # train 지역 ≥1 보장
        perm = rng.permutation(len(uniq))
        val_regions = {uniq[i] for i in perm[:n_val_reg]}
        reg_arr = np.asarray(regions)
        val_mask = np.isin(reg_arr, list(val_regions))
        val_idx = np.flatnonzero(val_mask)
        train_idx = np.flatnonzero(~val_mask)
    else:
        used_mode = "decision"
        perm = rng.permutation(N)
        n_val = max(1, int(round(val_frac * N)))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx, used_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ncrp_labels.py 출력 pickle")
    ap.add_argument("--model_dir", default="",
                    help="챔피언 디렉터리(생략 시 pkl meta.model_dir 사용)")
    ap.add_argument("--probe_epochs", type=int, default=2, help="BC 프로브 epoch 수")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4, help="BC lr(옵티마이저에 강제 주입)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split", choices=["region", "decision"], default="region",
                    help="held-out 분할 축(region=지역단위 일반화, decision=결정단위 in-dist)")
    ap.add_argument("--device", default="cpu", help="cpu(공유박스 안전) 또는 cuda")
    ap.add_argument("--threads", type=int, default=4, help="torch intra-op 스레드(박스 예의)")
    ap.add_argument("--out_json", default="", help="머신 리포트 JSON 경로(생략 시 미저장)")
    A = ap.parse_args()

    # ---- 라벨 로드 ----
    with open(A.labels, "rb") as f:
        d = pickle.load(f)
    obs = d["obs"].astype(np.float32)
    acts = d["actions"].astype(np.int64)
    masks = d["masks"].astype(bool)
    regions = list(d["regions"])
    greedy = d.get("greedy_actions")
    switched = d.get("switched")
    if switched is None:
        # 부가키 없으면 라벨≠greedy 로 재구성(구 형식 방어)
        switched = (acts != greedy) if greedy is not None else np.zeros(len(acts), bool)
    switched = np.asarray(switched, dtype=bool)
    meta = d.get("meta", {})
    N = len(acts)
    n_valid = masks.sum(axis=1).astype(np.float64)  # 샘플당 유효행동 수(≥2)

    # ---- 통계 섹션(BC 불요) ----
    chance_all = float(np.mean(1.0 / n_valid))
    switch_rate = float(switched.mean())
    agree = float((acts == greedy).mean()) if greedy is not None else float("nan")
    # 지역별 switch율 top5
    per_reg = {}
    reg_arr = np.asarray(regions)
    for rg in sorted(set(regions)):
        sel = reg_arr == rg
        per_reg[rg] = float(switched[sel].mean())
    top5 = sorted(per_reg.items(), key=lambda kv: -kv[1])[:5]

    print("=" * 72)
    print(f"[probe] 라벨 = {A.labels}")
    print(f"[probe] meta: K={meta.get('K')} h={meta.get('h')} m={meta.get('m')} "
          f"clairvoyant={meta.get('clairvoyant')} git={meta.get('git_sha')} "
          f"model_dir={meta.get('model_dir')}")
    print(f"[probe] 총 결정(샘플) N={N}  지역={len(set(regions))}")
    print(f"[probe] switch율 전체 = {switch_rate:.3f}  (n_switch={int(switched.sum())})")
    print(f"[probe] 평균 유효행동수 = {n_valid.mean():.2f}  → chance acc = {chance_all:.3f}")
    print(f"[probe] greedy-라벨 agreement = {agree:.3f}  (= 1 − switch율 정합확인)")
    print("[probe] 지역별 switch율 top5:")
    for rg, v in top5:
        print(f"    {rg}: {v:.3f}")

    # ---- 분할 ----
    train_idx, val_idx, used_split = split_indices(regions, switched, A.val_frac, A.split)
    sw_val = switched[val_idx]
    nv_val = n_valid[val_idx]
    print(f"[probe] 분할({A.split}→{used_split}): train N={len(train_idx)} "
          f"({len(set(reg_arr[train_idx]))}지역), val N={len(val_idx)} "
          f"({len(set(reg_arr[val_idx]))}지역), val_switched={int(sw_val.sum())}")

    # ---- 챔피언 로드(역직렬화 import 필수) ----
    import torch as th
    th.set_num_threads(max(1, A.threads))
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from sb3_contrib import MaskablePPO
    from train_ppo_bc import bc_pretrain
    from exit_distill import policy_probs, kl_mean

    model_dir = A.model_dir or meta.get("model_dir")
    if not model_dir:
        raise SystemExit("model_dir 미상 — --model_dir 지정 또는 pkl meta.model_dir 필요")
    model_zip = model_dir if model_dir.endswith(".zip") else os.path.join(model_dir, "final_model.zip")
    model = MaskablePPO.load(model_zip, device=A.device)
    policy, device = model.policy, model.device
    obs_dim_m, n_act_m = int(model.observation_space.shape[0]), int(model.action_space.n)
    if d["obs_dim"] != obs_dim_m or d["n_actions"] != n_act_m:
        raise ValueError(f"라벨/모델 차원 불일치 — obs {d['obs_dim']} vs {obs_dim_m}, "
                         f"act {d['n_actions']} vs {n_act_m} (obs variant/champion 확인)")

    obs_val, act_val, mask_val = obs[val_idx], acts[val_idx], masks[val_idx]

    # ---- BC 전: held-out 분포·acc(switched 는 정의상 acc≈0) ----
    probs_pre = policy_probs(policy, obs_val, mask_val, device)
    acc_pre = _acc(probs_pre, act_val)

    # ---- BC(masked NLL — train split 만): lr 강제 후 bc_pretrain ----
    for g in policy.optimizer.param_groups:
        g["lr"] = A.lr
    train_ds = {"obs": obs[train_idx], "actions": acts[train_idx], "masks": masks[train_idx],
                "obs_dim": obs_dim_m, "n_actions": n_act_m}
    bc_pretrain(model, train_ds, epochs=A.probe_epochs, batch_size=A.batch_size, device=device)

    # ---- BC 후: held-out acc(전체/switched/non-switched) + KL(pre‖post) ----
    probs_post = policy_probs(policy, obs_val, mask_val, device)
    acc_post = _acc(probs_post, act_val)
    acc_sw = _acc(probs_post[sw_val], act_val[sw_val])
    acc_nsw = _acc(probs_post[~sw_val], act_val[~sw_val])
    probs_tr = policy_probs(policy, obs[train_idx], masks[train_idx], device)
    acc_train = _acc(probs_tr, acts[train_idx])  # 과적합 진단(train vs val 격차)
    chance_val = float(np.mean(1.0 / nv_val)) if len(nv_val) else float("nan")
    chance_sw = float(np.mean(1.0 / nv_val[sw_val])) if sw_val.any() else float("nan")
    ratio_sw = (acc_sw / chance_sw) if (sw_val.any() and chance_sw > 0) else float("nan")
    kl = kl_mean(probs_pre, probs_post)

    print("-" * 72)
    print(f"[probe] BC {A.probe_epochs}ep (lr={A.lr:g}, split={used_split})")
    print(f"[probe] held-out top-1 acc:  전체 {acc_pre:.3f}(pre) → {acc_post:.3f}(post)")
    print(f"[probe]   switched 한정 acc(post)     = {acc_sw:.3f}  "
          f"(n={int(sw_val.sum())}, chance {chance_sw:.3f}, 배율 {ratio_sw:.2f}×)")
    print(f"[probe]   non-switched 한정 acc(post) = {acc_nsw:.3f}  (n={int((~sw_val).sum())})")
    print(f"[probe]   train acc(post)             = {acc_train:.3f}  (과적합 진단용)")
    print(f"[probe] KL(pre‖post) = {kl:.4f} nat  |  전체 chance(val) = {chance_val:.3f}")
    print("=" * 72)

    report = {
        "labels": A.labels, "meta": meta,
        "n_samples": N, "n_regions": len(set(regions)),
        "switch_rate_overall": _safe(switch_rate),
        "switch_rate_top5": {k: _safe(v) for k, v in top5},
        "mean_n_valid": _safe(n_valid.mean()),
        "chance_acc_overall": _safe(chance_all),
        "greedy_label_agreement": _safe(agree),
        "bc_probe": {
            "split_requested": A.split, "split_used": used_split, "val_frac": A.val_frac,
            "probe_epochs": A.probe_epochs, "lr": A.lr, "device": str(device),
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "n_train_regions": int(len(set(reg_arr[train_idx]))),
            "n_val_regions": int(len(set(reg_arr[val_idx]))),
            "n_val_switched": int(sw_val.sum()),
            "acc_overall_pre": _safe(acc_pre),
            "acc_overall_post": _safe(acc_post),
            "acc_switched_post": _safe(acc_sw),
            "acc_nonswitched_post": _safe(acc_nsw),
            "acc_train_post": _safe(acc_train),
            "chance_acc_val": _safe(chance_val),
            "chance_acc_switched_val": _safe(chance_sw),
            "switched_acc_over_chance": _safe(ratio_sw),
            "kl_pre_post": _safe(kl),
        },
    }
    if A.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(A.out_json)) or ".", exist_ok=True)
        with open(A.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[probe] JSON 저장: {A.out_json}")


if __name__ == "__main__":
    main()
