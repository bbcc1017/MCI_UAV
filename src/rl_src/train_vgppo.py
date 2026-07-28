"""v7 Value-Guided PPO 파인튠 드라이버 (설계 §4·§10.1).

챔피언(v4_plr2 등)에서 resume 하여 NCRP 가치라벨로 크리틱을 유도하는 소규모 파인튠.
train_ppo_feature.py 의 resume 경로를 그대로 미러(make_env_fn 재사용)하되, MaskablePPO 대신
value_guided_ppo.ValueGuidedMaskablePPO 를 로드하고 set_value_guidance 로 aux value(+CRR) 손실을
상시 혼합한다. 산출: final_model.zip + vecnormalize.pkl + ev_history.csv(진단 곡선).

핵심 진단(설계 §8.3): ev_history.csv 의 `ev`(신규 롤아웃 크리틱 explained_variance)가
파인튠 동안 **상승**하면 "가치는 배운다"(핵심가설 1차 지지). 무변화면 배관버그(단위/타깃 연결).

단위환산(부록 A): 라벨 obs 는 챔피언 VecNormalize 정규화본 → 재정규화 금지(그대로 사용).
타깃 = q_best_disc(할인 suffix) / σ_ret(챔피언 sqrt(ret_rms.var)). load_value_labels 가 수행.
VecNormalize 는 **동결**(training=False, R8): 라벨이 챔피언 통계로 정규화됐고 σ_ret 도 그 스냅샷.

예(소규모):
  MCI_OBS_VARIANT=essential+load MCI_CAP_GATE=occ CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/rl_src/train_vgppo.py \
    --config_path scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json \
    --resume_from results/rl/redesign/v4_plr2_s0 \
    --value_labels results/rl/redesign/ncrp_value_labels.pkl \
    --aux_target q_best --aux_coef 0.5 --crr off \
    --total_timesteps 300000 --n_envs 4 --vec subproc \
    --log_dir results/rl/redesign/vg_a_probe_s0
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from train_ppo_feature import make_env_fn
from value_guided_ppo import ValueGuidedMaskablePPO, load_value_labels


class _EVFlush(BaseCallback):
    """매 롤아웃마다 model._ev_history 를 CSV 로 증분 flush — 학습 중 kl/EV 라이브 확인용
    (stdout=/dev/null 이라도 파일로 진행 감시). train() 이 append 하므로 1 업데이트 지연(무해)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        hist = getattr(self.model, "_ev_history", [])
        if hist:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                w.writeheader()
                w.writerows(hist)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True, help="챔피언 학습 매니페스트(.json) 또는 config(.yaml)")
    p.add_argument("--resume_from", required=True, help="챔피언 디렉터리(final_model.zip+vecnormalize.pkl)")
    p.add_argument("--value_labels", required=True, help="ncrp_value_labels.pkl(공유 스키마)")
    p.add_argument("--baseline", default="relative", choices=["relative", "absolute"],
                   help="value 타깃(설계 R4). relative(기본)=V_champ+dpdr_disc/σ(절단 리프 앵커=정답). "
                        "absolute=q_target_disc/σ(진단·절단 스케일편향 주의).")
    p.add_argument("--aux_target", default="q_best", choices=["q_greedy", "q_best", "q_exec"],
                   help="absolute 모드 타깃(relative 는 dpdr 사용, 무관).")
    p.add_argument("--aux_coef", type=float, default=0.5, help="aux value 회귀 계수(설계값 0.5).")
    p.add_argument("--crr", default="off", choices=["off", "binary", "exp"],
                   help="방법 B(CRR/AWAC) dpdr 가중 masked-NLL. 기본 off(A 단독).")
    p.add_argument("--crr_coef", type=float, default=0.05, help="CRR BC 계수(crr!=off 일 때).")
    p.add_argument("--crr_beta", type=float, default=0.05, help="exp 가중 온도 β(pdrwog 스케일).")
    p.add_argument("--crr_eps", type=float, default=5e-3,
                   help="binary 필터 임계 1[dpdr>eps](노이즈 스위치 컷). 0.005=switched~4.4%, 0.002=~9.4%.")
    p.add_argument("--total_timesteps", type=int, default=300_000, help="추가 스텝(resume 이어카운트).")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--finetune_lr", type=float, default=0.0,
                   help="파인튠 상수 lr(>0 이면 챔피언 복원 스케줄 대신 이 값). 0=챔피언 스케줄 유지.")
    p.add_argument("--freeze_obs_rms", action="store_true", default=True,
                   help="VecNormalize 통계 동결(기본 True, R8 — 라벨 정규화 일관성). --no_freeze 로 해제.")
    p.add_argument("--no_freeze", dest="freeze_obs_rms", action="store_false")
    p.add_argument("--log_dir", default="results/rl/redesign/vg_probe")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    # 챔피언 학습과 동일 보상모드 강제(Subproc 자식 전파). obs/게이트는 호출자가 env 로 설정(문서 예시).
    os.environ.setdefault("MCI_REWARD_MODE", "pdrwog")
    print(f"[vgppo] OBS_VARIANT={os.environ.get('MCI_OBS_VARIANT','(essential)')} "
          f"CAP_GATE={os.environ.get('MCI_CAP_GATE','occ')} REWARD={os.environ.get('MCI_REWARD_MODE')} "
          f"aux_target={args.aux_target} aux_coef={args.aux_coef} crr={args.crr}", flush=True)

    # ---- 역직렬화 import(pointer 챔피언) ----
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    try:
        from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    except Exception:
        pass

    # ---- env(챔피언과 동일 구성: make_env_fn 재사용) ----
    env_fns = [make_env_fn(args.config_path, seed=args.seed + i, rank=i, n_envs=args.n_envs)
               for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)

    # ---- 챔피언 VecNormalize 로드 + 동결(R8) ----
    model_zip = args.resume_from
    if os.path.isdir(model_zip):
        model_zip = os.path.join(model_zip, "final_model.zip")
    vn_path = os.path.join(os.path.dirname(model_zip), "vecnormalize.pkl")
    venv = VecNormalize.load(vn_path, venv)
    venv.training = not args.freeze_obs_rms   # 동결 시 obs_rms/ret_rms 갱신 중단(정규화는 계속 적용)
    venv.norm_reward = True
    sigma_ret = float(np.sqrt(venv.ret_rms.var + 1e-8))  # 챔피언 스냅샷(부록 A)
    print(f"[vgppo] VecNormalize 로드: training={venv.training}(동결={args.freeze_obs_rms}) "
          f"σ_ret={sigma_ret:.4f}", flush=True)

    # ---- 챔피언 정책/옵티마이저/num_timesteps 복원(ValueGuidedMaskablePPO 로 로드) ----
    model = ValueGuidedMaskablePPO.load(model_zip, env=venv,
                                        tensorboard_log=os.path.join(args.log_dir, "tb"))
    print(f"[vgppo] resume from {model_zip}: num_timesteps={model.num_timesteps} "
          f"(+{args.total_timesteps})", flush=True)
    if args.finetune_lr > 0:
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = float(args.finetune_lr)
        model.lr_schedule = get_schedule_fn(model.learning_rate)
        print(f"[vgppo] finetune lr={args.finetune_lr:g}(상수)", flush=True)

    # ---- 가치라벨 로드 + 단위환산 + 유도 설정 ----
    labels = load_value_labels(args.value_labels, model, aux_target=args.aux_target,
                               sigma_ret=sigma_ret, baseline=args.baseline,
                               crr=(None if args.crr == "off" else args.crr),
                               crr_beta=args.crr_beta, crr_eps=args.crr_eps)
    model.set_value_guidance(labels, aux_coef=args.aux_coef,
                             crr_coef=(0.0 if args.crr == "off" else args.crr_coef),
                             seed=args.seed)

    # ---- 파인튠 (ev_history 증분 flush 콜백으로 kl/EV 라이브 감시) ----
    ev_csv = os.path.join(args.log_dir, "ev_history.csv")
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="vgppo",
                reset_num_timesteps=False, progress_bar=False, callback=_EVFlush(ev_csv))

    # ---- 저장 + 진단 곡선 CSV ----
    final_path = os.path.join(args.log_dir, "final_model.zip")
    model.save(final_path)
    venv.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
    ev_csv = os.path.join(args.log_dir, "ev_history.csv")
    hist = getattr(model, "_ev_history", [])
    if hist:
        with open(ev_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)
        ev0, ev1 = hist[0]["ev"], hist[-1]["ev"]
        print(f"[vgppo] EV(신규롤아웃): {ev0:.4f} → {ev1:.4f} (Δ{ev1 - ev0:+.4f}) over {len(hist)} updates | "
              f"aux_value_loss {hist[0]['aux_value_loss']:.4f}→{hist[-1]['aux_value_loss']:.4f}", flush=True)
    # 요약 meta.json (stdout=/dev/null 로 돌려도 파일로 결과 회수 — sim print 스팸 회피)
    import json
    evs = [h["ev"] for h in hist] if hist else []
    axl = [h["aux_value_loss"] for h in hist] if hist else []
    meta = {"config_path": args.config_path, "resume_from": args.resume_from,
            "value_labels": args.value_labels, "baseline": args.baseline,
            "aux_target": args.aux_target, "aux_coef": args.aux_coef,
            "crr": args.crr, "crr_coef": args.crr_coef, "sigma_ret": sigma_ret,
            "total_timesteps": args.total_timesteps, "n_envs": args.n_envs,
            "final_num_timesteps": int(model.num_timesteps),
            "labels_meta": labels["meta"], "n_updates": len(hist),
            "ev_first": (evs[0] if evs else None), "ev_last": (evs[-1] if evs else None),
            "ev_min": (min(evs) if evs else None), "ev_max": (max(evs) if evs else None),
            "aux_value_loss_first": (axl[0] if axl else None),
            "aux_value_loss_last": (axl[-1] if axl else None)}
    with open(os.path.join(args.log_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved: {final_path}\nSaved EV curve: {ev_csv}\nSaved meta.json", flush=True)


if __name__ == "__main__":
    main()
