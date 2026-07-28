#!/usr/bin/env bash
# v12 wave 1 — GOPT식 크로스어텐션 6팔 (전부 obs 402 고정 · v10 레시피 동일 · seed 0 · 10M).
#
# 동기: 기준선 head L[c,d,m]=f_class[c]+S[d,m]+g_mode[m] 에 class 축이 없어 Red/Yellow 의
#   목적지 순위가 수학적으로 동일하다(등급 차이는 action mask 만). GOPT 의
#   logits=bmm(item, ems^T) 구조는 scorer 를 쪼개지 않고 이를 해결한다(v8 표본단절 회피).
#
#   v12_x1_bilinear_s0    gopt_bilinear, n_gopt_blocks 0  : 인코더는 v10 그대로, head 만 bilinear
#                                                            → 중증도 조건부 목적지 순위 효과 격리 (★핵심)
#   v12_x2_xattn1_s0      gopt_bilinear, n_gopt_blocks 1  : + 수요↔목적지 크로스어텐션 1블록
#   v12_x3_gopt3_s0       gopt_bilinear, n_gopt_blocks 3, heads 8 : GOPT 원논문 설정(풀버전)
#   v12_x4_attn0_s0       pointer, n_attn_blocks 0        : attention 제거 하한(기여도 측정)
#   v12_x5_cap518_s0      pointer, head_hidden 518        : X1 용량 대조군(파라미터 동수)
#   v12_x6_poolcritic_s0  pointer, --pooled_critic        : v10 actor + GOPT식 순열불변 critic
#
# 파라미터(실측, extractor+head+vf 총):
#   v10 923,720 / X1 999,845 / X2 1,199,781 / X3 1,599,653 / X4 907,080 / X5 999,770 / X6 174,216
#   → X5 는 X1 과 −75 차이(동수). ⚠️X3 총량을 v10 구조로 맞추면 head_hidden 3596(192→3596→2)이
#     되어 퇴화한 대조군이므로 쓰지 않는다 — X2/X3 가 이기면 wave 2 에서 별도 매칭 설계.
#   ⚠️X6 는 flat vf(869k)가 풀링 critic 으로 대체되어 v10 의 1/5.3 크기다(설계의 본질, 보고 시 명시).
#
# GPU: 실측상 신경망은 wall-clock 의 6.8%(v10)~19.5%(2.28M) — 병목은 환경/IPC 라 깊이 증축이
#   거의 무료다(10M 7.5h→8.5h). 6런 = 54 슬롯 ≈ 실코어 12~16.
#
# 판정: 대표점250 seed 0–29 paired vs PPO v10(기존 cube 에 에피소드별 값 존재).
#   ⚠️ 이 라운드는 **스크리닝**이다(시드 대조군 없음 — 사용자 결정). 승자는 seed 1·2 복제 후
#   최종 수치를 재확립한다.
#
# 불변식: 전 팔 fresh-start(--init_from/--resume_from 금지 — v6/v9 는 대표점250 학습 이력이
#   있어 계보 오염). 학습 매니페스트·PPO 하이퍼·보상·obs 전부 v10 동일.
set -u

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
MANIFEST=$REPO/scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json
OUT=$REPO/results/rl/redesign

# train_ppo_feature.py 는 MCI_OBS_VARIANT 를 스스로 export 하지 않는다(호출자 책임).
# 미설정이면 essential(dim 209)로 조용히 오학습된다 — v5 DAR 1차 10M 낭비의 원인.
export MCI_OBS_VARIANT=essential+load+valid
export MCI_H_PAD=47
export MCI_CAP_GATE=occ
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$REPO" || exit 1

launch() {   # launch <run_name> <extra flags...>
  local name=$1; shift
  local dir=$OUT/$name
  if [ -f "$dir/final_model.zip" ]; then
    echo "[skip] $name (final_model.zip 존재)"
    return
  fi
  mkdir -p "$dir"
  nohup "$PY" src/rl_src/train_ppo_feature.py \
    --config_path "$MANIFEST" \
    --reward_mode pdrwog --norm_reward \
    --learning_rate 0.0003 --lr_anneal --target_kl 0.03 \
    --n_steps 512 --batch_size 512 --n_epochs 5 \
    --embed_dim 64 --ctx_dim 128 \
    --n_envs 8 --vec subproc \
    --total_timesteps 10000000 --seed 0 \
    --log_dir "$dir" \
    "$@" \
    > /dev/null 2> "$dir/train.err" &
  echo "[launch] $name pid=$! extra=$*"
}

launch v12_x1_bilinear_s0   --extractor gopt_bilinear --head_hidden 128 --n_gopt_blocks 0
launch v12_x2_xattn1_s0     --extractor gopt_bilinear --head_hidden 128 --n_gopt_blocks 1
launch v12_x3_gopt3_s0      --extractor gopt_bilinear --head_hidden 128 --n_gopt_blocks 3 --n_heads 8
launch v12_x4_attn0_s0      --extractor pointer       --head_hidden 128 --n_attn_blocks 0
launch v12_x5_cap518_s0     --extractor pointer       --head_hidden 518
launch v12_x6_poolcritic_s0 --extractor pointer       --head_hidden 128 --pooled_critic

sleep 5
echo "--- loadavg ---"; uptime
