#!/usr/bin/env bash
# v12 wave 2 — 시드 복제(잡음 바닥) + 승자 결합. 9런, 전부 obs 402 고정·v10 레시피·10M.
#
# wave 1 결과(대표점250 seed 0–29 paired, 하드게이트 0.000e+00):
#   X4_attn0 0.14534(+0.00305, 89/138/23) > X6_poolcritic 0.14606(+0.00233, 71/132/47)
#   > X5_cap518 0.14739 > V10 0.14839 > X1_bilinear 0.14897 > X2 0.15423 > X3 0.15454
#   → GOPT 크로스어텐션 기각(X1 은 용량 동수 X5 에도 −0.00158 열세). 대신 **attention 제거가
#     개선**, **pooled critic 도 개선**(파라미터 1/5.3). 둘은 actor/critic 서로 다른 축.
#
# wave 2 가 답할 것 — wave 1 은 시드 대조군이 없어 +0.0031/+0.0023 이 시드 운인지 구조인지
#   판별 불가였다(v9 에서 같은 문제로 막힘). 여기서 잡음 바닥을 세우고 결합 이득을 본다.
#
#   v12_v10ctrl_s{1,2}      v10 레시피 그대로 시드만 → **잡음 바닥**(이게 없으면 결론 불가)
#   v12_x4_attn0_s{1,2}     attention 제거 재현성
#   v12_x6_poolcritic_s{1,2} 순열불변 critic 재현성
#   v12_x7_attn0pool_s{0,1,2} **결합**: attention 제거 + pooled critic (params 157,576 = 1/5.9)
#
# 용량 근거: 어제 실측 학습 6런 동시 = loadavg 9.66(런당 ~1.6 — env 워커 대부분이 파이프 대기라
#   runnable 이 아니다). 9런 ≈ 15. co-tenant(shin_full_baselines 97워커, 같은 ryu 계정) 와 합쳐
#   ~112 < 128 예산. 단 CPU 경합으로 wall-clock 은 늘어난다(단독 6.5h → 12~16h 예상).
#
# ⚠️ 전 팔 fresh-start. v10ctrl_s1/s2 는 정본 구성이지만 seed≠0 이라 meta 가
#   scoreboard_off_spec_args=['seed'] 로 표시하고 정본 PPO_POINTER_V10 id 를 자칭하지 않는다.
set -u

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
MANIFEST=$REPO/scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json
OUT=$REPO/results/rl/redesign
LOAD_LIMIT=${LOAD_LIMIT:-118}   # 런 launch 전 loadavg 상한(co-tenant 변동 대비)

export MCI_OBS_VARIANT=essential+load+valid
export MCI_H_PAD=47
export MCI_CAP_GATE=occ
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$REPO" || exit 1

wait_for_load() {
  for _ in $(seq 240); do   # 최대 2h 대기
    la=$(awk '{print int($1)}' /proc/loadavg)
    [ "$la" -lt "$LOAD_LIMIT" ] && return 0
    echo "  [gate] loadavg=$la ≥ $LOAD_LIMIT — 30s 대기"
    sleep 30
  done
  echo "  [gate] 대기 초과 — 그래도 진행"
}

launch() {   # launch <run_name> <seed> <extra flags...>
  local name=$1; shift
  local seed=$1; shift
  local dir=$OUT/$name
  if [ -f "$dir/final_model.zip" ]; then
    echo "[skip] $name (final_model.zip 존재)"
    return
  fi
  wait_for_load
  mkdir -p "$dir"
  nohup "$PY" src/rl_src/train_ppo_feature.py \
    --config_path "$MANIFEST" \
    --extractor pointer \
    --reward_mode pdrwog --norm_reward \
    --learning_rate 0.0003 --lr_anneal --target_kl 0.03 \
    --n_steps 512 --batch_size 512 --n_epochs 5 \
    --embed_dim 64 --ctx_dim 128 --head_hidden 128 \
    --n_envs 8 --vec subproc \
    --total_timesteps 10000000 --seed "$seed" \
    --log_dir "$dir" \
    "$@" \
    > /dev/null 2> "$dir/train.err" &
  echo "[launch] $name pid=$! seed=$seed extra=$*"
  sleep 8
}

# 잡음 바닥 먼저(가장 중요) → 재현성 → 결합
launch v12_v10ctrl_s1       1
launch v12_v10ctrl_s2       2
launch v12_x4_attn0_s1      1 --n_attn_blocks 0
launch v12_x4_attn0_s2      2 --n_attn_blocks 0
launch v12_x6_poolcritic_s1 1 --pooled_critic
launch v12_x6_poolcritic_s2 2 --pooled_critic
launch v12_x7_attn0pool_s0  0 --n_attn_blocks 0 --pooled_critic
launch v12_x7_attn0pool_s1  1 --n_attn_blocks 0 --pooled_critic
launch v12_x7_attn0pool_s2  2 --n_attn_blocks 0 --pooled_critic

sleep 10
echo "--- loadavg ---"; uptime
echo "--- 실행 중 ---"; pgrep -cf "[t]rain_ppo_feature.*v12_"
