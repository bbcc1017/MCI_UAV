#!/usr/bin/env bash
# v12 Track A 판정 — 대표점250 seed 0–29 paired, v10 + X1~X6 를 **한 번의 실행**으로 평가.
#
# 한 실행에 다 넣는 이유: paired_eval_ladder 는 지역·에피소드 루프 안에서 모든 정책에 **같은
# 시드**를 주므로(_rollout_woG(fac, pol, s)), 한 번에 돌리면 모든 쌍비교가 완전한 same-seed
# paired 가 된다. 별도 실행으로 나누면 결합은 가능하지만 하드게이트가 늘어난다.
#
# 하드게이트: 이 실행의 V10 행이 기존 cube(scoreboard_common30_sigungu.csv 의 pdr_wog_PPO_POINTER_V10)
#   와 일치해야 한다. cube 는 float32 저장이므로 허용오차는 1e-9 가 아니라 **float32 정밀도
#   (≈1e-8 스케일)** 로 본다(Track B gate 에서 최대 3.25e-09 관측 = 동일 궤적 확인).
#
# 규칙 소스: Track B 와 동일한 좌표별 best-of-64(eval250) — v10 t4_summary 와 정합.
#
# ⚠️ 이 라운드는 스크리닝이다(시드 대조군 없음). 승자는 seed 1·2 복제 후 최종 수치 재확립.
set -eu

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
OUTDIR=$REPO/results/scoreboard/v12/eval250
RULES=$REPO/results/scoreboard/v12/lbT_sweep/eval250_best_rules.csv
MANIFEST=$REPO/scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json
MR=$REPO/results/rl/redesign
WORKERS=${1:-32}
NEPS=${2:-30}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTDIR"
cd "$REPO"

[ -f "$RULES" ] || { echo "규칙 CSV 없음: $RULES (run_v12_lbT_sweep.sh 먼저 실행)"; exit 1; }

V=essential+load+valid
MODELS="V10=$MR/v10_random4_1000_pointer_s0=$V"
MODELS="$MODELS,X1_bilinear=$MR/v12_x1_bilinear_s0=$V"
MODELS="$MODELS,X2_xattn1=$MR/v12_x2_xattn1_s0=$V"
MODELS="$MODELS,X3_gopt3=$MR/v12_x3_gopt3_s0=$V"
MODELS="$MODELS,X4_attn0=$MR/v12_x4_attn0_s0=$V"
MODELS="$MODELS,X5_cap518=$MR/v12_x5_cap518_s0=$V"
MODELS="$MODELS,X6_poolcritic=$MR/v12_x6_poolcritic_s0=$V"

for d in v10_random4_1000_pointer_s0 v12_x1_bilinear_s0 v12_x2_xattn1_s0 v12_x3_gopt3_s0 \
         v12_x4_attn0_s0 v12_x5_cap518_s0 v12_x6_poolcritic_s0; do
  [ -f "$MR/$d/final_model.zip" ] || { echo "모델 미완주: $MR/$d/final_model.zip"; exit 1; }
done

"$PY" src/rl_src/paired_eval_ladder.py \
  --manifest "$MANIFEST" --heur_csv "$RULES" --match sigcd --dataset_role eval250 \
  --models "$MODELS" --baselines "heur,lb_T4" \
  --n_eps "$NEPS" --seed 0 --workers "$WORKERS" \
  --env_variant "$V" \
  --out "$OUTDIR/v12_eval250_${NEPS}ep.csv" \
  --dump_pe "$OUTDIR/v12_eval250_${NEPS}ep_pe.npz"

echo
"$PY" tools/v12_scoreboard.py --pe "$OUTDIR/v12_eval250_${NEPS}ep_pe.npz" --out_dir "$OUTDIR"
