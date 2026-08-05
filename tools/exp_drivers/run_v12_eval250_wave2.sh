#!/usr/bin/env bash
# v12 wave 2 판정 — 시드 복제로 **잡음 바닥**을 세우고 X4·X6·X7 의 유의성을 가린다.
#
# 11모델을 **한 번의 paired 실행**에 넣어 모든 쌍비교가 same-seed 가 되게 한다:
#   V10(s0) + v10ctrl_s1/s2   → 동일 아키텍처 3시드 = 학습 시드 잡음 바닥
#   X4_attn0    s0/s1/s2      → attention 제거
#   X6_poolcritic s0/s1/s2    → 순열불변 critic
#   X7_attn0pool  s0/s1/s2    → 결합(params 157,576 = v10 의 1/5.9)
#
# 에피소드 30(seed 0–29): 이 판정의 불확실성은 **학습 시드**가 지배한다. 30ep×250지역이면
#   모델당 7,500 에피소드로 에피소드 CI ≈ ±0.00036 (효과 0.003 의 1/8) — 에피소드를 늘려도
#   시드 잡음은 줄지 않는다. 1000ep 는 최종 승자 헤드라인 수치용으로 별도 실행한다.
#
# 하드게이트: V10 행이 기존 cube(scoreboard_common30_sigungu.csv)와 일치(wave1 에서 0.000e+00).
# baselines 는 넣지 않는다 — heur/LB-T4 는 이미 cube 에 있고 wave1 에서 재현 확인됐다.
set -eu

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
OUTDIR=$REPO/results/scoreboard/v12/eval250_wave2
RULES=$REPO/results/scoreboard/v12/lbT_sweep/eval250_best_rules.csv
MANIFEST=$REPO/scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json
MR=$REPO/results/rl/redesign
WORKERS=${1:-20}
NEPS=${2:-30}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTDIR"
cd "$REPO"

V=essential+load+valid
declare -a RUNS=(
  "V10:v10_random4_1000_pointer_s0"
  "V10ctrl_s1:v12_v10ctrl_s1"
  "V10ctrl_s2:v12_v10ctrl_s2"
  "X4_attn0_s0:v12_x4_attn0_s0"
  "X4_attn0_s1:v12_x4_attn0_s1"
  "X4_attn0_s2:v12_x4_attn0_s2"
  "X6_pool_s0:v12_x6_poolcritic_s0"
  "X6_pool_s1:v12_x6_poolcritic_s1"
  "X6_pool_s2:v12_x6_poolcritic_s2"
  "X7_a0pool_s0:v12_x7_attn0pool_s0"
  "X7_a0pool_s1:v12_x7_attn0pool_s1"
  "X7_a0pool_s2:v12_x7_attn0pool_s2"
)
MODELS=""
for r in "${RUNS[@]}"; do
  name=${r%%:*}; dir=${r#*:}
  [ -f "$MR/$dir/final_model.zip" ] || { echo "모델 미완주: $MR/$dir"; exit 1; }
  MODELS="${MODELS:+$MODELS,}$name=$MR/$dir=$V"
done
echo "[wave2] 모델 ${#RUNS[@]}개, neps=$NEPS, workers=$WORKERS"

"$PY" src/rl_src/paired_eval_ladder.py \
  --manifest "$MANIFEST" --heur_csv "$RULES" --match sigcd --dataset_role eval250 \
  --models "$MODELS" --baselines "" \
  --n_eps "$NEPS" --seed 0 --workers "$WORKERS" \
  --env_variant "$V" \
  --out "$OUTDIR/v12_wave2_eval250_${NEPS}ep.csv" \
  --dump_pe "$OUTDIR/v12_wave2_eval250_${NEPS}ep_pe.npz"

echo
"$PY" tools/v12_wave2_report.py --pe "$OUTDIR/v12_wave2_eval250_${NEPS}ep_pe.npz" --out_dir "$OUTDIR"
