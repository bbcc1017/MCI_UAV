#!/usr/bin/env bash
# v13 최종 PPO+NCRP+MILP 교사 증류: 기존 v10 산출물은 건드리지 않는다.
set -euo pipefail

cd /home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
ROOT=results/scoreboard/v13/sota_distill
DATA="$ROOT/data"
LOG="$ROOT/logs"
TRAIN="$DATA/hybrid_train750_p0p2_seed5000.npz"
VAL="$DATA/hybrid_val250_p3_seed7000.npz"
SPLIT="$ROOT/students_split750"
FULL="$ROOT/students_full1000"

mkdir -p "$DATA" "$LOG"

if [[ -e "$TRAIN" || -e "$VAL" ]]; then
  echo "기존 v13 데이터가 있어 중단합니다. 재개/삭제 여부를 먼저 확인하세요." >&2
  exit 2
fi

"$PY" src/rl_src/v13_hybrid_distill.py \
  --folds p0,p1,p2 --role train --seed 5000 --workers 64 --chunk 5 \
  --out "$TRAIN" >"$LOG/collect_train750.log" 2>&1 &
PID_T=$!

"$PY" src/rl_src/v13_hybrid_distill.py \
  --folds p3 --role validation --seed 7000 --workers 24 --chunk 5 \
  --out "$VAL" >"$LOG/collect_val250.log" 2>&1 &
PID_V=$!

cleanup() {
  kill "$PID_T" "$PID_V" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$PID_T"
wait "$PID_V"
trap - INT TERM

"$PY" src/rl_src/v13_hybrid_student_suite.py \
  --train_data "$TRAIN" --val_data "$VAL" --workers 8 --out_dir "$SPLIT" \
  >"$LOG/fit_split750.log" 2>&1

# 모델 격자는 사전고정되어 있으므로 전체 1,000좌표로 다시 적합한다.
"$PY" src/rl_src/v13_hybrid_student_suite.py \
  --train_data "$TRAIN,$VAL" --val_data "$VAL" --workers 8 --out_dir "$FULL" \
  >"$LOG/fit_full1000.log" 2>&1

"$PY" src/rl_src/v10_tree_eval.py \
  --tree_dir "$FULL" --n_eps 30 --seed0 0 --workers 96 \
  --out "$ROOT/hybrid_students_eval250_seed0_29.csv" \
  >"$LOG/eval250.log" 2>&1

echo "v13 SOTA 증류 파이프라인 완료: $ROOT"
