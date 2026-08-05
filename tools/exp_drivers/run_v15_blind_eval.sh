#!/usr/bin/env bash
set -euo pipefail

# v15 정책 동결 후 생성한 신규 블라인드250. 결과를 본 뒤 팔·설정을 바꾸지 않는다.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/home/ryu/anaconda3/envs/UAV/bin/python
FREEZE="$ROOT/results/scoreboard/v15/final/selection_freeze.json"
OUT="$ROOT/results/scoreboard/v15/blind/portfolio_vs_refs_blind250_seed12000_12029.csv"
WORKERS="${V15_WORKERS:-80}"

test -f "$FREEZE"
"$PY" "$ROOT/tools/verify_v15_freeze.py"
mkdir -p "$(dirname "$OUT")"

exec "$PY" "$ROOT/src/rl_src/v15_portfolio_eval.py" \
  --manifest "$ROOT/scenarios/manifests/v15_blind250_osrm_manifest.json" \
  --fold '' --n_regions 0 \
  --tree_dir "$ROOT/results/scoreboard/v13/sota_distill/students_full1000" \
  --ppo_tree_path "$ROOT/results/scoreboard/v10/distill/students_parallel/I1_FIELD_GBDT_L31_SOFT.pkl" \
  --n_eps 30 --seed0 12000 \
  --arms PPO,PURE_G1,FINAL,BASE_G1 \
  --h 20 --m 16 --workers "$WORKERS" --chunk 5 \
  --out "$OUT"
