#!/usr/bin/env bash
# v12 Track B — LB-T 전수 스윕 (학습 없음). T=2..40 + 상한없음.
#
# 목적 3개:
#   1) T–PDR 곡선 — T=4 가 실제 최적인지, 어디서 포화하는지(상한이 안 걸려 순수 최근접에 수렴).
#   2) **지역별 argmin_T 산포** — T 동적화 여지의 직접 측정. 전부 4 근처면 동적 T 는 학습 없이
#      기각되고, 넓게 퍼지면 T-메타 RL(t_meta_wrapper.py 자산)을 되살릴 근거가 된다.
#   3) 두 기준선 — (a) train1000 적합 T 를 eval250 에 전이(배포 가능), (b) eval250 자체 argmin
#      = 평가후 발췌 oracle(v10 프로토콜의 HEUR64 Best-of-64 와 동일 관례).
#
# 규칙 소스: v10 `t4_summary` 와 **동일한** 좌표별 best-of-64 규칙(heuristic_best_summary.csv 의
#   dataset=eval250 행, base_rule_index 일치 확인됨) → 스윕의 T=4 가 cube 의 LB_T4 행과 정합.
# 시드/에피소드: seed 0..29 (cube 공통 30ep) · env occ / essential+load+valid / H_PAD 47.
#
# 사용: bash tools/exp_drivers/run_v12_lbT_sweep.sh [gate|full] [workers]
#   gate = 5지역만 돌려 cube LB_T4 와 하드게이트 비교(파서 확장 회귀 검증)
#   full = 250지역 전수
set -eu

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
MODE=${1:-gate}
WORKERS=${2:-32}
OUTDIR=$REPO/results/scoreboard/v12/lbT_sweep
RULES=$OUTDIR/eval250_best_rules.csv
MANIFEST=$REPO/scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTDIR"
cd "$REPO"

# ---- 1) 규칙 CSV 파생 (v10 t4 와 동일 소스) ----
if [ ! -f "$RULES" ]; then
  "$PY" - <<PYEOF
import pandas as pd
src = "results/scoreboard/v10/full1000/heuristic_best_summary.csv"
d = pd.read_csv(src, encoding="utf-8-sig")
e = d[d.dataset == "eval250"][["region", "sigcd", "best_rule"]].copy()
assert len(e) == 250 and e.sigcd.nunique() == 250, (len(e), e.sigcd.nunique())
e.to_csv("$RULES", index=False, encoding="utf-8")
print(f"[rules] {len(e)}개 좌표 규칙 → $RULES")
PYEOF
fi

# ---- 2) baseline 목록: T=2..40 + 상한없음 ----
BASE="lb_T2"
for t in $(seq 3 40); do BASE="$BASE,lb_T$t"; done
BASE="$BASE,lb_Tinf"

if [ "$MODE" = "gate" ]; then
  REGIONS=$("$PY" - <<'PYEOF'
import json
mf = json.load(open("scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json",
                   encoding="utf-8"))
print(",".join(sorted(mf)[:5]))
PYEOF
)
  echo "[gate] regions=$REGIONS"
  "$PY" src/rl_src/paired_eval_ladder.py \
    --manifest "$MANIFEST" --heur_csv "$RULES" --match sigcd --dataset_role eval250 \
    --models "" --baselines "heur,lb_T4,lb_adaptT" \
    --regions "$REGIONS" --n_eps 30 --seed 0 --workers 5 \
    --env_variant essential+load+valid \
    --out "$OUTDIR/gate_lbT4.csv" --dump_pe "$OUTDIR/gate_lbT4_pe.npz"
else
  "$PY" src/rl_src/paired_eval_ladder.py \
    --manifest "$MANIFEST" --heur_csv "$RULES" --match sigcd --dataset_role eval250 \
    --models "" --baselines "heur,$BASE,lb_adaptT" \
    --n_eps 30 --seed 0 --workers "$WORKERS" \
    --env_variant essential+load+valid \
    --out "$OUTDIR/lbT_sweep_eval250_30ep.csv" \
    --dump_pe "$OUTDIR/lbT_sweep_eval250_30ep_pe.npz"
fi
