#!/usr/bin/env bash
# v12 Track B — LB-T 발송상한 전수 스윕. **전 1,250좌표 × 1,000 에피소드 × T=2..40(39개)**.
#
# 설계(단일 격자·단일 에피소드수로 통일):
#   좌표 1,250 = 학습 random4 1,000 + 대표점 250 (v10 프로토콜 total_coordinates 와 동일)
#   에피소드   = 좌표당 1,000, seed 0..999 (v10 휴리스틱 기준선과 **정확히 같은 시드**)
#   정책       = lb_T2 .. lb_T40 (39개)
#
#   heur / lb_T4 / lb_adaptT / lb_Tinf 를 넣지 않는 이유:
#     * heur(좌표별 best-of-64)와 T=4 는 **이미 v10 에 1000ep·seed 0..999 로 존재**한다
#       (results/scoreboard/v10/full1000/{heuristic_best_summary,t4_summary}.csv) → 재계산 낭비이고,
#       우리 스윕의 T=4 가 t4_summary 와 일치하는지가 그대로 하드게이트가 된다.
#     * lb_adaptT 는 30ep 파일럿에서 lb_T4 와 **완전 동일**로 확인됨(부하 regime bin 미발동) = 중복.
#     * lb_Tinf(상한 없음)는 T=40 이 포화 앵커 역할을 하므로 불필요.
#
# 규칙 소스: v10 t4_summary 와 동일한 좌표별 best-of-64(heuristic_best_summary.csv 의 eval250 행,
#   base_rule_index 일치 확인). 규칙은 고정하고 **T 만** 변화 → ceteris paribus.
#
# 청크: paired_eval_ladder 는 중간 체크포인트가 없어 단일 장시간 실행이 막판에 죽으면 전량 손실.
#   5청크(eval250 / train p0..p3, 각 250좌표)로 쪼개 청크 단위 재개(완료분 스킵)한다.
#
# 초기 30ep 파일럿은 results/scoreboard/v12/lbT_sweep/pilot30ep/ 로 격리(v10 pilot30_300 관례).
#
# 사용: bash tools/exp_drivers/run_v12_lbT_sweep.sh [workers] [neps]
set -u

REPO=/home/ryu/MCI_UAV
PY=/home/ryu/anaconda3/envs/UAV/bin/python
WORKERS=${1:-96}
NEPS=${2:-30}
T_LO=2
T_HI=40
OUTDIR=$REPO/results/scoreboard/v12/lbT_sweep
RULES=$OUTDIR/eval250_best_rules.csv
EVAL_MF=$REPO/scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json
TRAIN_MF=$REPO/scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTDIR"
cd "$REPO"

# ---- 규칙 CSV 파생 (v10 t4 와 동일 소스) ----
if [ ! -f "$RULES" ]; then
  "$PY" - <<PYEOF
import pandas as pd
d = pd.read_csv("results/scoreboard/v10/full1000/heuristic_best_summary.csv", encoding="utf-8-sig")
e = d[d.dataset == "eval250"][["region", "sigcd", "best_rule"]].copy()
assert len(e) == 250 and e.sigcd.nunique() == 250, (len(e), e.sigcd.nunique())
e.to_csv("$RULES", index=False, encoding="utf-8")
print(f"[rules] {len(e)}개 좌표 규칙 → $RULES")
PYEOF
fi

BASE="lb_T$T_LO"
for t in $(seq $((T_LO + 1)) "$T_HI"); do BASE="$BASE,lb_T$t"; done
NPOL=$(echo "$BASE" | tr ',' '\n' | wc -l)
echo "[v12 LB-T] T=$T_LO..$T_HI ($NPOL 정책) × 1,250좌표 × ${NEPS}ep, workers=$WORKERS"
echo "[v12 LB-T] 총 에피소드 = $((1250 * NPOL * NEPS))"

sweep() {  # sweep <tag> <manifest> <dataset_role> <key_filter>
  local tag=$1 mf=$2 role=$3 kf=$4
  local out=$OUTDIR/lbT_${tag}_${NEPS}ep.csv
  if [ -f "$out" ]; then echo "[skip] $tag (존재)"; return 0; fi
  echo "=== [$(date '+%m-%d %H:%M:%S')] $tag 시작 ==="
  "$PY" src/rl_src/paired_eval_ladder.py \
    --manifest "$mf" --heur_csv "$RULES" --match sigcd --dataset_role "$role" \
    --models "" --baselines "$BASE" ${kf:+--key_filter "$kf"} \
    --n_eps "$NEPS" --seed 0 --workers "$WORKERS" \
    --env_variant essential+load+valid \
    --out "$out" --dump_pe "$OUTDIR/lbT_${tag}_${NEPS}ep_pe.npz"
  echo "--- [$(date '+%m-%d %H:%M:%S')] $tag 완료"
}

sweep eval250    "$EVAL_MF"  eval250   ""
sweep train_p0   "$TRAIN_MF" train1000 "_p0"
sweep train_p1   "$TRAIN_MF" train1000 "_p1"
sweep train_p2   "$TRAIN_MF" train1000 "_p2"
sweep train_p3   "$TRAIN_MF" train1000 "_p3"
echo "[v12 LB-T] 전체 완료 $(date)"
