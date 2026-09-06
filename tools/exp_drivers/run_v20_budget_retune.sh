#!/usr/bin/env bash
# v20 — 임계값을 정본 튜닝셋(budget750)에서 다시 고른다.
# ⚠️ 최초 스윕은 tradeoff250 에서 돌렸는데 그 250 좌표가 v19 가 '판정 전용' 으로 선언한
# test750 의 부분집합이다(전수 포함 확인). 임계값을 판정좌표에서 고른 것이므로 선택누수다.
# budget750 은 v19 가 '스텝 예산 결정 전용' 으로 선언한 셋이고 test750 과 좌표가 겹치지 않는다.
# 여기서 다시 고른 값이 같으면 결론이 유지되고 누수 반론이 사라진다.
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
MAN=$REPO/scenarios/manifests/sigungu30_budget750_manifest.json
OUT=$REPO/results/scoreboard/v20/budget; LOG=$OUT/logs; mkdir -p $LOG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-45}; NEPS=${NEPS:-10}
POL=""
for l in 4 6 8 10 12 14 17 21 26 32; do POL="$POL;K$l=card:$l,12,0"; done
for l in 4 8 12 16 21 27 35 45;      do POL="$POL;H$l=cardt:$l,6.6,0,hinge1"; done
for l in 6 10 14 18 22 26 32 40 50;  do POL="$POL;Q$l=cardt:$l,6.6,0,hingerate"; done
POL=${POL#;}
f=$OUT/lam_base.csv
if [ ! -f $f.meta.json ]; then
  echo "[$(date +%H:%M)] budget750 재도출 시작 (27팔 x 750좌표 x $NEPS)"
  $P src/rl_src/v17_rule_eval.py --manifest $MAN --policies "$POL" \
     --n_eps $NEPS --workers $W --out $f > $LOG/base.log 2>&1
  echo "[$(date +%H:%M)] rc=$?  $(tail -1 $LOG/base.log | cut -c1-80)"
fi
touch $OUT/DONE
