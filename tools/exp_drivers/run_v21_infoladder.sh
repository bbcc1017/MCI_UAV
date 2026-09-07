#!/usr/bin/env bash
# v21 — 정보수준 × 함수형 격자. "현장이 무엇까지 알 수 있는가" 를 축으로 카드 여러 벌을 만든다.
#
# 목적지 점수는 전부 `도달시간(분) + lambda * Z(부하)` 한 형태이고, 바뀌는 것은 두 축뿐이다.
#
#   [정보수준 축]  Z 가 읽는 부하 신호의 출처
#     I3  occ + in_flight   병원 입원 census + 내 이송기록   → ★병원 통신 필요
#     I2  occ               병원 입원 census 만              → ★병원 통신 필요
#     I1a in_flight         지금 그 병원으로 가는 중          → 통신 불요(지휘소 배차 기록)
#     I1b p_sent            내가 그 병원에 보낸 누적          → 통신 불요(지휘소 화이트보드)
#     I0  없음              부하를 안 본다                    → 통신 불요
#
#   [함수형 축]  같은 신호를 어떤 식으로 벌점화하는가
#     rate   max(0, q+1-c)/c   대기행렬 유도형. 수술실수 c 를 알아야 한다
#     hinge  max(0, q+1-c)     초과분만. c 는 알지만 나눗셈은 안 한다
#     lin    q                 선형. c 를 몰라도 쓴다(가장 단순한 카드)
#
# lambda 는 셀마다 다시 고른다(같은 셀 안에서만 비교하면 축이 섞인다).
# 튜닝셋은 budget750 — 판정셋 test750 과 좌표 교집합 0 (v20 §6-4 누수 조치 승계).
# 등급(yhold)·수단(red_gain) 규칙은 전부 현장 지득 정보라 이 축과 무관하다 → 고정.
#
# 사용: bash tools/exp_drivers/run_v21_infoladder.sh [W]
set -u
W=${1:-${W:-44}}
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
OUT=$REPO/results/scoreboard/v21/infoladder; LOG=$OUT/logs; mkdir -p $LOG
BUD=$REPO/scenarios/manifests/sigungu30_budget750_manifest.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NEPS=${NEPS:-10}

# 신규 셀만 돌린다. 기존 셀은 v20 산출을 재사용한다(같은 매니페스트·seed0 → CRN 유지).
#   I3-rate  = v20/budget/lam_base.csv  Q6..Q50
#   I3-hinge = v20/budget/lam_base.csv  H4..H45
#   I1b-rate = v20/fieldinfo/budget750.csv  P6..P50
#   I1a-rate = v20/fieldinfo/budget750.csv  F14/F18/F22  (여기서 F6/F10/F26 보강)
POL=""
add () { POL="$POL;$1"; }
for l in 6 9 12 15 18 24; do add "T$l=cardt:$l,6.6,0,load"; done          # I3-lin
for l in 10 14 18 22 26 32; do add "OR$l=cardt:$l,6.6,0,hingerate_occ"; done  # I2-rate
for l in 6 9 12 16 21 27; do add "OH$l=cardt:$l,6.6,0,hinge1_occ"; done       # I2-hinge
for l in 6 9 12 15 18 24; do add "OL$l=cardt:$l,6.6,0,occ"; done              # I2-lin
for l in 6 10 26; do add "F$l=cardt:$l,6.6,0,hingerate_if"; done              # I1a-rate 보강
for l in 6 9 12 16 21 27; do add "FH$l=cardt:$l,6.6,0,hinge1_if"; done        # I1a-hinge
for l in 6 9 12 15 18 24; do add "FL$l=cardt:$l,6.6,0,in_flight"; done        # I1a-lin
for l in 6 9 12 16 21 27; do add "PH$l=cardt:$l,6.6,0,hinge1_psent"; done     # I1b-hinge
for l in 6 9 12 15 18 24; do add "PL$l=cardt:$l,6.6,0,p_sent"; done           # I1b-lin
add "Z0=cardt:0,6.6,0,zero"                                                   # I0
POL=${POL#;}

f=$OUT/budget750.csv
if [ -f $f.meta.json ]; then echo "skip budget750 (완주)"; else
  for i in $(seq 1 90); do la=$(awk '{printf "%d",$1}' /proc/loadavg); [ "$la" -lt 110 ] && break
    echo "  [loadgate] $la 대기"; sleep 45; done
  echo "[$(date +%H:%M)] v21 infoladder budget750 시작 (팔 $(echo "$POL"|tr ';' '\n'|wc -l)개, W=$W)"
  $P src/rl_src/v17_rule_eval.py --manifest $BUD --policies "$POL" \
     --n_eps $NEPS --workers $W --out $f > $LOG/budget750.log 2>&1
  echo "[$(date +%H:%M)] rc=$? $(tail -1 $LOG/budget750.log | cut -c1-70)"
fi
touch $OUT/DONE; echo "[$(date +%H:%M)] 완료"
