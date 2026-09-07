#!/usr/bin/env bash
# v20 — CARD 일반화 이론을 빈틈없이 닫는 폐루프 실험 묶음.
#
# 스테이지별 목적
#   treat    ★서비스시간 축. lambda = 서비스시간/수술실수 라는 주장의 직접 검증. 지금까지 이 축만 고정이었다.
#   interact 축 상호작용. 지금까지 OFAT(한 번에 하나)만 흔들었다.
#   yhold    등급 규칙(T3) 검증. 생존곡선 기울기 교차 t*=71.5분 이론이 맞으면
#            접근이 빠른 조건(고속·도심)에서 Red 우선이 유리해져야 한다.
#   red      수단 규칙(T2) 검증. 운동학 손익분기 d0 예측대로 임계가 움직이는가.
#   extreme  이론이 깨지는 지점 탐색. 용량 0.3배·환자 500·AMB 3대·UAV 0대(도입 전).
#   info28   현장정보(I1) 구성이 28 물리조건 전부에서 버티는가.
#   test750  미개봉 판정셋 최종 확인(논문 표용).
#
# 사용: bash tools/exp_drivers/run_v20_theory.sh <stage> [W]
set -u
STAGE=${1:?stage}
W=${2:-${W:-25}}
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
TEST=$REPO/scenarios/manifests/sigungu30_test750_manifest.json
OUT=$REPO/results/scoreboard/v20/theory; LOG=$OUT/logs; mkdir -p $LOG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NEPS=${NEPS:-10}
KNOBS="MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER MCI_TREAT_SCALE"

# 팔 조립기
arms_lam () {            # lambda 축: K(거리+선형) H(분+초과) Q(분+초과/c) P(분+초과/c, 현장정보만)
  local s=""
  for l in 4 6 8 10 12 14 17 21 26 32; do s="$s;K$l=card:$l,12,0"; done
  for l in 4 8 12 16 21 27 35 45;      do s="$s;H$l=cardt:$l,6.6,0,hinge1"; done
  for l in 6 10 14 18 22 26 32 40 50;  do s="$s;Q$l=cardt:$l,6.6,0,hingerate"; done
  for l in 6 10 14 18 22 26 32 40 50;  do s="$s;P$l=cardt:$l,6.6,0,hingerate_psent"; done
  for w in 0.25 0.5 0.62 0.75 1.0 1.5; do s="$s;S$w=cards:$w,6.6,0"; done
  echo "${s#;}"
}
arms_yhold () { local s=""; for y in 0 2 4 8 16 32 9999; do s="$s;Y$y=cardt:18,6.6,$y,hingerate"; done
                for y in 0 8 32 9999; do s="$s;R$y=card:12,12,$y"; done; echo "${s#;}"; }
arms_red ()   { local s=""; for g in -6 -2 0 3 6.6 10 14 20 30 45; do s="$s;G$g=cardt:18,$g,0,hingerate"; done
                for r in 2 6 12 20 32; do s="$s;D$r=card:12,$r,0"; done; echo "${s#;}"; }
arms_final () { echo "START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst;CARD_K12=card:12,12,0;CARD_H12=cardt:12,6.6,0,hinge1;CARD_Q18=cardt:18,6.6,0,hingerate;CARD_P18=cardt:18,6.6,0,hingerate_psent;CARD_S062=cards:0.62,6.6,0;CARD_MODE=cardtm:12,6,6.6,0,hinge1"; }

run () {   # 태그  "환경변수들"  정책스펙  매니페스트  에피소드수
  local tag=$1 envs=$2 pol=$3 man=${4:-$MAN} ne=${5:-$NEPS}
  local f=$OUT/${STAGE}_${tag}.csv
  [ -f $f.meta.json ] && { echo "  skip $tag"; return; }
  for i in $(seq 1 90); do la=$(awk '{printf "%d",$1}' /proc/loadavg); [ "$la" -lt 118 ] && break
    echo "  [loadgate] $la 대기"; sleep 45; done
  ( for k in $KNOBS; do unset $k; done
    for kv in $envs; do export "$kv"; done
    echo "[$(date +%H:%M)] $STAGE/$tag [$envs]"
    $P src/rl_src/v17_rule_eval.py --manifest $man --policies "$pol" \
       --n_eps $ne --workers $W --out $f > $LOG/${STAGE}_${tag}.log 2>&1
    echo "[$(date +%H:%M)] $STAGE/$tag rc=$? $(tail -1 $LOG/${STAGE}_${tag}.log|cut -c1-60)" )
}

case $STAGE in
 treat)   POL=$(arms_lam)
   for s in 0.5 0.75 1.5 2.0 3.0; do run "ts$s" "MCI_TREAT_SCALE=$s" "$POL"; done
   run "base" "" "$POL" ;;
 interact) POL=$(arms_lam)
   run a10c05 "MCI_AMB_NUM=10 MCI_CAPA_SCALE=0.5" "$POL"
   run a10c20 "MCI_AMB_NUM=10 MCI_CAPA_SCALE=2.0" "$POL"
   run a20c05 "MCI_AMB_NUM=20 MCI_CAPA_SCALE=0.5" "$POL"
   run n200c05 "MCI_INCIDENT_SIZE=200 MCI_CAPA_SCALE=0.5" "$POL"
   run n200c20 "MCI_INCIDENT_SIZE=200 MCI_CAPA_SCALE=2.0" "$POL"
   run v30c05 "MCI_AMB_VELOCITY=30 MCI_CAPA_SCALE=0.5" "$POL"
   run a10n200 "MCI_AMB_NUM=10 MCI_INCIDENT_SIZE=200" "$POL"
   run v100c20 "MCI_AMB_VELOCITY=100 MCI_CAPA_SCALE=2.0" "$POL"
   run u6a10 "MCI_UAV_NUM=6 MCI_AMB_NUM=10" "$POL"
   run n50c20 "MCI_INCIDENT_SIZE=50 MCI_CAPA_SCALE=2.0" "$POL"
   run v30n200 "MCI_AMB_VELOCITY=30 MCI_INCIDENT_SIZE=200" "$POL"
   run ts2c05 "MCI_TREAT_SCALE=2.0 MCI_CAPA_SCALE=0.5" "$POL" ;;
 yhold)   POL=$(arms_yhold)
   for spec in "base:" "vamb30:MCI_AMB_VELOCITY=30" "vamb70:MCI_AMB_VELOCITY=70" "vamb100:MCI_AMB_VELOCITY=100" \
               "amb5:MCI_AMB_NUM=5" "amb10:MCI_AMB_NUM=10" "n200:MCI_INCIDENT_SIZE=200" "n50:MCI_INCIDENT_SIZE=50" \
               "capa05:MCI_CAPA_SCALE=0.5" "capa20:MCI_CAPA_SCALE=2.0" "ts05:MCI_TREAT_SCALE=0.5" "ts20:MCI_TREAT_SCALE=2.0"; do
     run "${spec%%:*}" "${spec#*:}" "$POL"; done ;;
 red)     POL=$(arms_red)
   for spec in "base:" "vamb30:MCI_AMB_VELOCITY=30" "vamb70:MCI_AMB_VELOCITY=70" "vamb100:MCI_AMB_VELOCITY=100" \
               "vuav100:MCI_UAV_VELOCITY=100" "vuav150:MCI_UAV_VELOCITY=150" "vuav300:MCI_UAV_VELOCITY=300" "vuav400:MCI_UAV_VELOCITY=400" \
               "huav5:MCI_UAV_HANDOVER=5" "huav20:MCI_UAV_HANDOVER=20" "hamb10:MCI_AMB_HANDOVER=10" \
               "uav1:MCI_UAV_NUM=1" "uav6:MCI_UAV_NUM=6" "uav13:MCI_UAV_NUM=13"; do
     run "${spec%%:*}" "${spec#*:}" "$POL"; done ;;
 extreme) POL=$(arms_lam)
   run capa03 "MCI_CAPA_SCALE=0.3" "$POL"
   run n500 "MCI_INCIDENT_SIZE=500" "$POL"
   run amb3 "MCI_AMB_NUM=3" "$POL"
   run vamb20 "MCI_AMB_VELOCITY=20" "$POL"
   run ts40 "MCI_TREAT_SCALE=4.0" "$POL"
   run uav0 "MCI_UAV_NUM=0" "$POL"
   run amb1 "MCI_AMB_NUM=1" "$POL" ;;
 info28)  POL=""
   for l in 6 10 14 18 22 26 32 40 50; do POL="$POL;P$l=cardt:$l,6.6,0,hingerate_psent"; done
   POL="${POL#;};Q18=cardt:18,6.6,0,hingerate"
   for spec in "vamb30:MCI_AMB_VELOCITY=30" "vamb40:MCI_AMB_VELOCITY=40" "vamb70:MCI_AMB_VELOCITY=70" \
     "vuav100:MCI_UAV_VELOCITY=100" "vuav150:MCI_UAV_VELOCITY=150" "vuav300:MCI_UAV_VELOCITY=300" "vuav400:MCI_UAV_VELOCITY=400" \
     "capa075:MCI_CAPA_SCALE=0.75" "capa15:MCI_CAPA_SCALE=1.5" "capa20:MCI_CAPA_SCALE=2.0" \
     "hamb10:MCI_AMB_HANDOVER=10" "huav5:MCI_UAV_HANDOVER=5" "huav20:MCI_UAV_HANDOVER=20" \
     "amb5:MCI_AMB_NUM=5" "amb15:MCI_AMB_NUM=15" "amb20:MCI_AMB_NUM=20" \
     "uav1:MCI_UAV_NUM=1" "uav3:MCI_UAV_NUM=3" "uav13:MCI_UAV_NUM=13" \
     "n50:MCI_INCIDENT_SIZE=50" "n150:MCI_INCIDENT_SIZE=150" "n300:MCI_INCIDENT_SIZE=300"; do
     run "${spec%%:*}" "${spec#*:}" "$POL"; done ;;
 test750) run final "" "$(arms_final)" "$TEST" 10 ;;
 *) echo "unknown stage $STAGE"; exit 1;;
esac
touch $OUT/${STAGE}.DONE; echo "[$(date +%H:%M)] $STAGE 완료"
