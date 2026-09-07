#!/usr/bin/env bash
# v20 — 기전 계측. PDR 을 "미진입 손실" 과 "지연 손실" 로 분해하고 수술실 진입률을 보조지표로 남긴다.
# 규칙 9종 x 물리 7조건. 같은 좌표·같은 시드(CRN).
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
OUT=$REPO/results/scoreboard/v20/mechanism; LOG=$OUT/logs; mkdir -p $LOG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-40}; NEPS=${NEPS:-10}
POL="NOLOAD=cardt:0,6.6,0,zero"
POL="$POL;START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst"
POL="$POL;K12=card:12,12,0;H12=cardt:12,6.6,0,hinge1"
POL="$POL;Q6=cardt:6,6.6,0,hingerate;Q18=cardt:18,6.6,0,hingerate;Q50=cardt:50,6.6,0,hingerate"
POL="$POL;P18=cardt:18,6.6,0,hingerate_psent;S062=cards:0.62,6.6,0"
for spec in "base:" "capa05:MCI_CAPA_SCALE=0.5" "capa20:MCI_CAPA_SCALE=2.0" \
            "n200:MCI_INCIDENT_SIZE=200" "amb10:MCI_AMB_NUM=10" "vamb30:MCI_AMB_VELOCITY=30" \
            "ts20:MCI_TREAT_SCALE=2.0"; do
  tag=${spec%%:*}; kv=${spec#*:}
  f=$OUT/$tag.csv; [ -f $f.meta.json ] && { echo "skip $tag"; continue; }
  for i in $(seq 1 90); do la=$(awk '{printf "%d",$1}' /proc/loadavg); [ "$la" -lt 118 ] && break; sleep 45; done
  ( unset MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER MCI_TREAT_SCALE
    [ -n "$kv" ] && export "$kv"
    echo "[$(date +%H:%M)] mech/$tag [$kv]"
    $P src/rl_src/v20_mechanism_eval.py --manifest $MAN --policies "$POL" \
       --n_eps $NEPS --workers $W --out $f > $LOG/$tag.log 2>&1
    echo "[$(date +%H:%M)] $tag rc=$? $(grep -E '항등식|완료' $LOG/$tag.log | tail -2 | tr '\n' ' ')" )
done
touch $OUT/DONE; echo "[$(date +%H:%M)] 기전 계측 완료"
