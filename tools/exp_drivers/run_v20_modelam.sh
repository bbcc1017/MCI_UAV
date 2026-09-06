#!/usr/bin/env bash
# v20 — 수단별 부하 교환율. 단일 lambda 의 잔여 조건의존성이 "UAV 이송 비율 변화" 때문인지 가른다.
# lam_amb 를 12 로 고정하고 lam_uav 만 훑는다. 가설이 맞으면 lam_uav* 는 조건에 불변이고
# lam_amb=12 도 그대로 유지되어, 자원·속도 축의 잔여 후회가 사라진다.
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
OUT=$REPO/results/scoreboard/v20/sweep; LOG=$OUT/logs; mkdir -p $LOG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-30}
POL=""
for lu in 2 4 6 8 12 16 21; do POL="$POL;M$lu=cardtm:12,$lu,6.6,0,hinge1"; done
POL=${POL#;}
echo "$1" | tr ' ' '\n' | while read -r spec; do
  [ -z "$spec" ] && continue
  tag=${spec%%:*}; kv=${spec#*:}
  f=$OUT/mlam_$tag.csv; [ -f $f.meta.json ] && { echo "skip $tag"; continue; }
  ( unset MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY
    [ "$kv" != "$tag" ] && [ -n "$kv" ] && export "$kv"
    echo "[$(date +%H:%M)] mlam/$tag envs=[$kv]"
    $P src/rl_src/v17_rule_eval.py --manifest $MAN --policies "$POL" --n_eps 10 --workers $W \
       --out $f > $LOG/mlam_$tag.log 2>&1; echo "  rc=$?" )
done
touch $OUT/mlam.DONE
