#!/usr/bin/env bash
# v20 — 정보수준 검증. "현장 대원이 실제로 아는 것만" 으로 규칙이 성립하는가.
#
# Q 형태 `도달시간(분) + lambda * max(0, 부하+1-수술실수)/수술실수` 에서
# 각 항의 정보 출처는 이렇다.
#   도달시간   = 도로거리표 x 60 / 속도  -> 정적. 내비게이션이 주는 값. km 와 같은 정보다.
#   수술실수   = 병원 명부의 정적 값     -> 인쇄된 표
#   부하(occ)  = 병원 입원 census        -> ★병원과 통신해야 안다  <- 유일한 문제
#   부하(p_sent) = 내가 그 병원으로 보낸 누적 인원 -> 현장 지휘소 자기 기록(화이트보드)
#
# 그래서 부하항만 p_sent 로 바꾼 팔(P)을 만들어 통신 없는 구성의 비용을 잰다.
# 튜닝셋은 budget750(판정셋 test750 과 좌표 교집합 0).
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
OUT=$REPO/results/scoreboard/v20/fieldinfo; LOG=$OUT/logs; mkdir -p $LOG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-45}; NEPS=${NEPS:-10}
POL="Q18=cardt:18,6.6,0,hingerate"
for l in 6 10 14 18 22 26 32 40 50; do POL="$POL;P$l=cardt:$l,6.6,0,hingerate_psent"; done
for l in 14 18 22;                  do POL="$POL;F$l=cardt:$l,6.6,0,hingerate_if"; done

# 1) 정본 튜닝셋에서 lambda 재도출 + 통신 비용 측정
f=$OUT/budget750.csv
if [ ! -f $f.meta.json ]; then
  echo "[$(date +%H:%M)] budget750 정보수준 스윕 시작"
  $P src/rl_src/v17_rule_eval.py --manifest $REPO/scenarios/manifests/sigungu30_budget750_manifest.json \
     --policies "$POL" --n_eps $NEPS --workers $W --out $f > $LOG/budget750.log 2>&1
  echo "[$(date +%H:%M)] rc=$?  $(tail -1 $LOG/budget750.log | cut -c1-70)"
fi
# 2) 물리 조건을 바꿔도 현장정보 구성이 버티는지
for spec in "amb10:MCI_AMB_NUM=10" "n200:MCI_INCIDENT_SIZE=200" "capa05:MCI_CAPA_SCALE=0.5" "vamb100:MCI_AMB_VELOCITY=100"; do
  tag=${spec%%:*}; kv=${spec#*:}
  g=$OUT/$tag.csv; [ -f $g.meta.json ] && continue
  ( unset MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY
    export "$kv"
    echo "[$(date +%H:%M)] $tag 시작 [$kv]"
    $P src/rl_src/v17_rule_eval.py --manifest $REPO/scenarios/manifests/v19/tradeoff250_manifest.json \
       --policies "$POL" --n_eps $NEPS --workers $W --out $g > $LOG/$tag.log 2>&1
    echo "[$(date +%H:%M)] $tag rc=$?" )
done
touch $OUT/DONE; echo "[$(date +%H:%M)] 완료"
