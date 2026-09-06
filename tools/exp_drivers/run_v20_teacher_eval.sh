#!/usr/bin/env bash
# v20 — 물리 조건별 교사 평가.
#
# v19 의 자원 트레이드오프 곡선은 base(amb30·uav26)에서 학습한 정책을 다른 자원수에 그대로
# 얹은 **OOD 시험**이었다. 여기서는 각 조건에서 다시 학습한 교사(results/rl/v20/<tag>)와
# base 교사(results/rl/v19/national)를 **같은 좌표·같은 시드**로 나란히 재서 그 교란을 없앤다.
# 규칙 팔(K12·H12 등)은 같은 매니페스트·seed0 로 돌린 스윕 CSV 를 그대로 짝지어 쓴다(CRN).
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV; cd $REPO
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
OUT=$REPO/results/scoreboard/v20/teacher; LOG=$OUT/logs; mkdir -p $OUT $LOG
W=${W:-40}; NEPS=${NEPS:-10}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

run () {  # 태그  환경변수들
  local tag=$1 envs=$2
  local f=$OUT/ppo_$tag.csv
  [ -f $f.meta.json ] && { echo "skip $tag"; return; }
  ( unset MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE \
          MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER
    for kv in $envs; do export "$kv"; done
    echo "[$(date +%H:%M)] $tag 시작 envs=[$envs]"
    # 1) base 교사(v19 national) 를 이 조건에 얹은 것 = OOD
    $P src/rl_src/v17_ppo_eval.py --manifest $MAN --model_dir results/rl/v19/national \
       --obs_variant field --policy_name PPO_BASE --n_eps $NEPS --workers $W \
       --out $f >> $LOG/$tag.log 2>&1 || echo "  PPO_BASE rc=$?"
    # 2) 이 조건에서 다시 학습한 교사
    if [ -f results/rl/v20/$tag/final_model.zip ]; then
      $P src/rl_src/v17_ppo_eval.py --manifest $MAN --model_dir results/rl/v20/$tag \
         --obs_variant field --policy_name PPO_RETRAIN --n_eps $NEPS --workers $W \
         --out $f >> $LOG/$tag.log 2>&1 || echo "  PPO_RETRAIN rc=$?"
    fi
    echo "[$(date +%H:%M)] $tag 종료  $(tail -1 $LOG/$tag.log | cut -c1-80)" )
}

run base   ""
run amb10  "MCI_AMB_NUM=10"
run amb20  "MCI_AMB_NUM=20"
run uav6   "MCI_UAV_NUM=6"
run n50    "MCI_INCIDENT_SIZE=50"
run n200   "MCI_INCIDENT_SIZE=200"
run capa05 "MCI_CAPA_SCALE=0.5"
touch $OUT/ALL.DONE
echo "[$(date +%H:%M)] 교사 평가 완료"
