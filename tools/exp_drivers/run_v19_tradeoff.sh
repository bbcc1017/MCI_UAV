#!/usr/bin/env bash
# v19 자원수 트레이드오프 — AMB/UAV 대수 축을 변주해 4정책의 PDR_woG 곡선을 만든다.
#
# 좌표 = tradeoff250(test750 에서 시군구당 1점) · seed 0..9 · 축 9설정 × 4정책 = 90,000 에피소드.
# ⚠️ RL·CARD 는 baseline(amb30·uav26) 에서 학습·튜닝된 정책이라 축 변주는 OOD 일반화 시험이다.
# ⚠️ MCI_UAV_NUM 은 출발지만 슬라이스한다 — 착륙 가능 헬기장 26곳은 uav_num 과 무관하게 유지.
# ⚠️ uav_num=0 / amb_num=0 은 action space 가 96 으로 바뀌어 pointer head 가 못 받으므로 제외.
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV
OUT=$REPO/results/scoreboard/v19/tradeoff
LOG=$OUT/logs
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
BY_SIDO=$REPO/scenarios/manifests/v19/tradeoff250_by_sido.json
NEPS=10
W=100
RULES="CARD_GRID_S=cardloc:results/scoreboard/v19/cards/params_grid_s.json;START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst"

cd $REPO
# (축, 값) 목록 — base 는 두 축의 공통점(amb30 uav26)
SETTINGS="base:0 amb:5 amb:10 amb:15 amb:20 uav:1 uav:3 uav:6 uav:13"

for s in $SETTINGS; do
  axis=${s%%:*}; val=${s##*:}
  tag=${axis}${val}; [ "$axis" = base ] && tag=base
  unset MCI_AMB_NUM MCI_UAV_NUM
  [ "$axis" = amb ] && export MCI_AMB_NUM=$val
  [ "$axis" = uav ] && export MCI_UAV_NUM=$val

  echo "[$(date +%H:%M)] === $tag 시작 ==="
  # 1) 규칙 2종 (한 실행에 동시)
  if [ ! -f $OUT/rule_$tag.csv.meta.json ]; then
    $P src/rl_src/v17_rule_eval.py --manifest $MAN --policies "$RULES" \
      --n_eps $NEPS --workers $W --out $OUT/rule_$tag.csv > $LOG/rule_$tag.log 2>&1
    echo "  rule_$tag rc=$?"
  fi
  # 2) 전국 PPO
  if [ ! -f $OUT/ppoN_$tag.csv.meta.json ]; then
    $P src/rl_src/v17_ppo_eval.py --manifest $MAN --model_dir results/rl/v19/national \
      --obs_variant field --policy_name PPO_NATIONAL --n_eps $NEPS --workers $W \
      --out $OUT/ppoN_$tag.csv > $LOG/ppoN_$tag.log 2>&1
    echo "  ppoN_$tag rc=$?"
  fi
  # 3) 광역시도 PPO — 시도별 모델 × 그 시도 좌표
  for sido in $($P -c "import json;print(' '.join(json.load(open('$BY_SIDO'))))"); do
    f=$OUT/ppoS_${tag}_$sido.csv
    [ -f $f.meta.json ] && continue
    regs=$($P -c "import json;print(','.join(json.load(open('$BY_SIDO'))['$sido']))")
    nreg=$($P -c "import json;print(len(json.load(open('$BY_SIDO'))['$sido']))")
    $P src/rl_src/v17_ppo_eval.py --manifest $MAN --model_dir results/rl/v19/sido_$sido \
      --obs_variant field --policy_name PPO_SIDO --regions "$regs" \
      --n_eps $NEPS --workers $nreg --out $f > $LOG/ppoS_${tag}_$sido.log 2>&1
    rc=$?; [ $rc -ne 0 ] && echo "  ppoS_${tag}_$sido rc=$rc"
  done
  echo "[$(date +%H:%M)] === $tag 완료 ==="
done
touch $OUT/ALL.DONE
echo "[$(date +%H:%M)] 전체 완료"
