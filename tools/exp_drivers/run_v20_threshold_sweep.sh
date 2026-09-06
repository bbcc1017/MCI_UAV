#!/usr/bin/env bash
# v20 — 임계값 일반화 스윕.
#
# 물음: CARD 의 임계값(부하 교환율 lambda, 수단 전환 red, 등급 yhold)은 특정 시나리오
# (환자 100 · AMB 30 · UAV 26 · 50/200 km/h · 인계 5/10분)에만 맞춘 상수인가, 아니면
# 파라미터가 바뀌면 예측 가능하게 따라 움직이는 구조상수인가.
#
# 방법: 런타임 물리 노브로 축을 한 번에 하나씩(OFAT) 바꾸고, 각 설정에서 임계값 격자를
# 폐루프로 전수 실행해 최적점과 U 자 곡선 전체를 남긴다. RL 재학습이 필요 없다 —
# 규칙의 응답면은 규칙만 돌리면 나온다(교사 재학습 팔은 results/rl/v20/* 에서 따로 진행).
#
# 팔 3종 (같은 실행에 넣어 CRN 확보 = 같은 좌표·같은 시드)
#   K : card:<lam_km>,12,0            거리(km) 축 — 현행 CARD
#   T : cardt:<lam_t>,6.6,0           시간(분) 축 — 속도 불변 가설
#   H : cardt:<lam_t>,6.6,0,hinge1    시간 축 + 대기행렬 유도형 부하(초과분만)
#
# 사용: bash tools/exp_drivers/run_v20_threshold_sweep.sh <stage1a|stage1b|stage2|stage3> [W]
#       DRY=1 을 주면 커맨드만 출력한다.
set -u
STAGE=${1:-stage1a}
W=${2:-${W:-60}}
DRY=${DRY:-0}
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV
MAN=$REPO/scenarios/manifests/v19/tradeoff250_manifest.json
OUT=$REPO/results/scoreboard/v20/sweep
LOG=$OUT/logs
NEPS=${NEPS:-10}
mkdir -p $OUT $LOG
cd $REPO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ── 팔 조립 ────────────────────────────────────────────────────────────────
LAM_KM="4 6 8 10 12 14 17 21 26 32"
LAM_T="2 4 6 9 12 15 19 24 30 38"
LAM_H="4 8 12 16 21 27 35 45"
# L : card:<lam_km>,12,0,hinge1  — 거리(km)축 + 대기행렬 hinge 부하.
# H(시간축+hinge) 가 K(거리축+선형부하) 를 이겼을 때 그 원인이 "시간축" 인지 "hinge 형태" 인지
# 분리하는 어블레이션. stage1a/b 를 이미 돌린 뒤 추가했으므로 별도 파일(prefix lamL)에 쓴다.
# 같은 매니페스트·같은 seed0 이라 시나리오 난수가 같아 CRN 은 유지된다.
LAM_L="4 6 8 10 12 14 17 21 26 32"
arms_stage1L () {
  local s=""
  for l in $LAM_L; do s="$s;L$l=card:$l,12,0,raw,hinge1"; done
  echo "${s#;}"
}
# Q : cardt:<S>,6.6,0,hingerate — 시간축 + 완전 유도형 부하(서버수로 나눔).
# 여기서 lambda 자리는 **평균 서비스시간(분)** 이어야 한다는 것이 이론의 예측이다.
LAM_Q="6 10 14 18 22 26 32 40 50"
arms_stage1Q () {
  local s=""
  for l in $LAM_Q; do s="$s;Q$l=cardt:$l,6.6,0,hingerate"; done
  echo "${s#;}"
}
# S : cards:<wait_scale>,6.6,0 — 목적함수(생존확률) 직접 최대화. 이론값은 wait_scale=1.0 이고
# 그 구성에는 튜닝 상수가 하나도 없다. 배율을 훑어 1.0 이 실제 최적인지 확인한다.
WS="0 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0"
arms_stage1S () {
  local s=""
  for w in $WS; do s="$s;S$w=cards:$w,6.6,0"; done
  echo "${s#;}"
}
arms_stage1 () {
  local s=""
  for l in $LAM_KM; do s="$s;K$l=card:$l,12,0"; done
  for l in $LAM_T;  do s="$s;T$l=cardt:$l,6.6,0"; done
  for l in $LAM_H;  do s="$s;H$l=cardt:$l,6.6,0,hinge1"; done
  echo "${s#;}"
}

# ── 설정 축 ────────────────────────────────────────────────────────────────
# "태그|환경변수들"  (base 는 노브 없음). 이론적으로 결정적인 축을 먼저 돌린다.
STAGE1A="base|
vamb30|MCI_AMB_VELOCITY=30
vamb70|MCI_AMB_VELOCITY=70
vamb100|MCI_AMB_VELOCITY=100
vuav100|MCI_UAV_VELOCITY=100
vuav400|MCI_UAV_VELOCITY=400
capa05|MCI_CAPA_SCALE=0.5
capa075|MCI_CAPA_SCALE=0.75
capa15|MCI_CAPA_SCALE=1.5
capa20|MCI_CAPA_SCALE=2.0
hamb10|MCI_AMB_HANDOVER=10
huav20|MCI_UAV_HANDOVER=20
huav5|MCI_UAV_HANDOVER=5"

STAGE1B="amb5|MCI_AMB_NUM=5
amb10|MCI_AMB_NUM=10
amb15|MCI_AMB_NUM=15
amb20|MCI_AMB_NUM=20
uav1|MCI_UAV_NUM=1
uav3|MCI_UAV_NUM=3
uav6|MCI_UAV_NUM=6
uav13|MCI_UAV_NUM=13
n50|MCI_INCIDENT_SIZE=50
n150|MCI_INCIDENT_SIZE=150
n200|MCI_INCIDENT_SIZE=200
n300|MCI_INCIDENT_SIZE=300
vamb40|MCI_AMB_VELOCITY=40
vuav150|MCI_UAV_VELOCITY=150
vuav300|MCI_UAV_VELOCITY=300"

loadgate () {
  for i in $(seq 1 60); do
    la=$(awk '{printf "%d", $1}' /proc/loadavg)
    [ "$la" -lt 115 ] && return 0
    echo "  [loadgate] loadavg=$la 대기 60s"; sleep 60
  done
}

run_setting () {   # 태그  환경변수들  정책스펙  파일접두
  local tag=$1 envs=$2 policies=$3 prefix=$4
  local f=$OUT/${prefix}_${tag}.csv
  [ -f $f.meta.json ] && { echo "  skip $prefix/$tag (완료)"; return; }
  if [ "$DRY" = "1" ]; then
    echo "DRY  tag=$tag envs=[$envs] out=$f"
    echo "     policies=$(echo "$policies" | cut -c1-160)..."
    return
  fi
  loadgate
  ( unset MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE \
          MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER
    for kv in $envs; do export "$kv"; done
    echo "[$(date +%H:%M)] $prefix/$tag 시작 envs=[$envs] loadavg=$(cut -d' ' -f1 /proc/loadavg)"
    $P src/rl_src/v17_rule_eval.py --manifest $MAN --policies "$policies" \
       --n_eps $NEPS --workers $W --out $f > $LOG/${prefix}_${tag}.log 2>&1
    rc=$?
    echo "[$(date +%H:%M)] $prefix/$tag 종료 rc=$rc  $(tail -1 $LOG/${prefix}_${tag}.log | cut -c1-90)" )
}

case $STAGE in
  stage1a|stage1b)
    LIST=$([ $STAGE = stage1a ] && echo "$STAGE1A" || echo "$STAGE1B")
    POL=$(arms_stage1)
    echo "== $STAGE : $(echo "$LIST" | wc -l) 설정 × $(echo "$POL" | tr ';' '\n' | wc -l) 팔 × 250 좌표 × $NEPS ep =="
    echo "$LIST" | while IFS='|' read -r tag envs; do
      [ -z "$tag" ] && continue
      run_setting "$tag" "$envs" "$POL" lam
    done
    touch $OUT/${STAGE}.DONE
    ;;
  stage1S)  # 목적함수 직접 최대화 — 무튜닝 구성이 최적인가
    POL=$(arms_stage1S)
    echo "== stage1S : $(echo "$STAGE1A
$STAGE1B" | wc -l) 설정 × $(echo "$POL" | tr ';' '\n' | wc -l) 팔 =="
    echo "$STAGE1A
$STAGE1B" | while IFS='|' read -r tag envs; do
      [ -z "$tag" ] && continue
      run_setting "$tag" "$envs" "$POL" lamS
    done
    touch $OUT/stage1S.DONE
    ;;
  stage1Q)  # 완전 유도형 — lambda 자리가 평균 서비스시간인가
    POL=$(arms_stage1Q)
    echo "== stage1Q : $(echo "$STAGE1A
$STAGE1B" | wc -l) 설정 × $(echo "$POL" | tr ';' '\n' | wc -l) 팔 =="
    echo "$STAGE1A
$STAGE1B" | while IFS='|' read -r tag envs; do
      [ -z "$tag" ] && continue
      run_setting "$tag" "$envs" "$POL" lamQ
    done
    touch $OUT/stage1Q.DONE
    ;;
  stage1L)  # 어블레이션 — 거리축 + hinge 부하
    POL=$(arms_stage1L)
    echo "== stage1L : $(echo "$STAGE1A
$STAGE1B" | wc -l) 설정 × $(echo "$POL" | tr ';' '\n' | wc -l) 팔 =="
    echo "$STAGE1A
$STAGE1B" | while IFS='|' read -r tag envs; do
      [ -z "$tag" ] && continue
      run_setting "$tag" "$envs" "$POL" lamL
    done
    touch $OUT/stage1L.DONE
    ;;
  stage2)   # 수단 전환 임계 — lambda* 는 stage1 집계 JSON 에서 읽는다
    OPT=$OUT/../optima.json
    [ -f $OPT ] || { echo "먼저 tools/v20_threshold_report.py optima 를 돌려 $OPT 를 만들어라"; exit 1; }
    echo "$STAGE1A
$STAGE1B" | while IFS='|' read -r tag envs; do
      case $tag in base|vamb*|vuav*|uav*|hamb*|huav*) ;; *) continue;; esac
      lt=$($P -c "import json;d=json.load(open('$OPT'));print(d['$tag']['T']['lam'])" 2>/dev/null) || continue
      lk=$($P -c "import json;d=json.load(open('$OPT'));print(d['$tag']['K']['lam'])" 2>/dev/null) || continue
      s=""
      for g in -4 0 3 6.6 10 14 20 30; do s="$s;TG$g=cardt:$lt,$g,0"; done
      for r in 4 8 12 16 24 40;        do s="$s;KR$r=card:$lk,$r,0"; done
      run_setting "$tag" "$envs" "${s#;}" red
    done
    touch $OUT/${STAGE}.DONE
    ;;
  stage3)   # 등급 임계 — 희소성 축에서만
    OPT=$OUT/../optima.json
    echo "base|
amb5|MCI_AMB_NUM=5
amb10|MCI_AMB_NUM=10
n200|MCI_INCIDENT_SIZE=200
n300|MCI_INCIDENT_SIZE=300
capa05|MCI_CAPA_SCALE=0.5
vamb30|MCI_AMB_VELOCITY=30" | while IFS='|' read -r tag envs; do
      lt=$($P -c "import json;d=json.load(open('$OPT'));print(d['$tag']['T']['lam'])" 2>/dev/null) || continue
      s=""
      for y in 0 2 4 8 16 32 9999; do s="$s;TY$y=cardt:$lt,6.6,$y"; done
      run_setting "$tag" "$envs" "${s#;}" yhold
    done
    touch $OUT/${STAGE}.DONE
    ;;
  *) echo "unknown stage: $STAGE"; exit 1;;
esac
echo "[$(date +%H:%M)] $STAGE 전체 완료"
