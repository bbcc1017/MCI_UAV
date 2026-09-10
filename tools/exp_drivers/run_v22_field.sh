#!/usr/bin/env bash
# v22 — v20 이론 스테이지를 누수 없는 튜닝셋(budget750)에서 전부 재도출한다.
#
# ── §1. 왜 다시 도는가: 판정셋 누수 ─────────────────────────────────────────
# v20 의 theory 스테이지(yhold·red·extreme)와 threshold_sweep 은 튜닝 격자를
#   scenarios/manifests/v19/tradeoff250_manifest.json
# 위에서 훑었다. 그런데 그 250 좌표는 v19 가 "판정 전용" 으로 동결한 test750 의
# **정확한 부분집합**이다. 실측(2026-09-10):
#   set(tradeoff250) <= set(test750)          → True   (250/250 전수 포함)
#   |tradeoff250 ∩ budget750|                 → 0
#   |test750     ∩ budget750|                 → 0
# 즉 임계값(λ·yhold·red_gain)의 최적점을 판정 좌표에서 골랐다 = 선택누수다.
# v20 은 이 사실을 확인한 뒤 **λ 재도출만** budget750 으로 옮겼고
# (tools/exp_drivers/run_v20_budget_retune.sh → results/scoreboard/v20/budget/lam_base.csv)
# 등급 축(yhold)·수단 축(red)·봉투 밖(extreme)은 옮기지 않았다.
# 이 드라이버가 그 나머지를 옮긴다. 같은 최적점이 나오면 v20 결론이 유지되고
# 누수 반론이 사라진다. 달라지면 달라진 값이 정본이 된다.
# AGENTS.md 계약 1("판정셋으로 튜닝하지 않는다") 의 직접 이행이다.
#
# ── §2. 왜 λ 격자를 넓히는가: ts40 최적점이 격자 밖 ─────────────────────────
# 봉투 밖 조건 ts40(치료시간 ×4)에서 v20 의 λ 응답면이 격자 끝에서도 아직
# 단조 하강이다(results/scoreboard/v20/theory/extreme_ts40.csv 재집계, 평균 PDR_woG):
#   Q족  Q32 0.166809 → Q40 0.162897 → Q50 0.160381   (끝점에서 Δ=-0.00252)
#   P족  P32 0.166820 → P40 0.163026 → P50 0.160732   (끝점에서 Δ=-0.00229)
#   H족  H27 0.161730 → H35 0.159996 → H45 0.159724   (끝점에서 Δ=-0.00027)
# 최적점이 격자 오른쪽 밖에 있으므로 "λ 재튜닝으로 얼마나 회수되는가" 를 격자
# 끝값으로 답하면 회수율을 과소평가한다. 배경 수치: ts40 에서 고정 Q18 의 손실은
# +0.02698 ± 0.00287 (W/T/L 208/41/1) 이고, 격자 안 재튜닝(18→50)이
# +0.02562 ± 0.00262 = 94.9% 를 회수한다. 남은 5.1%(+0.00137 ± 0.00054)가
# 무튜닝 S족의 몫인지, 아니면 그냥 격자가 짧아서 생긴 잔차인지 갈라야 한다.
# 그래서 extreme 스테이지에만 다음을 **추가**한다(기존 42팔은 그대로 승계):
#   Q,P 족 λ ∈ {65, 85, 110, 140}   H 족 λ ∈ {60, 80, 110, 150}
# 팔 이름 규칙(Q65·P85·H60 …)은 기존과 같아서 v20 응답면과 같은 축에서 읽힌다.
#
# ── §3. 왜 S족·P족을 base 스테이지로 신설하는가 ────────────────────────────
# 기존 budget750 재도출본 results/scoreboard/v20/budget/lam_base.csv 의 팔은
# 실측 27개이고 **K·H·Q 세 족뿐**이다(policy_specs 확인). 즉 "무튜닝 카드(S족)의
# 봉투 안 비용" 이라는 이번 주 핵심 비교가 누수 없는 좌표셋에서 측정된 적이 없다.
# base 스테이지는 S족 전체 + P족 전체 + 대조(Q18·K12·START_LB3)를 **한 번의 호출**에
# 넣어 그 결측을 메운다. P족 base 는 results/scoreboard/v20/fieldinfo/budget750.csv 에
# 이미 있지만(Q18 앵커 포함, 같은 매니페스트·seed0..9), 다른 실행의 CSV 를 붙이려면
# 결합 전 키 검사가 필요하므로 같은 호출에 다시 넣는 편이 안전하다(추가 비용 67.5k ep).
#
# ── §4. CRN(공통 난수) 규약 ────────────────────────────────────────────────
# 한 조건의 모든 팔은 반드시 한 번의 v17_rule_eval.py 호출에 들어간다.
# 같은 좌표·같은 seed0..seed0+n_eps-1 를 공유해야 paired 비교가 성립한다(계약 2).
# 조건이 다르면(노브가 다르면) 물리가 달라지므로 조건 간 paired 는 하지 않는다.
#
# ── §5. 스테이지 ───────────────────────────────────────────────────────────
#   base     조건 1 (노브 없음)  · 팔 18 — S족 무튜닝 + P족 통신불요 + 대조 3
#   yhold    조건 12            · 팔 15 — 등급 축(Y/R) + S족 등급 변형(SY)
#   red      조건 14            · 팔 15 — 수단 축(G/D)
#   extreme  조건 7             · 팔 54 — v20 42팔 + λ 격자 확장 12팔
#   lamx     조건 5             · 팔 20 — (λ × yhold) 결합 셀. v20 은 두 축을 따로만
#                                        쓸었다 → 축 독립 가설의 직접 검증
# 조건별 MCI_* 노브 값은 추측이 아니라 v20 산출 메타
# (results/scoreboard/v20/theory/{yhold,red,extreme}_*.csv.meta.json 의 scenario_knobs)
# 와 run_v20_theory.sh 원문에서 읽어 그대로 옮겼다.
#
# ── §6. 규모 추정 (전부 추정이며 실측이 아니다) ────────────────────────────
# 에피소드 수 = 조건 × 팔 × 750 좌표 × 10 ep
#   base 135,000 / yhold 1,350,000 / red 1,575,000 / extreme 2,835,000 / lamx 750,000
#   합계 6,645,000 ep
# (a) 프로덕션 실측 처리량 550 ep/s(원본 코어·46워커, 84~93 core-ms/ep) 기준
#     base 0.07h · yhold 0.68h · red 0.80h · extreme 1.43h · lamx 0.38h → 합 3.4h
# (b) v20 실측 로그를 좌표 3배·팔 수 비례로 늘린 기준(더 정직한 상한)
#     base 0.1h · yhold 1.1h · red 1.1h · extreme 2.8h · lamx 0.6h → 합 약 5.8h @48워커
#     (b)가 큰 이유는 두 가지다. ① n500 은 에피소드가 무거워 실측 554 core-ms/ep
#     (다른 봉투 밖 조건은 95~114) — extreme 총비용의 약 절반을 혼자 먹는다.
#     ② 좌표당 시나리오 적재 고정비가 팔 수로 나눠지므로 팔이 적은 스테이지의
#     core-ms/ep 가 커진다(v20 11팔 블록 실측 141 core-ms/ep).
# 산출 용량: 6.65M 행 × 약 120 B ≈ 0.8 GB (평가기 스키마에 열이 늘면 더 커진다).
#
# ── §7. 사용 ───────────────────────────────────────────────────────────────
#   bash tools/exp_drivers/run_v22_field.sh <base|yhold|red|extreme|lamx> [워커수]
#   DRY=1 을 주면 조건·팔 수·예상 에피소드만 출력하고 아무것도 실행하지 않는다.
#   권장 순서: base → lamx → yhold → red → extreme (싼 것부터, n500 이 마지막)
#   재개: 조건별 <stage>_<tag>.csv.meta.json 이 있으면 그 조건을 건너뛴다.
#         메타 없이 CSV 만 남은 부분 기록은 평가기가 RuntimeError 로 막으므로
#         해당 CSV 를 수동 확인·정리한 뒤 다시 돌린다(operations.md §장시간 작업 6).
#   실패 처리: rc≠0 인 조건이 하나라도 있으면 <stage>.DONE 을 쓰지 않고
#              <stage>.FAILED 에 태그를 남기고 exit 1 한다("실패해도 DONE" 함정 방지).
set -u

STAGE=${1:?stage: base|yhold|red|extreme|lamx}
W=${2:-${W:-48}}                # 공유 노드다. 현재 부하를 보고 올릴 것(기본은 보수적으로 48)
DRY=${DRY:-0}
NEPS=${NEPS:-10}
SEED0=${SEED0:-0}
LOADMAX=${LOADMAX:-112}         # loadavg 게이트 임계(v20 118 / v21 110 의 중간). 논리코어 128

P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV
cd "$REPO" || exit 1
MAN=$REPO/scenarios/manifests/sigungu30_budget750_manifest.json
OUT=$REPO/results/scoreboard/v22/retune
LOG=$OUT/logs
mkdir -p "$LOG"

# BLAS/OpenMP 스레드 4종 고정. numpy/torch import 전에 걸어야 효과가 있고,
# 스레드 수가 부동소수 결과를 바꾼 선례가 있다(sim 고속화 G8 FAIL 진범).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 판정셋 오사용 방어. 이 드라이버는 튜닝셋 전용이다.
case "$MAN" in
  *test750*|*tradeoff250*|*eval250*)
    echo "[치명] 판정셋 매니페스트가 지정됐다: $MAN" >&2; exit 2;;
esac
[ -f "$MAN" ] || { echo "[치명] 매니페스트 없음: $MAN" >&2; exit 2; }

# 물리 노브 전량. 매 조건 실행 직전·직후에 전부 unset 해서 조건 간 상속 오염을 막는다
# (operations.md: 평가기는 MCI_CAP_GATE 등만 설정하고 자원·속도·인계 노브는 초기화하지 않는다).
KNOBS="MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER MCI_TREAT_SCALE"

FAIL=0
FAILED_TAGS=""

# ── 팔 조립기 ──────────────────────────────────────────────────────────────
# v20 run_v20_theory.sh 의 조립기를 글자 그대로 승계한다. 팔 이름·스펙이 같아야
# v20 응답면과 같은 축에서 읽힌다(단 좌표셋이 다르므로 v20 CSV 와 행을 섞어
# paired 검정하지는 않는다 — 조건이 같아도 좌표가 다르면 짝이 아니다).
#   K card:<λ_km>,12,0                     거리(km)축 + 선형 부하   — 구 CARD
#   H cardt:<λ_분>,6.6,0,hinge1            시간(분)축 + 초과분 부하
#   Q cardt:<λ_분>,6.6,0,hingerate         시간축 + 대기행렬 유도형(÷수술실수)
#   P cardt:<λ_분>,6.6,0,hingerate_psent   Q 와 같은 형태인데 부하 신호가 현장
#                                          누적발송 = 병원 통신 불요
#   S cards:<wait_scale>,6.6,0             목적함수(생존확률) 직접 최대화 = 무튜닝
arms_lam () {
  local s="" l w
  for l in 4 6 8 10 12 14 17 21 26 32; do s="$s;K$l=card:$l,12,0"; done
  for l in 4 8 12 16 21 27 35 45;      do s="$s;H$l=cardt:$l,6.6,0,hinge1"; done
  for l in 6 10 14 18 22 26 32 40 50;  do s="$s;Q$l=cardt:$l,6.6,0,hingerate"; done
  for l in 6 10 14 18 22 26 32 40 50;  do s="$s;P$l=cardt:$l,6.6,0,hingerate_psent"; done
  for w in 0.25 0.5 0.62 0.75 1.0 1.5; do s="$s;S$w=cards:$w,6.6,0"; done
  echo "${s#;}"
}

# extreme 전용 λ 격자 확장. 근거는 머리말 §2(ts40 응답면이 격자 끝에서도 단조 하강).
# 기존 42팔에 12팔을 더할 뿐이고 기존 팔의 스펙은 건드리지 않는다.
arms_lam_ext () {
  local s l
  s=$(arms_lam)
  for l in 65 85 110 140; do s="$s;Q$l=cardt:$l,6.6,0,hingerate"; done
  for l in 65 85 110 140; do s="$s;P$l=cardt:$l,6.6,0,hingerate_psent"; done
  for l in 60 80 110 150; do s="$s;H$l=cardt:$l,6.6,0,hinge1"; done
  echo "$s"
}

# 등급 축. Y=Q족(λ=18 고정)에서 yhold 만, R=K족(λ=12 고정)에서 yhold 만.
# SY 는 신설 — S족은 λ 튜닝 상수가 없으니 남는 자유도가 등급뿐이고,
# 그 최적 yhold 가 Q족과 같은지는 아직 열려 있다.
arms_yhold () {
  local s="" y
  for y in 0 2 4 8 16 32 9999; do s="$s;Y$y=cardt:18,6.6,$y,hingerate"; done
  for y in 0 8 32 9999;         do s="$s;R$y=card:12,12,$y"; done
  for y in 0 2 4 8;             do s="$s;SY$y=cards:0.62,6.6,$y"; done
  echo "${s#;}"
}

# 수단 축. G=Q족에서 red_gain(분) 만, D=K족에서 red_km 만.
arms_red () {
  local s="" g r
  for g in -6 -2 0 3 6.6 10 14 20 30 45; do s="$s;G$g=cardt:18,$g,0,hingerate"; done
  for r in 2 6 12 20 32;                 do s="$s;D$r=card:12,$r,0"; done
  echo "${s#;}"
}

# base 스테이지: 무튜닝 S족 + 통신불요 P족 + 대조 3(같은 호출 = CRN paired). §3 참조.
arms_base () {
  local s="" w l
  for w in 0.25 0.5 0.62 0.75 1.0 1.5; do s="$s;S$w=cards:$w,6.6,0"; done
  for l in 6 10 14 18 22 26 32 40 50;  do s="$s;P$l=cardt:$l,6.6,0,hingerate_psent"; done
  s="$s;Q18=cardt:18,6.6,0,hingerate"
  s="$s;K12=card:12,12,0"
  s="$s;START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst"
  echo "${s#;}"
}

# (λ × yhold) 결합 셀. v20 은 λ 를 yhold=0 에서만, yhold 를 λ=18 에서만 쓸었다.
# 두 축이 독립이면 이 20셀의 최소점이 (λ*, y*) = (v20 λ 최적, v20 y 최적) 이어야 한다.
arms_lamx () {
  local s="" l y
  for l in 10 14 18 22 26; do
    for y in 0 2 4 8; do s="$s;Q${l}Y${y}=cardt:$l,6.6,$y,hingerate"; done
  done
  echo "${s#;}"
}

# ── loadavg 게이트 ─────────────────────────────────────────────────────────
# 공유 학습 노드다(논리 128코어). 남의 잡과 겹쳐 노드를 죽이지 않도록 조건 시작 전에만
# 본다 — 실행 중에는 개입하지 않는다(중간에 죽이면 부분 CSV 가 남아 재개가 막힌다).
loadgate () {
  local i la
  for i in $(seq 1 120); do
    la=$(awk '{printf "%d", $1}' /proc/loadavg)
    [ "$la" -lt "$LOADMAX" ] && return 0
    echo "  [loadgate] loadavg=$la >= $LOADMAX — 60s 대기 ($i/120)"
    sleep 60
  done
  echo "  [loadgate] 2시간 대기 후에도 미해소 — 그대로 진행한다"
  return 0
}

# ── 조건 1개 실행 ──────────────────────────────────────────────────────────
run () {   # 태그  "MCI_* 노브들"  정책스펙  [매니페스트]  [에피소드수]
  local tag=$1 envs=$2 pol=$3 man=${4:-$MAN} ne=${5:-$NEPS} rc=0
  local f=$OUT/${STAGE}_${tag}.csv
  local lg=$LOG/${STAGE}_${tag}.log
  local narm
  narm=$(printf '%s' "$pol" | tr ';' '\n' | grep -c .)

  if [ -f "$f.meta.json" ]; then
    echo "  skip ${STAGE}/${tag} (완주 메타 있음)"
    return 0
  fi
  if [ "$DRY" = "1" ]; then
    echo "DRY ${STAGE}/${tag}  arms=${narm}  knobs=[${envs}]  eps=$((750 * narm * ne))  out=$f"
    echo "    pol=$(printf '%s' "$pol" | cut -c1-140)..."
    return 0
  fi

  loadgate
  # 상속 오염 방지: 물리 노브 전량 해제 → 이번 조건 값만 export → 끝나면 다시 전량 해제.
  for k in $KNOBS; do unset "$k"; done
  for kv in $envs; do export "$kv"; done
  echo "[$(date +%m-%d\ %H:%M:%S)] ${STAGE}/${tag} 시작 arms=${narm} knobs=[${envs}] W=$W loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  "$P" src/rl_src/v17_rule_eval.py \
      --manifest "$man" --policies "$pol" \
      --n_eps "$ne" --seed0 "$SEED0" --workers "$W" --out "$f" > "$lg" 2>&1
  rc=$?
  for k in $KNOBS; do unset "$k"; done
  echo "[$(date +%m-%d\ %H:%M:%S)] ${STAGE}/${tag} 종료 rc=$rc  $(tail -1 "$lg" | cut -c1-100)"
  if [ "$rc" -ne 0 ]; then
    FAIL=$((FAIL + 1))
    FAILED_TAGS="$FAILED_TAGS $tag"
    echo "  [실패] ${STAGE}/${tag} — 로그: $lg"
  fi
  return 0
}

# ── 스테이지 ───────────────────────────────────────────────────────────────
# "태그:노브들" 목록. 노브 값은 v20 메타(scenario_knobs)·run_v20_theory.sh 원문 그대로다.
case $STAGE in

  base)   # S족·P족의 봉투 안 비용. 조건 노브 없음 = 정식 물리조건.
    run base "" "$(arms_base)"
    ;;

  yhold)  # 등급 축 재도출. 조건 12개 = v20 yhold 스테이지와 동일.
    POL=$(arms_yhold)
    for spec in "base:" \
                "vamb30:MCI_AMB_VELOCITY=30" "vamb70:MCI_AMB_VELOCITY=70" "vamb100:MCI_AMB_VELOCITY=100" \
                "amb5:MCI_AMB_NUM=5" "amb10:MCI_AMB_NUM=10" \
                "n200:MCI_INCIDENT_SIZE=200" "n50:MCI_INCIDENT_SIZE=50" \
                "capa05:MCI_CAPA_SCALE=0.5" "capa20:MCI_CAPA_SCALE=2.0" \
                "ts05:MCI_TREAT_SCALE=0.5" "ts20:MCI_TREAT_SCALE=2.0"; do
      run "${spec%%:*}" "${spec#*:}" "$POL"
    done
    ;;

  red)    # 수단 축 재도출. 조건 = base + v20 red 스테이지 13개.
    POL=$(arms_red)
    for spec in "base:" \
                "vamb30:MCI_AMB_VELOCITY=30" "vamb70:MCI_AMB_VELOCITY=70" "vamb100:MCI_AMB_VELOCITY=100" \
                "vuav100:MCI_UAV_VELOCITY=100" "vuav150:MCI_UAV_VELOCITY=150" \
                "vuav300:MCI_UAV_VELOCITY=300" "vuav400:MCI_UAV_VELOCITY=400" \
                "huav5:MCI_UAV_HANDOVER=5" "huav20:MCI_UAV_HANDOVER=20" "hamb10:MCI_AMB_HANDOVER=10" \
                "uav1:MCI_UAV_NUM=1" "uav6:MCI_UAV_NUM=6" "uav13:MCI_UAV_NUM=13"; do
      run "${spec%%:*}" "${spec#*:}" "$POL"
    done
    ;;

  extreme) # 봉투 밖. 42팔 + λ 격자 확장 12팔. 싼 조건부터, n500(실측 5.1배)을 마지막에.
    POL=$(arms_lam_ext)
    for spec in "uav0:MCI_UAV_NUM=0" "amb1:MCI_AMB_NUM=1" "amb3:MCI_AMB_NUM=3" \
                "vamb20:MCI_AMB_VELOCITY=20" "capa03:MCI_CAPA_SCALE=0.3" \
                "ts40:MCI_TREAT_SCALE=4.0" "n500:MCI_INCIDENT_SIZE=500"; do
      run "${spec%%:*}" "${spec#*:}" "$POL"
    done
    ;;

  lamx)   # (λ × yhold) 결합. 축이 독립인지 직접 본다.
    POL=$(arms_lamx)
    for spec in "base:" "capa20:MCI_CAPA_SCALE=2.0" "ts05:MCI_TREAT_SCALE=0.5" \
                "vamb100:MCI_AMB_VELOCITY=100" "ts40:MCI_TREAT_SCALE=4.0"; do
      run "${spec%%:*}" "${spec#*:}" "$POL"
    done
    ;;

  *) echo "unknown stage: $STAGE (base|yhold|red|extreme|lamx)" >&2; exit 1;;
esac

if [ "$DRY" = "1" ]; then
  echo "[DRY] $STAGE — 실행·마커 없음"
  exit 0
fi
if [ "$FAIL" -eq 0 ]; then
  touch "$OUT/${STAGE}.DONE"
  echo "[$(date +%m-%d\ %H:%M:%S)] $STAGE 완료 — 실패 0건"
else
  printf '%s\n' "${FAILED_TAGS# }" > "$OUT/${STAGE}.FAILED"
  echo "[$(date +%m-%d\ %H:%M:%S)] $STAGE 미완 — 실패 ${FAIL}건:${FAILED_TAGS} (DONE 미기록)"
  exit 1
fi
