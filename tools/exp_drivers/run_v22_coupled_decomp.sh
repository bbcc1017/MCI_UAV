#!/usr/bin/env bash
# v22 — 결합 점수(`coupled=on`)의 이득을 두 성분으로 분해한다.
#
# ── §1. 왜 도는가: ⑮ 의 Δ 를 해석할 수 없다 ────────────────────────────────
# research.md §v22 ⑮ 는 결합형이 순차형을 이긴다고 측정했다(budget750, 10시드):
#   base  최선 mu=6.6 : +0.000513 ± 0.000132  (W/T/L 65/684/1)
#   ts40  최선 mu=0   : +0.003782 ± 0.000618  (W/T/L 150/600/0)
# 그런데 그때의 `coupled=on` 구현은 **두 가지를 한꺼번에** 바꾼다.
#   ① 결정을 결합했다 — 수단을 최소 ETA 로 먼저 고정하지 않고 적격 (수단, 병원) 쌍
#      전체에서 `t_m(h) + λ·(q+1−c)⁺/c + μ·1[m=UAV]` 를 한 번에 argmin 한다.
#   ② Yellow 에게 UAV 를 허용했다 — 순차형은 `Yellow → AMB`(AMB 적격이 있을 때) 인데
#      결합형은 그 제약을 걸지 않으므로 Yellow 도 UAV 쌍을 후보로 본다.
# 그래서 ⑮ 의 Δ 는 ①+② 의 합산이고, 카드에 결합형을 넣을지 판단할 수 없다.
# 이 드라이버가 ② 만 끄는 팔(`yellow_amb_only=on`)을 추가해 두 성분을 가른다.
#
# ── §2. 스위치와 팔 설계 ───────────────────────────────────────────────────
# `cardt2:` 스펙에 9번째 토큰 `yellow_amb_only` 를 신설했다(생략 = off = 현행 동작).
#   REF        cardt:18,6.6,0,hingerate                     순차 — 현행 카드
#   FREE{mu}   cardt2:18,{mu},0,0,0,hingerate,on,on         결합 + Yellow 수단 자유 = ⑮ 대상
#   AMBONLY{mu} cardt2:18,{mu},0,0,0,hingerate,on,on,on     결합 + Yellow→AMB (순차와 같은 제약)
# 분해식(전부 **팔 대 팔 paired**, REF 경유 차감이 아니다 — v11 교훈):
#   결합분      = AMBONLY − REF
#   Yellow자유분 = FREE − AMBONLY
#   합          = FREE − REF (= ⑮ 의 Δ)
# 읽는 법: AMBONLY ≈ FREE 면 이득은 결합 때문 · AMBONLY ≈ REF 면 이득은 Yellow 수단
# 자유 때문 · 둘 사이면 두 성분의 크기를 각각 적는다. 판정선 0.00053(CRN paired).
#
# ⚠️ `AMBONLY` 는 "Yellow 는 언제나 AMB" 가 아니다. 순차형이 `m = 0 if has_a else 1`
# 이므로 AMB 적격이 하나도 없으면 Yellow 도 UAV 를 탄다(마스크 강제). 스위치도 같은
# 규약이라 **수단 자유도만** 끄고 이송 가능/불가는 바꾸지 않는다. 실측(게이트3, 5시드):
#   천안 서북 q17 mu=6.6 — Yellow 두수단자유 5상태 중 FREE 는 UAV 4회, AMBONLY 는 0회.
#   중구 q04 ts40 mu=0   — 자유 94상태 중 FREE 는 UAV 13회, AMBONLY 는 0회.
#   두 경우 모두 AMBONLY 의 Yellow_UAV 총량이 REF 와 정확히 같다(48=48, 9=9).
#
# ── §3. mu 격자 ────────────────────────────────────────────────────────────
# ⑮ 의 조건별 최선값 근방을 훑는다(base 최선 6.6 · ts40 최선 0).
#   mu ∈ {0, 3, 6.6, 10, 15}  — ⑮ 격자에서 25 만 뺐다(양 조건 모두 열세/동률이었다).
# mu 는 결합 점수의 UAV 가산항(분)이고, `coupled=off` 에서는 Red 의 UAV 전환 시간이득
# 임계와 같은 인자다. REF 는 그 임계 6.6 분으로 고정한 현행 카드다.
#
# ── §4. CRN(공통 난수) 규약 ────────────────────────────────────────────────
# 한 조건의 11팔 전부를 **한 번의 v17_rule_eval.py 호출**에 넣는다. 같은 좌표·같은
# seed0..seed0+n_eps-1 를 공유해야 paired 가 성립한다(AGENTS.md 계약 2).
# 조건이 다르면 물리가 다르므로 조건 간 paired 는 하지 않는다.
#
# ── §5. 규모 (실측 기반 추정) ──────────────────────────────────────────────
# 조건 2 × 팔 11 × 750좌표 × 10ep = 165,000 ep.
# 근거 실측: results/scoreboard/v22/coupled/{base,ts40}.csv.meta.json 이 각각
# 52,500 ep 을 173.7s / 167.8s @48워커에 끝냈다 = 302 / 313 ep/s.
# 같은 처리량이면 조건당 약 273s, 합 약 9분이다(56워커면 더 짧다).
# ⚠️ 이 값은 이 노드의 공유 부하에 따라 흔들린다 — 추정이지 보장이 아니다.
#
# ── §6. 사용 ───────────────────────────────────────────────────────────────
#   bash tools/exp_drivers/run_v22_coupled_decomp.sh [워커수]
#   DRY=1 이면 조건·팔 수·예상 에피소드만 출력하고 아무것도 실행하지 않는다.
#   재개: 조건별 <tag>.csv.meta.json 이 있으면 그 조건을 건너뛴다.
#         메타 없이 CSV 만 남은 부분 기록은 평가기가 RuntimeError 로 막으므로
#         해당 CSV 를 수동 확인·정리한 뒤 다시 돌린다.
#   집계: python tools/v20_threshold_report.py paired --csv <CSV> --a A --b B
set -u

W=${1:-${W:-56}}                # 공유 노드. 56 이하로 제한한다(논리 128코어).
DRY=${DRY:-0}
NEPS=${NEPS:-10}
SEED0=${SEED0:-0}
LOADMAX=${LOADMAX:-112}

P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV
cd "$REPO" || exit 1
MAN=$REPO/scenarios/manifests/sigungu30_budget750_manifest.json
OUT=$REPO/results/scoreboard/v22/coupled_decomp
LOG=$OUT/logs
mkdir -p "$LOG"

if [ "$W" -gt 56 ]; then
  echo "[치명] 워커 상한은 56 이다 (got $W)" >&2; exit 2
fi

# BLAS/OpenMP 스레드 4종 고정. numpy import 전에 걸어야 하고, 스레드 수가 부동소수
# 결과를 바꾼 선례가 있다(sim 고속화 G8 FAIL 진범).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 판정셋 오사용 방어. 이 드라이버는 튜닝셋 전용이다.
case "$MAN" in
  *test750*|*tradeoff250*|*eval250*)
    echo "[치명] 판정셋 매니페스트가 지정됐다: $MAN" >&2; exit 2;;
esac
[ -f "$MAN" ] || { echo "[치명] 매니페스트 없음: $MAN" >&2; exit 2; }

# 물리 노브 전량. 매 조건 실행 직전·직후에 전부 unset 해서 조건 간 상속 오염을 막는다.
KNOBS="MCI_AMB_NUM MCI_UAV_NUM MCI_INCIDENT_SIZE MCI_CAPA_SCALE MCI_AMB_VELOCITY MCI_UAV_VELOCITY MCI_AMB_HANDOVER MCI_UAV_HANDOVER MCI_TREAT_SCALE"

MUS=${MUS:-"0 3 6.6 10 15"}

FAIL=0
FAILED_TAGS=""

# ── 팔 조립기 ──────────────────────────────────────────────────────────────
arms () {
  local s="REF=cardt:18,6.6,0,hingerate" m
  for m in $MUS; do s="$s;FREE$m=cardt2:18,$m,0,0,0,hingerate,on,on"; done
  for m in $MUS; do s="$s;AMBONLY$m=cardt2:18,$m,0,0,0,hingerate,on,on,on"; done
  echo "$s"
}

# ── loadavg 게이트 ─────────────────────────────────────────────────────────
# 조건 시작 전에만 본다 — 실행 중 개입하면 부분 CSV 가 남아 재개가 막힌다.
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
run () {   # 태그  "MCI_* 노브들"  정책스펙
  local tag=$1 envs=$2 pol=$3 rc=0
  local f=$OUT/${tag}.csv
  local lg=$LOG/${tag}.log
  local narm
  narm=$(printf '%s' "$pol" | tr ';' '\n' | grep -c .)

  if [ -f "$f.meta.json" ]; then
    echo "  skip ${tag} (완주 메타 있음)"
    return 0
  fi
  if [ "$DRY" = "1" ]; then
    echo "DRY ${tag}  arms=${narm}  knobs=[${envs}]  eps=$((750 * narm * NEPS))  out=$f"
    echo "    pol=$(printf '%s' "$pol" | cut -c1-160)..."
    return 0
  fi

  loadgate
  for k in $KNOBS; do unset "$k"; done
  for kv in $envs; do export "$kv"; done
  echo "[$(date +%m-%d\ %H:%M:%S)] ${tag} 시작 arms=${narm} knobs=[${envs}] W=$W loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  "$P" src/rl_src/v17_rule_eval.py \
      --manifest "$MAN" --policies "$pol" \
      --n_eps "$NEPS" --seed0 "$SEED0" --workers "$W" --out "$f" > "$lg" 2>&1
  rc=$?
  for k in $KNOBS; do unset "$k"; done
  echo "[$(date +%m-%d\ %H:%M:%S)] ${tag} 종료 rc=$rc  $(tail -1 "$lg" | cut -c1-100)"
  if [ "$rc" -ne 0 ]; then
    FAIL=$((FAIL + 1))
    FAILED_TAGS="$FAILED_TAGS $tag"
    echo "  [실패] ${tag} — 로그: $lg"
  fi
  return 0
}

POL=$(arms)
# base = 정식 물리조건(노브 없음) · ts40 = 봉투 밖(치료시간 ×4). ⑮ 가 잰 두 조건 그대로다.
run base ""                     "$POL"
run ts40 "MCI_TREAT_SCALE=4.0"  "$POL"

if [ "$DRY" = "1" ]; then
  echo "[DRY] coupled_decomp — 실행·마커 없음"
  exit 0
fi
if [ "$FAIL" -eq 0 ]; then
  touch "$OUT/coupled_decomp.DONE"
  echo "[$(date +%m-%d\ %H:%M:%S)] coupled_decomp 완료 — 실패 0건"
else
  printf '%s\n' "${FAILED_TAGS# }" > "$OUT/coupled_decomp.FAILED"
  echo "[$(date +%m-%d\ %H:%M:%S)] coupled_decomp 미완 — 실패 ${FAIL}건:${FAILED_TAGS} (DONE 미기록)"
  exit 1
fi
