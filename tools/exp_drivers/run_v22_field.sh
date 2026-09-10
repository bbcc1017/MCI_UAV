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
#   treat    조건 6             · 팔 45 — ★치료시간 축. 이번 주 헤드라인의 정본(§8)
#   yhold    조건 12            · 팔 15 — 등급 축(Y/R) + S족 등급 변형(SY)
#   red      조건 5             · 팔 15 — 수단 축(G/D). v20 14조건에서 축소(§9)
#   extreme  조건 7             · 팔 54 — v20 42팔 + λ 격자 확장 12팔
#   lamx     조건 5             · 팔 20 — (λ × yhold) 결합 셀. v20 은 두 축을 따로만
#                                        쓸었다 → 축 독립 가설의 직접 검증
# 조건별 MCI_* 노브 값은 추측이 아니라 v20 산출 메타
# (results/scoreboard/v20/theory/{treat,yhold,red,extreme}_*.csv.meta.json 의
# scenario_knobs)와 run_v20_theory.sh 원문에서 읽어 그대로 옮겼다.
#
# 회귀 게이트(부수 효과): base 스테이지의 Q18·K12 는
# results/scoreboard/v20/budget/lam_base.csv 및
# results/scoreboard/v20/fieldinfo/budget750.csv 와 **같은 매니페스트·같은 seed 0..9**
# 라서 좌표·시드가 짝이 맞는다. 두 파일과 값이 일치해야 한다(불일치 = 코드/노브 변동
# 신호). 반면 results/scoreboard/v20/theory/* 는 tradeoff250 이라 행을 섞은 paired
# 결합이 불가능하다 — 조건이 같아도 좌표가 다르면 짝이 아니다.
#
# ── §6. 규모 추정 (추정이며 실측이 아니다 — 근거만 실측) ───────────────────
# 에피소드 수 = 조건 × 팔 × 750 좌표 × 10 ep
#   base 135,000 / treat 2,025,000 / yhold 1,350,000 / red 562,500 /
#   lamx 750,000 / extreme 2,835,000                    합계 7,657,500 ep
#
# (a) 프로덕션 실측 처리량 550 ep/s(원본 코어·46워커, 84~93 core-ms/ep) 기준
#     base 0.07h · treat 1.02h · yhold 0.68h · red 0.28h · lamx 0.38h · extreme 1.43h
#     → 합 약 3.9h
# (b) ★E0 계측(median cpu ms/ep)으로 조건별 비용을 직접 곱한 기준 = 가장 정확하다.
#     E0 실측: 에피소드 비용은 사고규모 N 에 **정확히 선형**(로그-로그 기울기 0.989,
#     결정 수 0.979, 결정당 1.7 ms 일정). N50 36.1 · N100 66.6 · N200 137.5 ·
#     N300 209.6 · N500 343.1 ms/ep. N 이외 노브(ts·capa·vamb·amb·uav)는 v20 wall
#     로그상 기본조건 대비 ±20% 이내라 여기서는 N 만 반영했다.
#       base 2.5 · treat 37.5 · yhold 26.2 · red 10.4 · lamx 13.9 · extreme 83.6
#       → 합 174 core-h = 48워커에서 3.6h (오버헤드 0 가정한 하한)
#     n500 단독이 38.6 core-h = extreme 의 46% · 전 캠페인의 22% 다. 그래서 마지막에 둔다.
# (c) 오버헤드 교차검증: v20 wall 에서 유도한 core-ms/ep 는 기본조건 95~114 인데
#     E0 cpu 는 66.6 이다 → 스케줄링·I/O·적재 오버헤드 배율 1.43~1.71.
#     (b)에 1.5 를 곱한 **약 5.4h @48워커**가 현실적인 기대치다.
#     오버헤드가 붙는 이유 둘: ① 좌표당 시나리오 적재 고정비가 팔 수로 나눠지므로
#     팔이 적은 스테이지의 ep당 비용이 커진다(v20 11팔 블록 141 core-ms/ep).
#     ② maxtasksperchild=1 이라 좌표마다 워커가 새로 뜬다.
# treat 신설(+2.03M ep)과 red 축소(−1.01M ep)의 순증은 +1.01M ep = +15% 다
# ("총 wall 대략 유지"). P족 아래끝 보강(P4·P6)은 그중 90,000 ep = 1.2%.
# 산출 용량: 7.66M 행 × 약 120 B ≈ 0.9 GB (평가기 스키마에 열이 늘면 더 커진다).
#
# ── §7. 사용 ───────────────────────────────────────────────────────────────
#   bash tools/exp_drivers/run_v22_field.sh <base|treat|yhold|red|extreme|lamx> [워커수]
#   DRY=1 을 주면 조건·팔 수·예상 에피소드만 출력하고 아무것도 실행하지 않는다.
#   권장 순서: base → treat → yhold → lamx → extreme
#     헤드라인 데이터(base·treat)를 먼저 확보하고 가장 비싼 extreme(n500 실측 5.1배)을
#     마지막에 둔다. 중간에 끊겨도 논문 표의 본체는 남는다.
#     red 는 죽은 축(§9)이라 우선순위 최하 — 여유가 있을 때 마지막에 붙인다.
#   재개: 조건별 <stage>_<tag>.csv.meta.json 이 있으면 그 조건을 건너뛴다.
#         메타 없이 CSV 만 남은 부분 기록은 평가기가 RuntimeError 로 막으므로
#         해당 CSV 를 수동 확인·정리한 뒤 다시 돌린다(operations.md §장시간 작업 6).
#   실패 처리: rc≠0 인 조건이 하나라도 있으면 <stage>.DONE 을 쓰지 않고
#              <stage>.FAILED 에 태그를 남기고 exit 1 한다("실패해도 DONE" 함정 방지).
#
# ── §8. treat 스테이지: 왜 신설하고 무엇을 사전등록하는가 ──────────────────
# v20 은 치료시간 축을 돌려놓고도(results/scoreboard/v20/theory/treat_*.csv, 6조건)
# ① 그 CSV 가 **tradeoff250 = 판정셋 부분집합**이고
# ② v20 스케일링 표에 치료시간 행이 없고 results/scoreboard/v20/optima.json 의
#    조건 28개에 ts/treat 항목이 **하나도 없다**(실측 확인).
# 즉 "λ 는 평균 서비스시간이다" 라는 이 연구의 헤드라인 주장에서 정작 서비스시간 축이
# 누수 좌표 위의 미기록 CSV 로만 존재한다. 그래서 좌표셋 정정 + 격자 확장을 함께 한다.
#
# v20 CSV 재집계(로그축 포물선 보간, 독립 재계산으로 코디네이터 수치 재현):
#   ts        0.5    0.75    1.0    1.5    2.0    3.0
#   Q λ*      7.88  13.52  18.57  30.70  41.57  50.00 ←격자 끝 절단
#   H λ*      4.00e  7.84  11.23  17.62  22.38  34.57  (ts=0.5 는 격자 아래끝 H4)
#   S w*      0.52   0.58   0.62   0.72   0.66   0.74
#   기울기 d log λ*/d log ts : Q +1.064(전점) / +1.199(격자끝 제외)   이론 +1
#                              H +1.170 / +1.055,  P +1.183,  S +0.187 ← 사실상 평평
# 해석: Q·H·P 는 이론 기울기 +1 을 따라가고(λ 는 튜닝 상수가 아니라 서비스시간),
# S족은 서비스시간을 명부에서 직접 읽으므로 재튜닝이 필요 없다(평평).
# ts=3.0 의 최적이 격자 끝 Q50 이라 그 점을 넣으면 기울기가 **낮게 편향**되고,
# ts=0.5 에서는 H·P 가 격자 **아래끝**에서 절단됐다 → 양쪽을 다 넓힌다.
#
# ★사전등록 예측: ts=4.0 의 Q λ* 는 **79 ~ 98** 구간에 들어온다
#   (ts=1.0 의 λ*=18.57 에 기울기 +1.064 적용 → 81.2, +1.199 적용 → 97.9).
#   확장 격자 {65, 85, 110, 140} 이 이 구간을 감싼다. 들어오면 "λ = 평균 서비스시간
#   (할인 0.61)" 이 봉투 밖으로 외삽된다는 뜻이고, 벗어나면 벗어난 값을 그대로 기록한다.
#   게이트 실패를 격자 재조정으로 사후 구제하지 않는다(AGENTS.md 계약 7).
# 양쪽 절단 방지: P족 격자는 아래끝을 4·6 까지 내려 {4,6,10,18,26,40,65,110} 이다.
#   P족 기울기 +1.183 으로 λ*(ts=0.5) ≈ 7.6 이 예측되므로 6 < 7.6 < 10 으로 감싸진다.
#   아래끝을 10 에 두면 그 점이 절단돼 P족 기울기가 위로 편향되고, 이는 인용값 +1.06 이
#   위쪽 절단(Q50)으로 아래로 편향된 것과 같은 실수의 반대 방향 재발이다.
#   Q 는 3~140, H 는 2~150 이라 ts 0.5~4.0 전 구간에서 양끝이 열려 있다.
#
# ── §9. red 스테이지를 14조건 → 5조건으로 줄인 근거 ────────────────────────
# 수단 축은 거의 죽은 축이다. tradeoff250 CRN paired 재확인 결과 채택값 G6.6 대비
# 그 조건 최적 팔의 이득이 최대 +0.00055(huav5) · 중위 +0.00011 로 대부분
# 판정선(CRN paired 0.00053) 미만이다. 14조건 × 15팔 = 1.58M ep 을 쓸 축이 아니다.
# 이득이 조금이라도 보인 base·huav5·hamb10·vamb70·uav6 만 남기고 9조건을 뺐다.
# 절약분 약 1.01M ep 이 treat 신설분 1.94M ep 의 절반을 상환한다.
set -u

STAGE=${1:?stage: base|treat|yhold|red|extreme|lamx}
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

# treat 스테이지 전용 λ 격자. 근거·사전등록은 머리말 §8.
# v20 격자는 위(ts=3.0 → Q50)와 아래(ts=0.5 → H4·P6) 양쪽에서 절단됐으므로
# Q 는 3 까지 내리고 140 까지, H 는 2 까지 내리고 150 까지 넓힌다.
# S 는 λ 가 없는 무튜닝 카드라 wait_scale 만 훑는다(0.4·1.25 신규 = 최적점 0.52~0.74 포위).
# P 는 값 자체보다 **기울기**를 재는 팔이라 격자를 성기게 잡되 양쪽 끝을 비워두지 않는다.
#   ★P족(hingerate_psent)은 병원 통신이 필요 없는 형태 = 현장 배포에서 실제로 쓸 후보다.
#   그 형태에서도 λ 가 서비스시간에 비례하면 "λ 는 명부에서 읽는다" 는 주장이 통신 없이도
#   성립한다. 그래서 P족 기울기는 정확히 재야 한다.
#   아래끝 4·6 은 그 때문에 추가했다: P족 기울기 +1.183 으로 λ*(ts=0.5) ≈ 7.6 이 나오는데
#   격자 아래끝이 10 이면 그 점이 절단돼 기울기가 **위로** 편향된다. 위쪽 절단(ts=3.0 의
#   Q50)이 인용값 +1.06 을 **아래로** 편향시킨 것과 똑같은 실수를 반대쪽에서 반복하는 셈이다.
#   6 < 7.6 < 10 으로 감싸는 비용은 조건당 15,000 ep(총 90,000 = 전체의 1.2%)뿐이다.
arms_treat () {
  local s="" l w
  for l in 3 4 6 10 14 18 22 26 32 40 50 65 85 110 140; do s="$s;Q$l=cardt:$l,6.6,0,hingerate"; done
  for l in 2 3 4 8 12 16 21 27 35 45 60 80 110 150;     do s="$s;H$l=cardt:$l,6.6,0,hinge1"; done
  for w in 0.25 0.4 0.5 0.62 0.75 1.0 1.25 1.5;         do s="$s;S$w=cards:$w,6.6,0"; done
  for l in 4 6 10 18 26 40 65 110;                      do s="$s;P$l=cardt:$l,6.6,0,hingerate_psent"; done
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

  treat)  # ★치료시간 축. λ = 평균 서비스시간 주장의 정본(§8). ts=1.0 은 base 가 담당하므로
          # 여기서 중복 실행하지 않는다. ts=4.0 은 extreme_ts40 과 같은 조건이지만
          # 이쪽은 λ 격자가 촘촘해서 최적점 자체를 잡는 용도다.
    POL=$(arms_treat)
    for spec in "ts05:MCI_TREAT_SCALE=0.5" "ts075:MCI_TREAT_SCALE=0.75" \
                "ts15:MCI_TREAT_SCALE=1.5" "ts20:MCI_TREAT_SCALE=2.0" \
                "ts30:MCI_TREAT_SCALE=3.0" "ts40:MCI_TREAT_SCALE=4.0"; do
      run "${spec%%:*}" "${spec#*:}" "$POL"
    done
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

  red)    # 수단 축 재도출. v20 14조건 → 5조건으로 축소했다(근거 §9: 죽은 축).
          # 남긴 조건은 tradeoff250 에서 채택값 대비 이득이 조금이라도 보인 곳뿐이다.
          # 뺀 9조건: vamb30 vamb100 vuav100 vuav150 vuav300 vuav400 huav20 uav1 uav13
    POL=$(arms_red)
    for spec in "base:" "huav5:MCI_UAV_HANDOVER=5" "hamb10:MCI_AMB_HANDOVER=10" \
                "vamb70:MCI_AMB_VELOCITY=70" "uav6:MCI_UAV_NUM=6"; do
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

  *) echo "unknown stage: $STAGE (base|treat|yhold|red|extreme|lamx)" >&2; exit 1;;
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
