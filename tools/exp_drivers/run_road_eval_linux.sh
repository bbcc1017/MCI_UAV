#!/usr/bin/env bash
# 자율주행 폐루프 평가 — **Linux Dedicated Server 플레이어**로 DS/RC/IP + 종방향 승차감을 낸다.
#
# Windows 판(`run_road_eval.sh`)과 같은 계약이지만 경로·실행형식만 리눅스다.
# ⚠모델은 **빌드 시점 Resources**에서만 로드된다(`Resources.Load<ModelAsset>`) — 파일경로 로드 경로가
#   없으므로, 평가할 onnx 는 빌드 전에 `Assets/Resources/Models/` 에 넣어 둬야 한다.
#   여기 `model` 인자는 그 Resources 상대경로(예 `Models/RoadDriving`)이지 파일경로가 아니다.
#
# 사용: run_road_eval_linux.sh <label> <episodes> [model]
#   model 생략 = 규칙 휴리스틱(기준선). 지정하면 그 onnx 추론.
set -u
LABEL="${1:?label}"; EPISODES="${2:-60}"; MODEL="${3:-}"
ROOT="${ROOT:-/home/ryu/roaddrive}"
EXE="${ENV_BIN:-$ROOT/RoadTrainLinux/UAV_test}"
[ -x "$EXE" ] || { echo "플레이어가 없다: $EXE"; exit 1; }

export CAR_TEST_MODE=eval
export CAR_TEST_HEADLESS=1
export CAR_TEST_EVAL_EPISODES="$EPISODES"
export CAR_TEST_EVAL_LABEL="$LABEL"
export CAR_TEST_TIMESCALE=10
# 학습(L4)과 같은 교통 밀도에서 잰다 — 서지는 밀집 교통에서 드러난다.
export CAR_TEST_NPC_COUNT="${CAR_TEST_NPC_COUNT:-400}"
export CAR_TEST_PED_COUNT="${CAR_TEST_PED_COUNT:-80}"
export CAR_TEST_DATA_ROOT="${CAR_TEST_DATA_ROOT:-$ROOT/data}"
if [ -n "$MODEL" ]; then export CAR_TEST_EVAL_MODEL="$MODEL"; else unset CAR_TEST_EVAL_MODEL || true; fi

LOG="$ROOT/Logs/road_eval_${LABEL}.log"
mkdir -p "$ROOT/Logs" "$ROOT/results"
echo "[eval] $LABEL episodes=$EPISODES model=${MODEL:-heuristic} → $LOG"
"$EXE" -batchmode -nographics -logFile "$LOG"
RC=$?
# 평가기는 Application.dataPath/../results 에 쓴다 = <build>/results/
SRC="$(dirname "$EXE")/results/road_eval_${LABEL}.csv"
if [ -f "$SRC" ]; then cp "$SRC" "$ROOT/results/" && echo "[eval] CSV → $ROOT/results/road_eval_${LABEL}.csv"
else echo "⚠CSV 미생성 — 로그 확인"; fi
# ★실효정책 확인: 규칙 런이 조용히 학습정책 런이 되는 회귀를 여기서 잡는다.
grep -a "\[Drive\] 실효정책=" "$LOG" | tail -1
grep -a "RoadEval" "$LOG" | tail -4
exit $RC
