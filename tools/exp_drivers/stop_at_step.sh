#!/usr/bin/env bash
# 돌고 있는 ML-Agents 런을 지정 스텝에서 **정상 종료**시킨다(체크포인트·ONNX·timers.json 보존).
#
# 왜 필요한가: `max_steps` 는 기동 시에만 읽히므로 yaml 을 고쳐도 이미 도는 런엔 안 먹는다.
# 그리고 그냥 죽이면(SIGTERM) 미저장 스텝을 날린다 — tmux 로 Ctrl-C 를 보내야 mlagents 가
# 저장하고 끝낸다(비대화형 백그라운드 잡은 SIGINT 를 SIG_IGN 으로 상속해 `kill -INT` 가 무시된다).
#
# ★순서가 핵심: **워치독 센티넬을 먼저 지우고** Ctrl-C 를 보낸다. 안 그러면 워치독이
#   "laststep < yaml 의 max_steps" 로 보고 미완주라 판단해 **곧바로 재기동한다**
#   (구 워치독은 max_steps 를 yaml 에서 읽는데 그 yaml 은 여전히 20M 이다).
#
# 사용: stop_at_step.sh <run_id> <target_step> [sentinel_path]
set -u
R="${ROAD_DRIVE_ROOT:-/home/ryu/roaddrive}"
RUN="${1:?run_id}"
TARGET="${2:?target_step}"
SENT="${3:-$R/.watchdog_on}"
LOG="$R/Logs/${RUN}_stopper.log"
echo "[$(date '+%F %T')] stopper 시작 run=$RUN target=$TARGET sentinel=$SENT" >> "$LOG"

while sleep 120; do
  if ! tmux has-session -t "$RUN" 2>/dev/null; then
    echo "[$(date '+%F %T')] 세션이 이미 없다 — stopper 종료" >> "$LOG"; exit 0
  fi
  s=$(grep -oE 'Step: [0-9]+' "$R/Logs/${RUN}_launcher.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  [ -n "${s:-}" ] || continue
  [ "$s" -lt "$TARGET" ] && continue

  rm -f "$SENT"
  sleep 3
  echo "[$(date '+%F %T')] $s >= $TARGET — 워치독 해제 후 graceful stop" >> "$LOG"
  tmux send-keys -t "$RUN" C-c
  break
done

# 저장 확인: 최대 15분 대기
for i in $(seq 1 90); do
  tmux has-session -t "$RUN" 2>/dev/null || break
  sleep 10
done
{
  echo "[$(date '+%F %T')] 종료 확인 — 산출물:"
  ls -l "$R/results/$RUN/RoadDriving/checkpoint.pt" 2>/dev/null
  ls -lt "$R/results/$RUN/RoadDriving"/*.onnx 2>/dev/null | head -3
  ls -l "$R/results/$RUN/run_logs/timers.json" 2>/dev/null || echo "  timers.json 없음(비정상 종료 가능)"
  echo "  마지막 스텝: $(grep -oE 'Step: [0-9]+' "$R/Logs/${RUN}_launcher.log" | tail -1)"
} >> "$LOG"
