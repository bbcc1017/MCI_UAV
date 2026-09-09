#!/usr/bin/env bash
# 학습이 죽으면 체크포인트에서 자동 재개한다. (v2 — 런 여러 개를 같은 박스에서 돌릴 수 있게 일반화)
#
# 왜: env 워커 하나가 멈추면 mlagents 가 `TimeoutError: Workers {N} stuck in waiting state`
# 로 **런 전체를 죽인다**. 근본 수정(C# 스폰 폴백)은 재빌드가 필요하고 재빌드는 학습 중
# 동역학을 바꾸므로 런이 끝난 뒤에 한다. 그때까지 손실을 분 단위로 묶는 게 이 스크립트다.
#
# ★v1 에서 고친 것 — v1 을 두 번째 런에 그대로 붙이면 **다른 런을 죽인다**:
#   ①센티넬이 전역 파일 하나(`.watchdog_on`)라 한쪽을 끄면 양쪽이 꺼진다 → 런별로 분리.
#   ②재개가 `BASE_PORT=6600` 하드코딩 + `ENV_BIN` 미전달 → 파인튠이 **다른 빌드·다른 포트**로
#     되살아난다(실측 사고: 구 빌드로 재기동돼 보상 변경이 통째로 무효였다).
#   ③좀비 청소가 `RoadTrain[L]inux` 고정 패턴이라 **다른 런의 플레이어 12개를 몰살**한다.
#     → 자기 EXE 경로에서 패턴을 유도한다.
#
# 의도적 정지: **먼저 sentinel 을 지우고** tmux 로 Ctrl-C.
#   rm /home/ryu/roaddrive/.watchdog_on_<run> && tmux send-keys -t <run> C-c
set -u
ROOT="${ROAD_DRIVE_ROOT:-/home/ryu/roaddrive}"
RUN="${1:-korea_drive_v6}"
ENVS="${2:-12}"
CFG="${3:-road_driving_v5.yaml}"
EXE="${ENV_BIN:-$ROOT/RoadTrainLinux/UAV_test}"
PORT="${BASE_PORT:-6600}"
SENT="${WATCHDOG_SENTINEL:-$ROOT/.watchdog_on_$RUN}"
LOG="$ROOT/Logs/${RUN}_watchdog.log"
MAXR="${MAXR:-30}"
INTERVAL="${INTERVAL:-120}"

# 좀비 청소 패턴을 EXE 디렉터리명에서 유도한다. 첫 글자를 대괄호로 감싸 자기 명령줄 자기매칭을 피한다
# (CLAUDE.md 의 pgrep 자살 함정 — 실측으로 원격 셸을 한 번 죽였다).
EXEDIR=$(basename "$(dirname "$EXE")")
ZPAT="[${EXEDIR:0:1}]${EXEDIR:1}/$(basename "$EXE")"

MAXSTEPS=$(awk "/max_steps:/{gsub(/.*max_steps:[ \t]*/,\"\");gsub(/[ \t].*/,\"\");printf \"%d\",\$0+0;exit}" "$ROOT/$CFG" 2>/dev/null)

if [ "${1:-}" = "--selftest" ]; then
  t=$(mktemp); printf "behaviors:\n  X:\n    max_steps: 2.0e7\n" > "$t"
  got=$(awk "/max_steps:/{gsub(/.*max_steps:[ \t]*/,\"\");gsub(/[ \t].*/,\"\");printf \"%d\",\$0+0;exit}" "$t")
  [ "$got" = "20000000" ] || { echo "FAIL 파싱: $got"; exit 1; }
  [ 20000105 -ge "$got" ] || { echo "FAIL 완주판정"; exit 1; }
  [ 19999999 -ge "$got" ] && { echo "FAIL 미완주를 완주로 오판"; exit 1; }
  # 좀비 패턴이 자기 EXE 만 잡고 남의 런을 안 잡는지
  E=/home/ryu/roaddrive/RTL_ft/UAV_test; D=$(basename "$(dirname "$E")")
  P="[${D:0:1}]${D:1}/$(basename "$E")"
  echo "/home/ryu/roaddrive/RoadTrainLinux/UAV_test" | grep -q "$P" && { echo "FAIL 남의 런을 잡는다"; exit 1; }
  echo "/home/ryu/roaddrive/RTL_ft/UAV_test" | grep -q "$P" || { echo "FAIL 자기 런을 못 잡는다"; exit 1; }
  rm -f "$t"; echo "selftest OK (max_steps=$got · zombie=$P)"; exit 0
fi

touch "$SENT"
echo "[$(date "+%F %T")] watchdog2 시작 run=$RUN envs=$ENVS cfg=$CFG exe=$EXE port=$PORT max=$MAXSTEPS (해제: rm $SENT)" >> "$LOG"
n=0
while [ -f "$SENT" ]; do
  sleep "$INTERVAL"
  [ -f "$SENT" ] || break
  if tmux has-session -t "$RUN" 2>/dev/null; then continue; fi
  laststep=$(grep -oE "Step: [0-9]+" "$ROOT/Logs/${RUN}_launcher.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  if [ -n "$laststep" ] && [ -n "$MAXSTEPS" ] && [ "$laststep" -ge "$MAXSTEPS" ]; then
    echo "[$(date "+%F %T")] 목표 스텝 도달($laststep/$MAXSTEPS) — 재개하지 않음" >> "$LOG"; break
  fi
  n=$((n+1))
  if [ "$n" -gt "$MAXR" ]; then
    echo "[$(date "+%F %T")] 재기동 $MAXR 회 초과 — 중단(사람이 봐야 함)" >> "$LOG"; break
  fi
  why=$(grep -oE "[A-Za-z]*Error: .*" "$ROOT/Logs/${RUN}_launcher.log" 2>/dev/null | tail -1)
  echo "[$(date "+%F %T")] 죽음 감지 (#$n) Step: ${laststep:-?} | $why → 재개" >> "$LOG"
  for pid in $(ps -eo pid,args --no-headers | grep "$ZPAT" | awk "{print \$1}"); do kill -9 "$pid" 2>/dev/null; done
  sleep 5
  ( cd "$ROOT" && PY_ENV=mlagents_gpu TORCH_DEVICE=cuda BASE_PORT="$PORT" ENV_BIN="$EXE" \
      bash run_road_drive_linux.sh "$RUN" "$ENVS" "$CFG" --resume >> "$LOG" 2>&1 )
done
echo "[$(date "+%F %T")] watchdog2 종료" >> "$LOG"
