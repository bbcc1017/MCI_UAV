#!/usr/bin/env bash
# 자율주행 ML-Agents 학습 — Linux 박스(`aigpu0617`) 헤드리스 판.
#
# 왜 Linux 인가: Windows 32코어에서 4 env = **34 steps/s**(20M 에 164시간, env_step 이 wall time 98.2%).
# env 하나가 CPU 한계로 ~1배속이라 `--time-scale` 은 무의미하고 **총 처리량 ≈ num_envs × 10 dec/s** 다.
# 128코어 박스에서 env 를 늘리는 것이 유일한 레버다.
#
# ⚠공동사용 노드다. 실행 전 `uptime` 으로 loadavg 를 보고, 기존 부하 + 우리 부하가 128 을 넘지 않게 한다
#   (실측 기준선 loadavg 48~55). 8 로 시작해 측정 후 12 까지만 올릴 것.
# ⚠`-nographics` 필수 — Dedicated Server 빌드가 아니라 일반 Mono 플레이어라 그래픽 초기화를 막아야 한다.
#   BEV 센서는 CPU 래스터라 그래픽 없이 동작하고, RGB/깊이/열화상 센서는 headless 에서 자동 비활성이다.
set -u
RUN_ID="${1:-korea_drive_v5}"
NUM_ENVS="${2:-8}"
CONFIG="${3:-road_driving_v5.yaml}"
EXTRA="${4:-}"

# ⚠**리포 작업트리를 쓰지 않는다.** `/home/ryu/MCI_UAV` 는 다른 Claude 계정 세션이 v19 증류
#   실험을 진행 중이라(로컬보다 앞선 커밋) 거기에 빌드·결과를 쓰면 그쪽 작업을 오염시킨다.
#   플레이어·결과·로그는 전부 이 전용 디렉터리 안에서만 다룬다.
ROOT="${ROAD_DRIVE_ROOT:-/home/ryu/roaddrive}"
EXE="${ENV_BIN:-$ROOT/RoadTrainLinux/UAV_test}"   # ENV_BIN 으로 다른 빌드 지정 가능(파인튠·A/B 를 같은 박스에서 병렬로 돌릴 때 필수)
# TRAINER GPU — 이유와 실측(2026-09-02, 공동사용자가 GPU 100% 점유 중에 측정)
#  · shared_critic 기본값이 false 라 크리틱이 자기 nature_cnn+LSTM 을 따로 갖는다.
#    학습 미니배치(b2048) CPU 3,496ms vs GPU 41ms = **85배**.
#  · `threaded: false` 기본값이라 업데이트가 env 스텝을 **완전히 막는다**.
#    로그 실측 교대 패턴 51s/82s → 스톨 31초 / 주기 133.5초 = **wall 의 22.8%**.
#    GPU 로 그게 0.4초가 된다 → 처리량 150 → 194 steps/s.
#  ⚠추론(batch 8)은 CPU 3.2ms vs GPU 2.7ms 로 **차이 없다**(커널 런치 오버헤드 지배).
#    "비전이라서 GPU" 가 아니라 "2048 배치 역전파라서 GPU" 다.
#  ⚠OMP_NUM_THREADS=1 은 env 워커용 관례였는데 트레이너에도 걸려 CPU 학습을 느리게 했다.
#    GPU 에서는 무관하니 그대로 둔다(Unity 플레이어에게 코어를 양보하는 편이 낫다).
PY_ENV="${PY_ENV:-mlagents_gpu}"
TORCH_DEVICE="${TORCH_DEVICE:-cuda}"
PY=/home/ryu/anaconda3/envs/$PY_ENV/bin
# ⚠`-x` 로 보지 말 것 — Windows 에서 rsync 로 옮기면 실행비트가 없을 수 있다. 존재로 판정하고 부여한다.
[ -f "$EXE" ] || { echo "플레이어가 없다: $EXE (Windows 에서 빌드 후 rsync)"; exit 1; }
chmod +x "$EXE" 2>/dev/null || true

LOAD=$(awk '{print int($1)}' /proc/loadavg)
echo "[train] 현재 loadavg=$LOAD · 요청 num_envs=$NUM_ENVS"
if [ "$LOAD" -gt 100 ]; then echo "⚠loadavg 100 초과 — 공동사용자 부하가 높다. 중단."; exit 2; fi

# ★재개 시 디바이스 전환 금지 가드 — 실측 사고(2026-09-02): GPU 로 저장한 checkpoint.pt 는
#  **CUDA 텐서**를 담는다. CPU-only torch 로 `--resume` 하면 torch.load 가 map_location 없이
#  역직렬화하다 RuntimeError 로 죽는데, mlagents 는 그걸 새 런으로 처리해
#  **checkpoint.pt 를 step 0 모델로 덮어쓴다**(979,042 스텝을 날릴 뻔했다. 번호 붙은
#  스냅샷 RoadDriving-<step>.pt 가 남아 복구했다). 학습 전에 막는다.
STATUS="$ROOT/results/$RUN_ID/run_logs/training_status.json"
case "${EXTRA:-}" in *--resume*)
  if [ -f "$STATUS" ]; then
    SAVED_TORCH=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('metadata',{}).get('torch_version',''))" "$STATUS" 2>/dev/null)
    case "$SAVED_TORCH:$TORCH_DEVICE" in
      *cu*:cpu) echo "⚠체크포인트가 CUDA($SAVED_TORCH)로 저장됐는데 device=cpu 로 재개하려 한다."
                echo "  그대로 두면 checkpoint.pt 가 step 0 으로 덮인다. TORCH_DEVICE=cuda 로 실행하거나"
                echo "  torch.load(map_location='cpu') 로 변환한 뒤 재시도할 것."; exit 4;;
    esac
  fi
;; esac

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export CAR_TEST_MODE=train CAR_TEST_HEADLESS=1
export CAR_TEST_NPC_COUNT="${CAR_TEST_NPC_COUNT:-400}"
export CAR_TEST_PED_COUNT="${CAR_TEST_PED_COUNT:-80}"
# 차선그래프/보행 bin — StreamingAssets 에 이미 있지만 외부 루트를 우선 탐색한다.
# 차선그래프는 StreamingAssets 에 동봉돼 있다. 외부 루트를 굳이 리포로 잡지 않는다(다른 계정 작업 보호).
export CAR_TEST_DATA_ROOT="${CAR_TEST_DATA_ROOT:-$ROOT/data}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# ★시험로 지역·스폰을 **명시적으로** 넘긴다.
#  실측(2026-09-06): 파인튠 프로세스가 CAR_TEST_REGION=pg_track01 을 갖고 있었는데 런처는 그걸
#  export 한 적이 없다 — `tmux new-session` 은 클라이언트가 아니라 **tmux 서버**의 환경을 물려주므로
#  전날 세션이 서버에 남긴 값이 조용히 따라온 것이다. 서버가 한 번 죽으면 다음 학습은 아무 경고 없이
#  기본 지역(강남)으로 떨어지고, 빌드에 그 씬이 없어 차량을 못 만들어 스텝이 0 이 된다.
#  평가 하네스가 실제로 그 함정을 밟았다(지역=seoul_gangnamgu 로 부팅 → 차량 미생성 → 종료).
export CAR_TEST_REGION="${CAR_TEST_REGION:-pg_track01}"
export CAR_TEST_SPAWN_LAT="${CAR_TEST_SPAWN_LAT:-37.501951}"
export CAR_TEST_SPAWN_LON="${CAR_TEST_SPAWN_LON:-127.036401}"

# ★nohup 이 아니라 tmux 로 띄운다 — **graceful stop 을 위해서**.
#  비대화형 셸의 백그라운드 잡(`cmd &`)은 POSIX 규정상 SIGINT/SIGQUIT 를 SIG_IGN 으로 상속받고,
#  CPython 은 시작 시 그 disposition 이 SIG_IGN 이면 자기 KeyboardInterrupt 핸들러를 **설치하지 않는다**.
#  결과: `kill -INT` 가 영원히 무시된다(실측 2026-09-02, 3회 시도 전부 무시 → 체크포인트 없이
#  SIGTERM 으로 죽일 수밖에 없었고 미저장 스텝을 날렸다). bash `trap - INT` 로도 못 되돌린다
#  ("이미 무시로 진입한 시그널은 트랩·리셋 불가"). tmux 는 제어 터미널을 주므로 정상 disposition 이다.
#  정지 = `tmux send-keys -t <run_id> C-c` → mlagents 가 체크포인트·ONNX·timers.json 을 쓰고 종료.
cd "$ROOT"
mkdir -p Logs results
echo "[train] run_id=$RUN_ID config=$CONFIG envs=$NUM_ENVS device=$TORCH_DEVICE"
# tmux 는 새 세션에 **클라이언트가 아니라 서버**의 환경을 준다 → 여기 구운 파일을 CMD 가 source 한다.
CENV="Logs/${RUN_ID}_env.sh"
: > "$CENV"
for v in CAR_TEST_MODE CAR_TEST_HEADLESS CAR_TEST_NPC_COUNT CAR_TEST_PED_COUNT CAR_TEST_DATA_ROOT CAR_TEST_REGION CAR_TEST_SPAWN_LAT CAR_TEST_SPAWN_LON OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS PYTHONUTF8 PYTHONIOENCODING; do
  [ -n "${!v-}" ] || continue   # 미설정은 굽지 않는다 — 빈 문자열은 C# 쪽에서 null 이 아니라 "" 로 읽혀 분기를 바꾼다
  echo "export $v=$(printf %q "${!v-}")" >> "$CENV"
done
CMD="Logs/${RUN_ID}_cmd.sh"
{
  echo '#!/usr/bin/env bash'
  echo "source '$ROOT/Logs/${RUN_ID}_env.sh'"
  echo "exec '$PY/mlagents-learn' '$CONFIG' \\"
  echo "  --run-id='$RUN_ID' --env='$EXE' \\"
  echo "  --num-envs='$NUM_ENVS' --base-port='${BASE_PORT:-6500}' --no-graphics \\"
  echo "  --results-dir='$ROOT/results' \\"
  echo "  --time-scale=10 --torch-device='$TORCH_DEVICE' $EXTRA \\"
  echo "  >> 'Logs/${RUN_ID}_launcher.log' 2>&1"
} > "$CMD"
chmod +x "$CMD"

tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "bash $CMD"
sleep 5
if tmux has-session -t "$RUN_ID" 2>/dev/null; then
  echo "[train] tmux:$RUN_ID 기동 — tail -f $ROOT/Logs/${RUN_ID}_launcher.log"
  echo "[train] 정지: tmux send-keys -t $RUN_ID C-c   (체크포인트·ONNX·timers.json 저장 후 종료)"
else
  echo "[train] tmux 세션이 안 떴다:"; tail -20 "Logs/${RUN_ID}_launcher.log"; exit 3
fi
