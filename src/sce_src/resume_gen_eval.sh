#!/bin/sh
# 평가점 시나리오 생성 재개 — Kakao 일일 할당량 리셋 후 cron 으로 호출.
#
# gen_eval_points.py 는 KAKAO_API_KEY(_2,_3,...) 다중 키를 순환 사용하고
# slot 단위 resume 이라, 매일 자정(KST) 이후 한 번 돌리면 남은 점만 이어
# 생성한다. 85개가 다 차면 hold-out 평가까지 실행하고 완료 마커를 남긴다.
#
# crontab 예: 10 0 * * * /…/src/sce_src/resume_gen_eval.sh >> /tmp/gen_cron.log 2>&1
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO" || exit 1

PY=/home/RYU/anaconda3/envs/UAV/bin/python
DONE_MARKER="$REPO/results/.gen_eval_done"   # results/ 는 gitignore

[ -f "$DONE_MARKER" ] && { echo "[$(date)] 이미 완료(marker 존재) — 종료"; exit 0; }

# Kakao 키를 ~/.bashrc 에서 주입 (비대화형 셸은 .bashrc 를 안 읽음)
eval "$(grep '^export KAKAO_API_KEY' "$HOME/.bashrc")"
export PYTHONIOENCODING=utf-8

echo "[$(date)] 시나리오 생성 재개"
"$PY" src/sce_src/gen_eval_points.py
GEN_RC=$?

N=$("$PY" -c "import json;m=json.load(open('scenarios/plan1nat_eval_manifest.json',encoding='utf-8'));print(sum(len(v) for v in m.values()))" 2>/dev/null)
echo "[$(date)] 생성 진행: ${N:-?}/85 (gen rc=$GEN_RC)"

if [ "${N:-0}" = "85" ]; then
    echo "[$(date)] 85/85 완료 → hold-out 평가 실행"
    env CUDA_VISIBLE_DEVICES="" MCI_REDUCED_OBS=1 PYTHONIOENCODING=utf-8 \
        "$PY" src/rl_src/eval_random_points.py --n_episodes 1000 --max_parallel 24
    if [ $? -eq 0 ]; then
        touch "$DONE_MARKER"
        echo "[$(date)] hold-out 평가 완료 — 작업 종료. crontab 라인 제거 가능."
    fi
else
    echo "[$(date)] 미완 — 다음 자정 리셋 후 자동 이어받기."
fi
