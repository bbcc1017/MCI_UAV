# -*- coding: utf-8 -*-
"""전국 fetch 감시자 — 완료되면 종료(0), 멈추면 fetch를 분리 프로세스로 재가동.

사용: python tools/nationwide/monitor.py   (백그라운드 권장)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
PY = sys.executable
ALLOC = os.path.join(HERE, "alloc.json")
BLOCKS = os.path.join(HERE, "blocks")
FETCH_LOG = os.path.join(HERE, "fetch.log")
FETCH_ERR = os.path.join(HERE, "fetch.err.log")

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def remaining():
    with open(ALLOC, encoding="utf-8") as fp:
        alloc = json.load(fp)
    blocks = [tuple(b) for b in alloc["unique_blocks"]]
    nt = sum(1 for b in blocks
             if not os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg"))
             and not os.path.exists(os.path.join(BLOCKS, f"m_{b[0]}_{b[1]}.jpg.unavailable")))
    nb = sum(1 for b in blocks
             if not os.path.exists(os.path.join(BLOCKS, f"b_{b[0]}_{b[1]}.json")))
    return nt, nb, len(blocks)


def relaunch():
    with open(FETCH_LOG, "a", encoding="utf-8") as out, \
         open(FETCH_ERR, "a", encoding="utf-8") as err:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        subprocess.Popen([PY, os.path.join(TOOLS, "nationwide_build.py"), "fetch"],
                         cwd=os.path.dirname(TOOLS), stdout=out, stderr=err,
                         env=env, creationflags=DETACHED)
    print("[monitor] fetch 재가동", flush=True)


def main():
    stall = 0
    prev = None
    restarts = 0
    while True:
        nt, nb, total = remaining()
        print(f"[monitor] 잔여 — 모자이크 {nt}, 건물 {nb} / {total}", flush=True)
        if nt == 0 and nb == 0:
            print("[monitor] 전체 페치 완료", flush=True)
            return
        cur = (nt, nb)
        if cur == prev:
            stall += 1
            if stall >= 3:   # 15분 무진전 → 재가동
                if restarts >= 100:
                    sys.exit("[monitor] 재가동 한도 초과 — 수동 확인 필요")
                relaunch()
                restarts += 1
                stall = 0
        else:
            stall = 0
        prev = cur
        time.sleep(300)


if __name__ == "__main__":
    main()
