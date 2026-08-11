"""경로 헬퍼 — 신·구 코어를 한 프로세스에 올리기 위한 sys.path 준비.

주의: `src/sim_src` 를 sys.path 에 넣으면 flat `import EventManager` 가 **구 코어**로
해석된다. 이는 의도된 것이다 — rl_src 래퍼들이 그렇게 import 하고, 동치검증도
구 코어를 그 경로로 로드한다. 신 코어는 패키지 상대 import 라 전혀 간섭받지 않는다.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC = os.path.join(REPO, "src")
SIM_SRC = os.path.join(SRC, "sim_src")
RL_SRC = os.path.join(SRC, "rl_src")


def _prepend(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_paths(rl: bool = True, old_sim: bool = True) -> None:
    """`sim_src_upgrade` 패키지 import 경로(+ 선택적으로 rl_src / 구 sim_src)를 보장."""
    _prepend(SRC)
    if rl:
        _prepend(RL_SRC)
    if old_sim:
        _prepend(SIM_SRC)
