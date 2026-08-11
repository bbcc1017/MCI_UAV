"""G1 궤적 계측 — 구/신 코어 어느 쪽도 **파일을 수정하지 않고** 이벤트열·액션열을 뽑는다.

방법: `EventManager` 클래스의 `ev_*` 핸들러와 `run_next` 를 런타임에 감싼다.
`run_next` 는 `getattr(self, "ev_"+name)` 로 핸들러를 부르므로, 클래스 속성을 바꾸면
실행 경로가 그대로 잡힌다. 핸들러 진입 시점엔 `self.time` 이 이미 그 이벤트 시각으로
갱신돼 있어 `(time, ev_name, entity_idx)` = 팝된 이벤트와 동치다.

기록 값은 **정규화**한다 — numpy 스칼라는 `.item()`, float 은 `float.hex()`.
최적화로 dtype 이 np.int64→int 로 바뀌어도 값이 같으면 같은 문자열이 나오게 하기 위함이며,
동시에 float 은 hex 라 last-bit 차이까지 잡힌다.

⚠️ 계측은 검증 전용이다. 벤치·실주행에서는 절대 호출하지 마라(오버헤드).
"""
from __future__ import annotations

import hashlib

import numpy as np

_ACTIVE: dict = {"rec": None}


class Recorder:
    """한 에피소드의 (이벤트열 + 액션열) 기록."""

    __slots__ = ("items",)

    def __init__(self):
        self.items: list[str] = []

    def digest(self) -> str:
        h = hashlib.sha256()
        for it in self.items:
            h.update(it.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def __len__(self) -> int:
        return len(self.items)


def set_recorder(rec: Recorder | None) -> None:
    _ACTIVE["rec"] = rec


def _canon(x) -> str:
    """dtype 비의존·정밀 무손실 정규화."""
    if isinstance(x, (tuple, list)):
        return "(" + ",".join(_canon(v) for v in x) + ")"
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bool):
        return "T" if x else "F"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return float.hex(x)
    if x is None:
        return "N"
    return repr(x)


def instrument(evm_cls) -> None:
    """`EventManager` 클래스 하나를 계측(멱등)."""
    if getattr(evm_cls, "_trace_instrumented", False):
        return

    for name in [n for n in dir(evm_cls) if n.startswith("ev_")]:
        orig = getattr(evm_cls, name)
        if not callable(orig):
            continue

        def _make(nm, fn):
            def wrapper(self, log, entity_idx):
                rec = _ACTIVE["rec"]
                if rec is not None:
                    rec.items.append("E|" + _canon(self.time) + "|" + nm + "|" + _canon(entity_idx))
                return fn(self, log, entity_idx)
            return wrapper

        setattr(evm_cls, name, _make(name, orig))

    orig_run_next = evm_cls.run_next

    def run_next(self, action=None):
        rec = _ACTIVE["rec"]
        if rec is not None:
            rec.items.append("A|" + _canon(action))
        return orig_run_next(self, action)

    evm_cls.run_next = run_next
    evm_cls._trace_instrumented = True


def instrument_both():
    """구 코어(flat `EventManager`)와 신 코어(패키지)를 모두 계측하고 클래스를 돌려준다."""
    from .._paths import ensure_paths

    ensure_paths()
    import EventManager as old_mod  # 구 코어 (src/sim_src)
    from ..core import EventManager as new_mod  # 신 코어

    instrument(old_mod.EventManager)
    instrument(new_mod.EventManager)
    return old_mod.EventManager, new_mod.EventManager
