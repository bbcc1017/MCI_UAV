"""spawn/forkserver 자식에도 고속 코어를 적용하기 위한 인터프리터 부팅 훅.

배경
----
`drivers/run_fast.py` 의 몽키패치는 `multiprocessing` **fork** 자식에게만 상속된다.
그런데 SB3 `SubprocVecEnv` 는 기본 시작방식이 **forkserver**(리눅스)라 학습 워커는
부모의 패치를 물려받지 못한다. 자식은 인터프리터를 새로 띄우므로, 그 시점에 끼어들 방법은
`sitecustomize` 뿐이다(파이썬이 site 초기화 때 자동 import 한다).

활성 조건
---------
`MCI_FASTCORE=1` 일 때만 동작한다. 런처가 `PYTHONPATH` 에 이 폴더를 넣고 환경변수를 세팅하며,
그 환경변수는 spawn/forkserver 자식에게 상속된다.

실패 시 정책
------------
패치에 실패하면 **원본 코어로 조용히 되돌리고 경고만** 남긴다. 고속 코어와 원본은 결과가
비트동일이므로, 일부 워커가 원본으로 돌아도 결과는 여전히 정확하다(느려질 뿐).
잘못된 상태로 계속 도는 것보다 이쪽이 안전하다.
"""
import os
import sys

if os.environ.get("MCI_FASTCORE") == "1":
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _src = os.path.abspath(os.path.join(_here, os.pardir, os.pardir))  # → <repo>/src
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from sim_src_upgrade.drivers.run_fast import apply_fast_core

        apply_fast_core(mask_only=os.environ.get("MCI_FASTCORE_MASK_ONLY") == "1",
                        quiet=True)
    except Exception as _exc:  # noqa: BLE001
        sys.stderr.write(
            f"[fastcore] 자식 프로세스 패치 실패 — 원본 코어로 진행한다(결과는 동일, 속도만 손해): {_exc!r}\n")
        try:
            from sim_src_upgrade import fast_obs_patch
            fast_obs_patch.revert()
        except Exception:  # noqa: BLE001
            pass
