"""G0 드리프트 게이트 — 사본이 어느 시점의 `src/sim_src` 에서 파생됐는지 고정한다.

사본은 원본과 로직이 같아야만 의미가 있다. 원본이 바뀌면(다른 세션·다른 머신에서)
사본은 조용히 낡는다. 그래서 파생 시점의 sha256 을 박아두고, 모든 검증 진입점이
먼저 `check()` 를 호출한다.

사용
----
    python src/sim_src_upgrade/origin_sync.py --write     # 현재 sim_src 로 기준 갱신
    python src/sim_src_upgrade/origin_sync.py --check     # 드리프트 검사(불일치 시 exit 1)
    python src/sim_src_upgrade/origin_sync.py --diff EventManager   # 원본 대비 사본 diff

`--write` 는 "사본을 원본과 정합시켰다"는 선언이다. 원본이 바뀌었다면 **먼저 사본에
그 변경을 반영**한 뒤 `--write` 해야 한다. 그냥 `--write` 만 하면 드리프트를 덮는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SIM_SRC = os.path.join(REPO, "src", "sim_src")
CORE = os.path.join(HERE, "core")
MANIFEST = os.path.join(CORE, "_origin_hashes.json")

MODULES = (
    "EntityManager",
    "EventManager",
    "ScenarioManager",
    "MCIEnvironment_gymnasium",
    "RuleManager",
    "ShinHeuristics",
    "ShinAlignedHeuristics",
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_hashes() -> dict[str, str]:
    return {m: sha256_file(os.path.join(SIM_SRC, m + ".py")) for m in MODULES}


def write() -> dict:
    payload = {
        "note": "이 사본이 파생된 src/sim_src 파일들의 sha256. origin_sync.py --check 가 대조한다.",
        "modules": origin_hashes(),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return payload


def check(strict: bool = True) -> bool:
    """원본 sha256 이 파생 시점과 같은지 확인. 다르면 (strict 면) 예외."""
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(f"G0 매니페스트 없음: {MANIFEST} (origin_sync.py --write 먼저)")
    with open(MANIFEST, "r", encoding="utf-8") as f:
        recorded = json.load(f)["modules"]
    current = origin_hashes()
    drift = [m for m in MODULES if recorded.get(m) != current[m]]
    if drift and strict:
        raise RuntimeError(
            "G0 드리프트: src/sim_src 가 사본 파생 이후 변경됨 → " + ", ".join(drift) +
            "\n  사본에 변경을 반영한 뒤 origin_sync.py --write 로 기준을 갱신하라."
        )
    return not drift


def diff(module: str) -> int:
    """원본 대비 사본 diff — 상대 import 전환 줄만 나와야 정상."""
    a = os.path.join(SIM_SRC, module + ".py")
    b = os.path.join(CORE, module + ".py")
    return subprocess.call(["diff", "-u", a, b])


def main() -> int:
    ap = argparse.ArgumentParser(description="G0 사본 드리프트 게이트")
    ap.add_argument("--write", action="store_true", help="현재 sim_src 해시로 기준 갱신")
    ap.add_argument("--check", action="store_true", help="드리프트 검사")
    ap.add_argument("--diff", metavar="MODULE", help="원본 대비 사본 diff")
    args = ap.parse_args()

    if args.diff:
        return diff(args.diff)
    if args.write:
        payload = write()
        print(f"[G0] 기준 기록: {MANIFEST}")
        for m, h in sorted(payload["modules"].items()):
            print(f"  {m:28s} {h[:16]}")
        return 0
    if args.check or True:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            print(f"[G0] FAIL\n{exc}")
            return 1
        print("[G0] PASS — 사본이 파생된 src/sim_src 와 원본 해시 일치")
        return 0


if __name__ == "__main__":
    sys.exit(main())
