"""G4 커버리지 — 통신축·병원수·자원·부하 조합마다 동치 게이트를 돌린다.

각 셀은 **별도 프로세스**로 띄운다. `MCI_OBS_VARIANT`·`MCI_H_PAD` 등은 래퍼 생성 시점에
읽히고 일부는 모듈 로드 시점에 굳으므로, 한 프로세스에서 값을 갈아끼우면 오염될 수 있다.
프로세스 격리가 유일하게 안전한 방법이다.

셀 구성
-------
* 통신축   : `MCI_CAP_GATE` occ/psent × `MCI_CARED_OBS` 1/0
* 병원수   : 고정 47(eval250) / 자연-H(natural, 34~51)
* 자원     : 기본 / AMB 0(=UAV-only) / UAV 0(=AMB-only)  ← action 레이아웃 96 vs 192
* 부하     : 기본 / 용량압박(capa_scale 0.3) / surge(incident 300)

    python src/sim_src_upgrade/verify/coverage_matrix.py --n_regions 2 --n_eps 3
    python src/sim_src_upgrade/verify/coverage_matrix.py --full          # v17 종료 후 전수
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
GATE = os.path.join(_HERE, "verify_equivalence.py")
PYTHON = sys.executable


def cells(full: bool) -> list[dict]:
    base = dict(manifest="eval250", gate="occ", cared="1", h_pad="47",
                amb_num="", uav_num="", incident_size="", capa_scale="",
                policies="full64:4,shin:2,shinalign:2,lb3,capT3")
    out = [dict(base, label="기준(고정47·occ)")]
    out.append(dict(base, label="통신단절(psent+cared off)", gate="psent", cared="0"))
    out.append(dict(base, label="psent만", gate="psent"))
    out.append(dict(base, label="cared off만", cared="0"))
    out.append(dict(base, label="자연-H(가변 병원수)", manifest="natural"))
    out.append(dict(base, label="AMB 0 (UAV-only)", amb_num="0", policies="full64:4,lb3"))
    out.append(dict(base, label="UAV 0 (AMB-only)", uav_num="0", policies="full64:4,shin:2,lb3,capT3"))
    out.append(dict(base, label="용량압박 capa0.3", capa_scale="0.3"))
    out.append(dict(base, label="surge incident 300", incident_size="300"))
    if full:
        out.append(dict(base, label="학습분포 train1000", manifest="train1000",
                        policies="full64:8,shin:4,shinalign:4,lb3,capT3"))
        out.append(dict(base, label="자연-H + 통신단절", manifest="natural",
                        gate="psent", cared="0"))
        out.append(dict(base, label="자연-H + UAV 0", manifest="natural", uav_num="0"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="G4 커버리지 매트릭스")
    ap.add_argument("--n_regions", type=int, default=2)
    ap.add_argument("--n_eps", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="셀·규모 확대(전수용)")
    ap.add_argument("--audit", action="store_true", default=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results/sim_upgrade/g4_coverage.json"))
    args = ap.parse_args()

    rows = []
    t_all = time.time()
    for i, c in enumerate(cells(args.full), 1):
        cmd = [PYTHON, GATE,
               "--manifest", c["manifest"], "--n_regions", str(args.n_regions),
               "--n_eps", str(args.n_eps), "--policies", c["policies"],
               "--gate", c["gate"], "--cared", c["cared"], "--h_pad", c["h_pad"],
               "--amb_num", c["amb_num"], "--uav_num", c["uav_num"],
               "--incident_size", c["incident_size"], "--capa_scale", c["capa_scale"],
               "--quiet"]
        if args.audit:
            cmd.append("--audit")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        dt = time.time() - t0
        tail = [ln for ln in p.stdout.splitlines() if "G1/G2" in ln]
        status = "PASS" if p.returncode == 0 else "FAIL"
        rows.append({"label": c["label"], "status": status, "elapsed_s": round(dt, 1),
                     "summary": tail[-1] if tail else (p.stdout or p.stderr)[-400:],
                     "cell": c})
        print(f"[{i:2d}/{len(cells(args.full))}] {status}  {c['label']:<26s} "
              f"{dt:5.1f}s  {tail[-1] if tail else ''}")
        if status == "FAIL":
            print((p.stdout or "")[-1500:])
            print((p.stderr or "")[-1500:])

    n_fail = sum(r["status"] == "FAIL" for r in rows)
    print(f"\n[G4] {'PASS' if n_fail == 0 else 'FAIL'} — {len(rows)}셀 중 실패 {n_fail} "
          f"(총 {time.time()-t_all:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_cells": len(rows), "n_fail": n_fail, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"[저장] {args.out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
