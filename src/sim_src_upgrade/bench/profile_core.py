"""고속 코어 프로파일 — 남은 병목을 확인하고 다음 최적화를 고른다(추측 금지).

    python src/sim_src_upgrade/bench/profile_core.py --variant new_mask --n_eps 25
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import os
import pstats
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import json  # noqa: E402

from sim_src_upgrade._paths import REPO  # noqa: E402

EVAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="new_mask", choices=["old", "new", "new_obs", "new_mask"])
    ap.add_argument("--n_eps", type=int, default=25)
    ap.add_argument("--region", default="종로구_11110")
    ap.add_argument("--rule", default="START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
                      MCI_H_PAD="47", MCI_REWARD_MODE="woG")
    os.environ.setdefault("MCI_CARED_OBS", "1")

    from sim_src_upgrade.bench.bench_core import build, rollout

    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        mani = json.load(f)
    cfg = mani[args.region] if isinstance(mani[args.region], str) else mani[args.region]["path"]

    factory, policy = build(args.variant, cfg, args.rule, "new")

    devnull = open(os.devnull, "w")
    pr = cProfile.Profile()
    try:
        with contextlib.redirect_stdout(devnull):
            for s in range(3):                 # warm-up (캐시·지연 import 배제)
                rollout(factory, policy, s)
            pr.enable()
            steps = 0
            for s in range(args.n_eps):
                _rw, n = rollout(factory, policy, s)
                steps += n
            pr.disable()
    finally:
        devnull.close()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(args.top)
    sys.stderr.write(buf.getvalue())
    sys.stderr.write(f"variant={args.variant} eps={args.n_eps} steps={steps}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
