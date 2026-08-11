"""속도 실측 — 같은 프로세스에서 변형을 **번갈아** 돌리고 변형별 최소 CPU 시간을 쓴다.

공유 노드(loadavg 120+)라 벽시계는 실행마다 ±50% 흔들린다. 그래서
① `time.process_time()` (프로세스 CPU 시간), ② 변형 인터리브, ③ min-of-N 을 쓴다.
min 은 "경합이 가장 적었던 회차"라 노이즈에 가장 강한 추정량이다.

변형
----
* ``old``        : 현행 경로 (rl_src `env_factory` + `HospitalFeatureWrapper`)
* ``new``        : 고속 코어 + 동일 래퍼 (RL 학습·평가에 해당)
* ``new_obs``    : new + `fast_obs_patch` (관측 집계 등가 교체)
* ``new_mask``   : new_obs + `MaskOnlyFeatureWrapper` (규칙·트리 정책 전용)

각 변형의 지표 서명(보상합·스텝수)을 함께 찍는다 — **전부 같아야** 속도 비교가 의미 있다.

    python src/sim_src_upgrade/bench/bench_core.py --n_eps 25 --reps 7
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import numpy as np  # noqa: E402

from sim_src_upgrade._paths import REPO  # noqa: E402

EVAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
DEFAULT_RULE = "START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst"
VARIANTS = ("old", "new", "new_obs", "new_mask")


def rollout(factory, policy, seed: int):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    rw = 0.0
    n = 0
    while True:
        mask = np.asarray(env.action_masks(), dtype=bool)
        a = int(policy(obs, mask, env.unwrapped))
        obs, _r, terminated, truncated, info = env.step(a)
        rw += float(info.get("r_woG", 0.0))
        n += 1
        if terminated or truncated:
            break
    return rw, n


def build(variant: str, cfg: str, rule: str, rule_core: str):
    """(factory, policy) 생성. 변형별로 obs 패치 상태를 맞춘다."""
    from sim_src_upgrade import fast_obs_patch
    from sim_src_upgrade.env_factory_fast import make_feature_env_fast, make_feature_env_old
    from sim_src_upgrade.verify.policies import make_rule_policy

    if variant in ("new_obs", "new_mask"):
        fast_obs_patch.apply()
    else:
        fast_obs_patch.revert()

    if variant == "old":
        return make_feature_env_old(cfg), make_rule_policy(rule, core="old")
    mask_only = variant == "new_mask"
    return (make_feature_env_fast(cfg, mask_only=mask_only),
            make_rule_policy(rule, core=rule_core))


def main() -> int:
    ap = argparse.ArgumentParser(description="sim 코어 속도 실측")
    ap.add_argument("--n_eps", type=int, default=25)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--region", default="종로구_11110")
    ap.add_argument("--rule", default=DEFAULT_RULE)
    ap.add_argument("--rule_core", default="new", choices=["new", "old"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--gate", default="occ")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.update(MCI_CAP_GATE=args.gate, MCI_OBS_VARIANT="essential+load+valid",
                      MCI_H_PAD="47", MCI_REWARD_MODE="woG")
    os.environ.setdefault("MCI_CARED_OBS", "1")

    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        mani = json.load(f)
    cfg = mani[args.region] if isinstance(mani[args.region], str) else mani[args.region]["path"]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    built = {}          # 변형별 (factory, policy) — env 는 1회만 만들어 재사용
    best: dict[str, float] = {}
    sig: dict[str, tuple] = {}

    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull):
            for rep in range(args.reps):
                for v in variants:
                    if v not in built:
                        built[v] = build(v, cfg, args.rule, args.rule_core)
                    else:
                        # 변형 전환 시 obs 패치 상태를 다시 맞춘다
                        build.__globals__  # noqa: B018  (가독성용 no-op)
                        from sim_src_upgrade import fast_obs_patch
                        (fast_obs_patch.apply() if v in ("new_obs", "new_mask")
                         else fast_obs_patch.revert())
                    factory, policy = built[v]
                    tot, steps = 0.0, 0
                    t0 = time.process_time()
                    for s in range(args.n_eps):
                        rw, n = rollout(factory, policy, s)
                        tot += rw
                        steps += n
                    dt = time.process_time() - t0
                    best[v] = min(best.get(v, 1e18), dt)
                    sig.setdefault(v, (round(tot, 10), steps))
    finally:
        devnull.close()

    base = best.get("old", min(best.values()))
    rows = []
    print(f"\n지역={args.region}  n_eps={args.n_eps}  reps={args.reps}  "
          f"(min-of-{args.reps} CPU 시간)")
    print(f"{'변형':<10} {'CPU(s)':>9} {'ms/ep':>9} {'배속':>7}   지표서명")
    for v in variants:
        ms = best[v] / args.n_eps * 1000
        sp = base / best[v]
        print(f"{v:<10} {best[v]:9.3f} {ms:9.1f} {sp:6.2f}x   {sig[v]}")
        rows.append({"variant": v, "cpu_s": best[v], "ms_per_ep": ms,
                     "speedup_vs_old": sp, "signature": list(sig[v])})

    identical = len(set(sig.values())) == 1
    print(f"\n지표 동일성: {'OK — 전 변형 동일' if identical else 'MISMATCH ' + str(sig)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": rows, "identical": identical},
                      f, ensure_ascii=False, indent=2)
        print(f"[저장] {args.out}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
