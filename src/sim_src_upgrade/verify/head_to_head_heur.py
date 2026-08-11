"""휴리스틱 Full64 동등비교 — 같은 좌표·같은 시드로 구/신 코어를 **각각 별도 프로세스**에서 돌린다.

`bench_core.py` 는 한 프로세스 안에서 인터리브해 잡음을 줄이는 대신 "실제로 실험이 얼마나
빨리 끝나는가"를 직접 답하지 못한다. 이 스크립트는 v10 드라이버의 휴리스틱 경로
(`make_feature_env` + `make_heuristic_policy` + rollout)를 **그대로** 쓰되 코어만 바꿔
한 번씩 통째로 돌리고, 결과 배열과 소요시간을 파일로 남긴다.

    # 구 코어
    python src/sim_src_upgrade/verify/head_to_head_heur.py --core old  --region 서울 --n_eps 30
    # 신 코어
    python src/sim_src_upgrade/verify/head_to_head_heur.py --core fast --region 서울 --n_eps 30
    # 대조
    python src/sim_src_upgrade/verify/head_to_head_heur.py --compare \\
        results/sim_upgrade/h2h/서울_old.npz results/sim_upgrade/h2h/서울_fast.npz

    # 이미 나와 있는 동결 결과와 대조 (v17 HEUR64 체크포인트의 앞 N 시드)
    python src/sim_src_upgrade/verify/head_to_head_heur.py --core fast \\
        --frozen results/scoreboard/v17/heur64_eta_aligned_full1000/work/heur/train1000/종로구_11110_p0.npz

지표는 v10 규약 그대로 `(reward, pdr, reward_woG, pdr_woG, time)` × 64규칙 × n_eps 이며
float32 로 저장한다(동결 NPZ 와 같은 dtype 이라 정확히 0 차이를 요구할 수 있다).
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

from sim_src_upgrade._paths import REPO, ensure_paths  # noqa: E402

METRIC_NAMES = ("reward", "pdr", "reward_woG", "pdr_woG", "time")
OUT_DIR = os.path.join(REPO, "results/sim_upgrade/h2h")

MANIFESTS = {
    "sido": os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"),
    "eval250": os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"),
    "train1000": os.path.join(REPO, "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"),
}


def all_rule_names() -> list[str]:
    out = []
    for priority in ("START", "ReSTART"):
        for hospital in ("RedOnly", "YellowNearest"):
            for red in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                for yellow in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                    out.append(f"{priority}, {hospital}, Red {red}, Yellow {yellow}")
    assert len(out) == 64
    return out


def resolve_config(region: str, manifest: str) -> tuple[str, str]:
    path = MANIFESTS[manifest]
    with open(path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    if region not in mani:
        raise SystemExit(f"{region!r} 없음 — 후보: {list(mani)[:10]} …")
    e = mani[region]
    return region, (e if isinstance(e, str) else e["path"])


def rollout(factory, policy, seed: int):
    """v10 `rollout` 과 동일 규약."""
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    reward = 0.0
    reward_wog = 0.0
    last_time = 0.0
    while True:
        mask = env.action_masks()
        action = policy(obs, mask, env.unwrapped)
        obs, r, terminated, truncated, info = env.step(action)
        reward += float(r)
        reward_wog += float(info.get("r_woG", 0.0))
        last_time = float(info.get("time", 0.0))
        if terminated or truncated:
            break
    prev = float(env.unwrapped.preventable)
    prev_wog = float(env.unwrapped.preventable_woG)
    pdr = 1.0 - reward / prev if prev > 0 else 0.0
    pdr_wog = 1.0 - reward_wog / prev_wog if prev_wog > 0 else 0.0
    return reward, pdr, reward_wog, pdr_wog, last_time


def run(core: str, cfg: str, n_eps: int, seed0: int, mask_only: bool):
    ensure_paths()
    rules = all_rule_names()
    values = np.full((len(rules), n_eps, len(METRIC_NAMES)), np.nan, dtype=np.float32)

    if core == "fast":
        from sim_src_upgrade import fast_obs_patch
        from sim_src_upgrade.env_factory_fast import make_feature_env_fast
        from sim_src_upgrade.verify.policies import make_rule_policy
        fast_obs_patch.apply()
        factory = make_feature_env_fast(cfg, mask_only=mask_only)
        make_policy = lambda r: make_rule_policy(r, core="new")  # noqa: E731
    else:
        from viper_distill import make_feature_env
        from distill_policy import make_heuristic_policy
        factory = make_feature_env(cfg, None)
        make_policy = make_heuristic_policy

    devnull = open(os.devnull, "w")
    t_wall0, t_cpu0 = time.perf_counter(), time.process_time()
    try:
        with contextlib.redirect_stdout(devnull):
            for ri, rname in enumerate(rules):
                pol = make_policy(rname)
                for ei, s in enumerate(range(seed0, seed0 + n_eps)):
                    values[ri, ei] = rollout(factory, pol, s)
    finally:
        devnull.close()
    return values, time.perf_counter() - t_wall0, time.process_time() - t_cpu0


def compare(a_path: str, b_path: str, label_a="A", label_b="B") -> int:
    A = np.load(a_path, allow_pickle=False)
    B = np.load(b_path, allow_pickle=False)
    va, vb = A["values"], B["values"]
    n = min(va.shape[1], vb.shape[1])
    va, vb = va[:, :n], vb[:, :n]
    same_bits = np.array_equal(va.view(np.uint32), vb.view(np.uint32))
    exact = np.array_equal(va, vb, equal_nan=True)
    dmax = float(np.nanmax(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
    print(f"\n[결과 대조] {label_a}  vs  {label_b}")
    print(f"  shape={va.shape} (64규칙 × {n}에피 × 5지표)  비교 원소 {va.size:,}개")
    print(f"  비트동일={same_bits}  값동일={exact}  최대차이={dmax:g}")
    for i, m in enumerate(METRIC_NAMES):
        print(f"    {m:<11s} {label_a} 평균={np.nanmean(va[:,:,i]):.10f}  "
              f"{label_b} 평균={np.nanmean(vb[:,:,i]):.10f}")
    ta = float(A["wall_s"]) if "wall_s" in A else float("nan")
    tb = float(B["wall_s"]) if "wall_s" in B else float("nan")
    if ta == ta and tb == tb:
        print(f"  소요(wall) {label_a}={ta:.1f}s  {label_b}={tb:.1f}s  → 배속 {ta/tb:.2f}x")
    print(f"\n[판정] {'PASS — 결과 동일' if exact else 'FAIL — 결과 불일치'}")
    return 0 if exact else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="휴리스틱 Full64 구/신 코어 동등비교")
    ap.add_argument("--core", choices=["old", "fast"], help="실행 모드")
    ap.add_argument("--region", default="서울")
    ap.add_argument("--manifest", default="sido", choices=list(MANIFESTS))
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--mask_only", action="store_true",
                    help="fast 전용 — 특징 obs 생성 생략(규칙정책은 obs 미사용)")
    ap.add_argument("--gate", default="occ", choices=["occ", "psent"])
    ap.add_argument("--out", default="")
    ap.add_argument("--frozen", default="", help="이미 나와 있는 동결 NPZ 와 대조(앞 n_eps 시드)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="두 결과 NPZ 대조만 수행")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1],
                       os.path.basename(args.compare[0]), os.path.basename(args.compare[1]))

    if not args.core:
        raise SystemExit("--core old|fast 또는 --compare A B 중 하나가 필요하다")

    os.environ.update(MCI_CAP_GATE=args.gate, MCI_OBS_VARIANT="essential+load+valid",
                      MCI_H_PAD="47", MCI_REWARD_MODE="woG")
    os.environ.setdefault("MCI_CARED_OBS", "1")

    from sim_src_upgrade import origin_sync
    origin_sync.check()

    region, cfg = resolve_config(args.region, args.manifest)
    print(f"[{args.core}] 좌표={region}  config={os.path.basename(cfg)}  "
          f"64규칙 × {args.n_eps}에피(seed {args.seed0}..{args.seed0+args.n_eps-1})", flush=True)

    values, wall, cpu = run(args.core, cfg, args.n_eps, args.seed0,
                            args.mask_only and args.core == "fast")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"{region}_{args.core}.npz")
    np.savez(out, values=values, wall_s=np.float64(wall), cpu_s=np.float64(cpu),
             rules=np.asarray(all_rule_names()),
             seeds=np.arange(args.seed0, args.seed0 + args.n_eps))
    print(f"[{args.core}] 완료 wall={wall:.1f}s cpu={cpu:.1f}s  "
          f"({wall/(64*args.n_eps)*1000:.1f} ms/ep)  → {out}", flush=True)
    print(f"[{args.core}] 평균 PDR_woG = {np.nanmean(values[:,:,3]):.10f}  "
          f"최저(=Best-of-64) = {np.nanmin(values[:,:,3].mean(axis=1)):.10f}")

    if args.frozen:
        F = np.load(args.frozen, allow_pickle=False)
        fv = F["values"][:, :args.n_eps]
        same = np.array_equal(values, fv, equal_nan=True)
        dmax = float(np.nanmax(np.abs(values.astype(np.float64) - fv.astype(np.float64))))
        print(f"\n[동결 대조] {os.path.relpath(args.frozen, REPO)}")
        print(f"  규칙명 일치={F['rule_names'].tolist() == all_rule_names() if 'rule_names' in F else 'n/a'}")
        print(f"  값동일={same}  최대차이={dmax:g}")
        print(f"[판정] {'PASS — 기존 결과와 동일' if same else 'FAIL'}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
