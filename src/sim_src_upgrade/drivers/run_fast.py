"""기존 드라이버를 **한 줄도 안 고치고** 고속 코어로 실행하는 범용 런처.

원리
----
`v16_baseline_alignment`·`v10_full_baselines`·`shin_full_baselines`·`paired_eval_ladder`
같은 드라이버는 전부

    from viper_distill import make_feature_env      # env 생성
    from distill_policy import parse_rule           # 규칙 생성

를 거쳐 sim 을 돌린다. 이 런처는 대상 모듈을 import 한 뒤 그 **진입점만 고속판으로 재바인딩**
하고 `main()` 을 부른다. 드라이버의 체크포인트·CSV·집계 로직은 원본 그대로 재사용되므로
출력 포맷·재개 규약이 100% 동일하다(코드 중복 0).

Pool 워커에도 적용되는 이유: 이 저장소의 드라이버는 `multiprocessing.Pool` 기본 시작방식
(리눅스=fork)을 쓰므로 자식이 부모의 패치된 모듈 상태를 그대로 물려받는다.
⚠️ `spawn`/`forkserver` 를 쓰는 경로(SB3 SubprocVecEnv 학습 등)에는 이 런처가 안 먹는다.

안전장치
--------
* **사전점검(pre-flight)**: 실제 작업 전에 한 좌표를 구·신 코어로 각각 돌려 지표가
  완전히 같은지 확인한다. 다르면 즉시 중단한다(`--skip_preflight` 로만 생략 가능).
* 드라이버가 검사하는 `src/sim_src` 소스 해시는 그대로다 — 진행 중이던 원본 실행의
  체크포인트를 이어받아 계속 돌릴 수 있다. 결과가 비트동일이므로 섞여도 무방하다.

사용
----
    python src/sim_src_upgrade/drivers/run_fast.py --target v16_baseline_alignment -- \
        --out_dir results/scoreboard/v17/... --n_eps 1000 --workers 56 --phase run

    python src/sim_src_upgrade/drivers/run_fast.py --target v10_full_baselines --mask_only -- \
        --workers 104 --n_eps 1000
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

from sim_src_upgrade import fast_obs_patch, origin_sync  # noqa: E402
from sim_src_upgrade._paths import REPO, ensure_paths  # noqa: E402

PATCH_LOG: list[str] = []

# `--mask_only` 는 특징 obs 벡터를 만들지 않는다. 관측을 **읽지 않는** 드라이버에서만 안전하다.
# 신경망 정책·평탄 obs 트리·obs 데이터셋 수집이 섞이면 조용히 망가진다(sklearn 은 NaN 을
# 결측치로 받아들여 예외도 안 낸다) → 안전 기본값을 **허용목록**으로 둔다.
MASK_ONLY_SAFE = {
    "v10_full_baselines",        # HEUR64 + T4 (규칙만)
    "v16_baseline_alignment",    # LB3 + Shin 정합변형 (규칙만)
    "shin_full_baselines",       # Shin 문헌 16종 (규칙만)
    "lb_validate17",
    "lb_validate_sigungu",
    "fit_v10_heuristic_rules",
}


def _rebind_everywhere(attr: str, new_obj, only_modules=None) -> int:
    """이미 import 된 모든 모듈에서 `attr` 이름이 가리키는 객체를 교체한다.

    `from RuleManager import Universal_Rule` 처럼 이름을 복사해 간 모듈이 많아
    원본 모듈 속성만 바꿔서는 안 먹는다.
    """
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if only_modules is not None and name not in only_modules:
            continue
        if getattr(mod, "__file__", None) is None:
            continue
        f = mod.__file__ or ""
        if "/src/rl_src/" not in f and "/src/sim_src/" not in f:
            continue
        if getattr(mod, attr, None) is not None:
            setattr(mod, attr, new_obj)
            n += 1
    return n


def enable_child_propagation(mask_only: bool = False) -> str:
    """spawn/forkserver 자식에도 패치가 적용되도록 `PYTHONPATH` + 환경변수를 심는다.

    fork 자식은 부모 메모리를 물려받아 패치가 자동 상속되지만, SB3 `SubprocVecEnv` 는
    기본이 forkserver 라 인터프리터를 새로 띄운다. 그 경우 `fastcore_boot/sitecustomize.py`
    가 부팅 시 패치를 다시 건다. 환경변수는 자식에게 상속되므로 여기서 세팅만 하면 된다.
    """
    boot = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "fastcore_boot")
    boot = os.path.abspath(boot)
    os.environ["MCI_FASTCORE"] = "1"
    os.environ["MCI_FASTCORE_MASK_ONLY"] = "1" if mask_only else "0"
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if boot not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([boot] + parts)
    return boot


def apply_fast_core(mask_only: bool = False, quiet: bool = False) -> None:
    """rl_src 진입점을 고속 코어로 재바인딩(원본 파일 무수정)."""
    ensure_paths()
    origin_sync.check()

    # 1) 관측 집계·이송중 카운트 (rl_src 래퍼 내부 핫스팟)
    fast_obs_patch.apply()
    if not quiet:
        PATCH_LOG.append("fast_obs_patch (patient_agg / fleet_agg / in_flight)")

    # 2) env 팩토리 — sim 코어 교체의 본체
    import viper_distill as VD
    import env_factory as EF
    from sim_src_upgrade.env_factory_fast import make_base_env_fast, make_feature_env_fast

    orig_make_feature_env = VD.make_feature_env

    def fast_make_feature_env(config_path, norm=None):
        # 매니페스트(.json) 경로는 MultiRegionEnv 라 원본 경로 유지
        if str(config_path).endswith(".json"):
            return orig_make_feature_env(config_path, norm)
        return make_feature_env_fast(config_path, norm=norm, mask_only=mask_only)

    VD.make_feature_env = fast_make_feature_env
    _rebind_everywhere("make_feature_env", fast_make_feature_env)
    EF.make_base_env = make_base_env_fast
    _rebind_everywhere("make_base_env", make_base_env_fast)
    if not quiet:
        PATCH_LOG.append(f"env factory → 고속 코어 (mask_only={mask_only})")

    # 3) 규칙 클래스 — 신 코어판(이송중 증분 카운터 재사용)
    from sim_src_upgrade.core import RuleManager as NRM
    from sim_src_upgrade.core import ShinAlignedHeuristics as NSA
    from sim_src_upgrade.core import ShinHeuristics as NSH

    n1 = _rebind_everywhere("Universal_Rule", NRM.Universal_Rule)
    n2 = _rebind_everywhere("ShinHeuristicRule", NSH.ShinHeuristicRule)
    n3 = _rebind_everywhere("ShinHospitalAlignedRule", NSA.ShinHospitalAlignedRule)
    import ShinAlignedHeuristics as OSA
    import ShinHeuristics as OSH
    import RuleManager as ORM
    OSA.ShinHospitalAlignedRule = NSA.ShinHospitalAlignedRule
    OSH.ShinHeuristicRule = NSH.ShinHeuristicRule
    ORM.Universal_Rule = NRM.Universal_Rule
    if not quiet:
        PATCH_LOG.append(f"규칙 클래스 재바인딩 (Universal_Rule×{n1+1}, Shin×{n2+1}, ShinAligned×{n3+1})")


def preflight(n_eps: int = 3, region: str | None = None) -> None:
    """실제 작업 전에 구·신 코어 지표가 **완전히 같은지** 확인한다. 다르면 중단."""
    import json

    import numpy as np

    from sim_src_upgrade.env_factory_fast import make_feature_env_fast, make_feature_env_old
    from sim_src_upgrade.verify.policies import make_rule_policy

    mani_path = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
    with open(mani_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    key = region or next(iter(mani))
    cfg = mani[key] if isinstance(mani[key], str) else mani[key]["path"]

    rules = ["START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst",
             "ReSTART, YellowNearest, Red OnlyUAV, Yellow Both_UAVFirst"]

    def run(side):
        fac = (make_feature_env_fast(cfg) if side == "new" else make_feature_env_old(cfg))
        out = []
        for r in rules:
            pol = make_rule_policy(r, core=("new" if side == "new" else "old"))
            for s in range(n_eps):
                env = fac(seed=s)
                obs, _ = env.reset(seed=s)
                w = 0.0
                n = 0
                while True:
                    m = np.asarray(env.action_masks(), dtype=bool)
                    a = int(pol(obs, m, env.unwrapped))
                    obs, _rr, term, trunc, info = env.step(a)
                    w += float(info.get("r_woG", 0.0))
                    n += 1
                    if term or trunc:
                        break
                out.append((round(w, 12), n))
        return out

    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull):
            a = run("old")
            b = run("new")
    finally:
        devnull.close()

    if a != b:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        raise RuntimeError(f"[사전점검 실패] 구·신 코어 지표 불일치 {len(diff)}건: {diff[:5]}")
    print(f"[사전점검] PASS — {key} / 규칙 {len(rules)} × 시드 {n_eps} 지표 완전일치", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="기존 드라이버를 고속 코어로 실행", add_help=True)
    ap.add_argument("--target", required=True,
                    help="rl_src 모듈명 (예: v16_baseline_alignment, v10_full_baselines)")
    ap.add_argument("--mask_only", action="store_true",
                    help="특징 obs 생성 생략 — 관측을 읽지 않는 드라이버(MASK_ONLY_SAFE)에서만")
    ap.add_argument("--force_mask_only", action="store_true",
                    help="허용목록 밖 대상에도 --mask_only 를 강행(직접 확인했을 때만)")
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--preflight_eps", type=int, default=3)
    ap.add_argument("driver_args", nargs=argparse.REMAINDER,
                    help="'--' 뒤에 대상 드라이버 인자를 그대로 전달")
    args = ap.parse_args()

    driver_args = args.driver_args
    if driver_args and driver_args[0] == "--":
        driver_args = driver_args[1:]

    if args.mask_only and args.target not in MASK_ONLY_SAFE and not args.force_mask_only:
        raise SystemExit(
            f"[거부] --mask_only 는 관측을 읽지 않는 드라이버에서만 안전하다.\n"
            f"  {args.target!r} 은 허용목록에 없다: {sorted(MASK_ONLY_SAFE)}\n"
            f"  신경망 정책·평탄 obs 트리·obs 데이터셋 수집이 섞여 있으면 결과가 조용히 망가진다\n"
            f"  (sklearn 은 NaN 을 결측치로 받아 예외조차 내지 않는다).\n"
            f"  → --mask_only 를 빼고 실행하라(그래도 약 2.2배 빠르다).\n"
            f"     정말 규칙 전용임을 확인했다면 --force_mask_only 로 우회할 수 있다.")

    ensure_paths()
    import importlib

    # 자식 전파 설정은 대상 모듈 import 보다 **먼저** 해야 한다 —
    # 대상이 import 시점에 하위 프로세스를 띄우는 경우까지 덮기 위함.
    boot = enable_child_propagation(mask_only=args.mask_only)

    target = importlib.import_module(args.target)   # 패치 전에 먼저 로드(참조 수집 목적)
    apply_fast_core(mask_only=args.mask_only)
    for line in PATCH_LOG:
        print(f"[고속코어] {line}", flush=True)
    print(f"[고속코어] 자식 전파: MCI_FASTCORE=1, PYTHONPATH+={boot}", flush=True)

    if not args.skip_preflight:
        preflight(n_eps=args.preflight_eps)

    if not hasattr(target, "main"):
        raise SystemExit(f"{args.target} 에 main() 이 없다 — 런처로 실행할 수 없다")

    sys.argv = [target.__file__] + list(driver_args)
    print(f"[고속코어] 실행: {args.target} {' '.join(driver_args)}", flush=True)
    rc = target.main()
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
