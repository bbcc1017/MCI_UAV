"""G5 RL 경로 게이트 — 특징 obs 벡터가 **비트동일**한지, 모델 액션열이 같은지 본다.

규칙정책 게이트(G1/G2)는 obs 를 안 읽으므로 관측 경로의 등가성을 증명하지 못한다.
여기서 두 가지를 확인한다.

1. **obs 비트동일** : 같은 시드·같은 규칙정책으로 구·신 경로를 각각 돌리며 매 스텝의
   402차원 obs 를 `tobytes()` 해시로 비교. `fast_obs_patch` 가 rl_src 래퍼를 **전역**으로
   바꾸므로 한 프로세스에서 동시에 못 띄운다 → 구(패치 OFF) 수집 → 신(패치 ON) 수집 →
   대조의 2단 방식을 쓴다. 규칙정책은 obs 를 안 보므로 두 실행의 궤적은 같다.
2. **모델 액션열 동일** (`--model` 지정 시) : 동결 PPO 모델을 얹어 액션·지표까지 대조.
   obs 가 비트동일하면 결정적 예측도 같아야 한다 — 실제 학습·평가 경로의 종단 확인.

    python src/sim_src_upgrade/verify/verify_rl_obs.py --n_regions 3 --n_eps 5
    python src/sim_src_upgrade/verify/verify_rl_obs.py --model results/rl/redesign/v10_random4_1000_pointer_s0
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import numpy as np  # noqa: E402

from sim_src_upgrade import origin_sync  # noqa: E402
from sim_src_upgrade._paths import REPO  # noqa: E402

EVAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
RULE = "START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst"


def _model_policy(model_dir: str):
    """동결 PPO 모델 → (policy_fn, norm). paired_eval_ladder 의 로딩 계약을 그대로 따른다."""
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (deepsets 역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    import pad_vecnorm  # noqa: F401 (PadAwareVecNormalize 언피클)
    from evaluate import ppo_policy
    from viper_distill import load_vecnorm

    zip_path = os.path.join(model_dir, "model.zip")
    if not os.path.exists(zip_path):
        cands = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if not cands:
            raise FileNotFoundError(f"모델 zip 을 못 찾음: {model_dir}")
        zip_path = os.path.join(model_dir, sorted(cands)[0])
    vn_path = os.path.join(model_dir, "vecnormalize.pkl")
    norm = load_vecnorm(vn_path) if os.path.exists(vn_path) else None
    return ppo_policy(MaskablePPO.load(zip_path, device="cpu")), norm


def collect(side: str, configs: dict, seeds, model_dir: str | None):
    """한쪽 코어로 (obs 해시열, 액션열, 지표) 수집. side='old'|'new'."""
    from sim_src_upgrade import fast_obs_patch
    from sim_src_upgrade.env_factory_fast import make_feature_env_fast, make_feature_env_old
    from sim_src_upgrade.verify.policies import make_rule_policy

    if side == "new":
        fast_obs_patch.apply()
    else:
        fast_obs_patch.revert()

    norm = None
    if model_dir:
        policy_proto, norm = _model_policy(model_dir)

    out = {}
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull):
            for key, cfg in configs.items():
                factory = (make_feature_env_fast(cfg, norm=norm) if side == "new"
                           else make_feature_env_old(cfg, norm=norm))
                policy = policy_proto if model_dir else make_rule_policy(
                    RULE, core=("new" if side == "new" else "old"))
                for seed in seeds:
                    env = factory(seed=seed)
                    obs, _ = env.reset(seed=seed)
                    h = hashlib.sha256()
                    acts: list[int] = []
                    rw = 0.0
                    n = 0
                    while True:
                        a_obs = np.ascontiguousarray(np.asarray(obs, dtype=np.float32))
                        h.update(a_obs.tobytes())
                        mask = np.asarray(env.action_masks(), dtype=bool)
                        a = int(policy(obs, mask, env.unwrapped))
                        acts.append(a)
                        obs, _r, term, trunc, info = env.step(a)
                        rw += float(info.get("r_woG", 0.0))
                        n += 1
                        if term or trunc:
                            break
                    a_obs = np.ascontiguousarray(np.asarray(obs, dtype=np.float32))
                    h.update(a_obs.tobytes())
                    out[(key, seed)] = (h.hexdigest(),
                                        hashlib.sha256(str(acts).encode()).hexdigest(),
                                        round(rw, 12), n)
    finally:
        devnull.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="G5 RL 관측·액션 동치 게이트")
    ap.add_argument("--n_regions", type=int, default=3)
    ap.add_argument("--n_eps", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default="", help="동결 PPO 모델 디렉터리(미지정 시 규칙정책으로 obs 만 대조)")
    ap.add_argument("--gate", default="occ", choices=["occ", "psent"])
    ap.add_argument("--cared", default="1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.update(MCI_CAP_GATE=args.gate, MCI_CARED_OBS=args.cared,
                      MCI_OBS_VARIANT="essential+load+valid", MCI_H_PAD="47",
                      MCI_REWARD_MODE="woG")

    try:
        origin_sync.check()
        print("[G0] PASS")
    except Exception as exc:  # noqa: BLE001
        print(f"[G0] FAIL — {exc}")
        return 1

    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        mani = json.load(f)
    keys = list(mani)
    step = max(1, len(keys) // args.n_regions)
    keys = keys[::step][:args.n_regions]
    configs = {k: (mani[k] if isinstance(mani[k], str) else mani[k]["path"]) for k in keys}
    seeds = list(range(args.seed0, args.seed0 + args.n_eps))

    model_dir = os.path.join(REPO, args.model) if args.model and not os.path.isabs(args.model) else args.model
    print(f"[규모] 좌표 {len(configs)} × 시드 {len(seeds)}  정책="
          f"{'모델 ' + os.path.basename(model_dir) if model_dir else '규칙'}")

    old = collect("old", configs, seeds, model_dir or None)
    new = collect("new", configs, seeds, model_dir or None)

    fails = []
    for k in old:
        if old[k] != new[k]:
            fails.append({"key": str(k), "old": old[k], "new": new[k]})
    ok = not fails
    print(f"[G5] {'PASS' if ok else 'FAIL'} — {len(old)}쌍 중 실패 {len(fails)} "
          f"(obs 해시·액션열·woG·스텝수 전부 대조)")
    for f in fails[:5]:
        print(f"  · {f['key']}\n     old={f['old']}\n     new={f['new']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "n_pairs": len(old), "fails": fails[:20]},
                      f, ensure_ascii=False, indent=2)
        print(f"[저장] {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
