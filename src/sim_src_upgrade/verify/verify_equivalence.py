"""G0/G1/G2 동치 게이트 — 구 코어 vs 신 코어를 **같은 프로세스**에서 짝지어 돌린다.

판정
----
* **G0** 드리프트 : 사본이 파생된 `src/sim_src` 해시가 그대로인가 (`origin_sync.check`)
* **G1** 궤적 동일 : 실행 이벤트열 `(time, ev_name, entity_idx)` + 액션열 SHA256 완전일치
* **G2** 지표 동일 : per-episode `(reward, r_woG, pdr, pdr_woG, time, n_steps)` float64 `==`

G1 이 G2 보다 강하다 — 궤적이 갈리는 순간을 잡으므로 "우연히 총합만 같은" 경우를 배제한다.
둘 다 통과해야 그 최적화를 채택한다.

사용
----
    python src/sim_src_upgrade/verify/verify_equivalence.py \
        --n_regions 5 --n_eps 20 --policies full64:4,shin:2,lb3,capT3 \
        --gate occ --rule_core new

⚠️ 계측 오버헤드가 크다(이벤트마다 문자열화). 속도 측정은 `bench/bench_core.py` 로.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import numpy as np  # noqa: E402

from sim_src_upgrade import origin_sync  # noqa: E402
from sim_src_upgrade._paths import REPO  # noqa: E402

EVAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
TRAIN_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json")
NATURAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_natural_osrm_manifest.json")

METRIC_NAMES = ("reward", "r_woG", "pdr", "pdr_woG", "time", "n_steps")


# ----------------------------------------------------------------- 환경 설정
def apply_env(gate: str, cared: str, obs_variant: str, h_pad: str, extra: dict | None = None) -> dict:
    """sim/obs 동작을 결정하는 환경변수를 **모든 import 이전에** 확정한다.

    extra 는 자원·부하 축 노브(`MCI_AMB_NUM`/`MCI_UAV_NUM`/`MCI_INCIDENT_SIZE`/`MCI_CAPA_SCALE`).
    빈 값이면 해당 키를 제거해 기본 시나리오값을 쓴다.
    """
    env = {
        "MCI_CAP_GATE": gate,
        "MCI_CARED_OBS": cared,
        "MCI_OBS_VARIANT": obs_variant,
        "MCI_REWARD_MODE": "woG",
    }
    if h_pad:
        env["MCI_H_PAD"] = h_pad
    else:
        os.environ.pop("MCI_H_PAD", None)
    for k, v in (extra or {}).items():
        if str(v).strip():
            env[k] = str(v)
        else:
            os.environ.pop(k, None)
    os.environ.update(env)
    return env


# ----------------------------------------------------------------- 정책 목록
def all_rule_names() -> list[str]:
    out = []
    for priority in ("START", "ReSTART"):
        for hospital in ("RedOnly", "YellowNearest"):
            for red in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                for yellow in ("OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"):
                    out.append(f"{priority}, {hospital}, Red {red}, Yellow {yellow}")
    return out


def shin_names(core: str) -> list[str]:
    from sim_src_upgrade.verify.policies import _rule_modules

    _rm, sh, _sa = _rule_modules(core)
    return [f"Shin {m}, Mode {md}" for m in sh.SHIN_METHODS for md in sh.SHIN_MODE_RULES]


def shin_aligned_specs(core: str) -> list[tuple]:
    from sim_src_upgrade.verify.policies import _rule_modules

    _rm, sh, sa = _rule_modules(core)
    return [(m, h, md)
            for m in sa.SHIN_ALIGNED_METHODS
            for h in sa.SHIN_ALIGNED_HOSPITAL_RULES
            for md in sh.SHIN_MODE_RULES]


def build_policy_specs(spec_str: str, rule_core: str) -> list[tuple[str, str, dict]]:
    """"full64:4,shin:2,shinalign:2,lb3,capT3" → [(라벨, 종류, 인자), ...]"""
    out: list[tuple[str, str, dict]] = []
    for token in [t.strip() for t in spec_str.split(",") if t.strip()]:
        kind, _, n_str = token.partition(":")
        n = int(n_str) if n_str else None
        if kind == "full64":
            names = all_rule_names()
            names = names[:: max(1, len(names) // n)][:n] if n else names
            out += [(f"RULE[{nm}]", "rule", {"rule_name": nm}) for nm in names]
        elif kind == "shin":
            names = shin_names(rule_core)
            names = names[:: max(1, len(names) // n)][:n] if n else names
            out += [(f"SHIN[{nm}]", "rule", {"rule_name": nm}) for nm in names]
        elif kind == "shinalign":
            specs = shin_aligned_specs(rule_core)
            specs = specs[:: max(1, len(specs) // n)][:n] if n else specs
            out += [(f"SHINALIGN{s}", "shin_aligned", {"spec": s}) for s in specs]
        elif kind == "lb3":
            out.append(("LB3-AGN", "lb3", {}))
        elif kind == "capT3":
            out.append(("LB3-CAP[START,RedOnly,R AMBFirst,Y AMBFirst]", "cap",
                        {"rule_name": "START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst",
                         "T": 3}))
        else:
            raise ValueError(f"알 수 없는 정책 종류: {kind}")
    return out


def make_policy(kind: str, kwargs: dict, rule_core: str):
    """kind 별 정책 인스턴스. lb3/cap 은 rl_src 원본을 그대로 쓴다(수정 금지 대상)."""
    from sim_src_upgrade.verify.policies import make_rule_policy, make_shin_aligned_policy

    if kind == "rule":
        return make_rule_policy(kwargs["rule_name"], core=rule_core)
    if kind == "shin_aligned":
        return make_shin_aligned_policy(kwargs["spec"], core=rule_core)
    if kind == "lb3":
        from lb3_policy import make_agnostic_lb_policy
        return make_agnostic_lb_policy(T=3)
    if kind == "cap":
        from loadbalance_heuristic import make_cap_policy
        return make_cap_policy(kwargs["rule_name"], kwargs["T"])
    raise ValueError(kind)


# ----------------------------------------------------------------- 자가진단
def inject_selftest(kind: str) -> None:
    """게이트가 **차이를 실제로 잡는지** 증명하기 위해 신 코어에 의도적 결함을 주입한다.

    항상 PASS 만 내는 게이트는 무가치하다. 최적화를 채택하기 전에 최소 1회는
    `--selftest traj` / `--selftest metric` 이 FAIL 을 내는지 확인해야 한다.

    * ``traj``   : GB 배차의 tier2/tier3 우선순위를 뒤집는다 → 궤적이 갈림 (G1 FAIL)
    * ``metric`` : 생존확률에 1e-12 를 더한다 → 궤적은 같고 지표만 갈림 (G2 FAIL, G1 PASS)
    """
    if kind == "none":
        return
    from sim_src_upgrade.core import EventManager as new_evm
    from sim_src_upgrade.core import MCIEnvironment_gymnasium as new_env

    if kind == "traj":
        orig = new_evm.EventManager.default_transportation_GB

        def broken(self, mode):
            # tier3 후보를 먼저 보게 만드는 최소 변형 (원본은 tier2 우선)
            props = self.properties['hospital']
            saved2, saved3 = props['hos_tier2_idx'], props['hos_tier3_idx']
            props['hos_tier2_idx'], props['hos_tier3_idx'] = saved3, saved2
            try:
                return orig(self, mode)
            finally:
                props['hos_tier2_idx'], props['hos_tier3_idx'] = saved2, saved3

        new_evm.EventManager.default_transportation_GB = broken
    elif kind == "metric":
        orig_sp = new_env.MCIEnvironment_gym.getSurvProb

        def broken_sp(self, time, p_class):
            return orig_sp(self, time, p_class) + 1e-12

        new_env.MCIEnvironment_gym.getSurvProb = broken_sp
    else:
        raise ValueError(f"알 수 없는 selftest 종류: {kind}")
    print(f"[SELFTEST] 신 코어에 '{kind}' 결함 주입 — 이 실행은 **FAIL 이 정상**")


# ----------------------------------------------------------------- 롤아웃
def rollout(factory, policy, seed: int, recorder=None):
    from sim_src_upgrade.verify import trace_hook

    trace_hook.set_recorder(recorder)
    try:
        env = factory(seed=seed)
        obs, _ = env.reset(seed=seed)
        reward = 0.0
        reward_wog = 0.0
        last_time = 0.0
        n_steps = 0
        while True:
            mask = np.asarray(env.action_masks(), dtype=bool)
            action = int(policy(obs, mask, env.unwrapped))
            obs, r, terminated, truncated, info = env.step(action)
            reward += float(r)
            reward_wog += float(info.get("r_woG", 0.0))
            last_time = float(info.get("time", 0.0))
            n_steps += 1
            if terminated or truncated:
                break
        prev = float(env.unwrapped.preventable)
        prev_wog = float(env.unwrapped.preventable_woG)
        pdr = 1.0 - reward / prev if prev > 0 else 0.0
        pdr_wog = 1.0 - reward_wog / prev_wog if prev_wog > 0 else 0.0
        return (reward, reward_wog, pdr, pdr_wog, last_time, float(n_steps))
    finally:
        trace_hook.set_recorder(None)


# ----------------------------------------------------------------- 메인
def run_gate(configs: dict[str, str], policy_specs, seeds, rule_core: str,
             mask_only: bool, verbose: bool) -> dict:
    from sim_src_upgrade.env_factory_fast import make_feature_env_fast, make_feature_env_old
    from sim_src_upgrade.verify import trace_hook

    trace_hook.instrument_both()

    n_pairs = 0
    fails: list[dict] = []
    devnull = io.StringIO()

    for key, cfg in configs.items():
        # mask_only="new" 는 "특징 obs 생성을 생략해도 결과가 같다"를 직접 증명한다
        # (구 쪽은 전체 obs 생성 유지). "both" 는 래퍼 자체의 일관성만 본다.
        fac_old = make_feature_env_old(cfg, mask_only=(mask_only == "both"))
        fac_new = make_feature_env_fast(cfg, mask_only=(mask_only in ("new", "both")))
        for label, kind, kwargs in policy_specs:
            pol_old = make_policy(kind, kwargs, "old")
            pol_new = make_policy(kind, kwargs, rule_core)
            for seed in seeds:
                rec_old, rec_new = trace_hook.Recorder(), trace_hook.Recorder()
                with contextlib.redirect_stdout(devnull):
                    m_old = rollout(fac_old, pol_old, seed, rec_old)
                    m_new = rollout(fac_new, pol_new, seed, rec_new)
                devnull.truncate(0)
                devnull.seek(0)
                n_pairs += 1

                d_old, d_new = rec_old.digest(), rec_new.digest()
                g1 = d_old == d_new
                g2 = all(a == b for a, b in zip(m_old, m_new))
                if not (g1 and g2):
                    first_div = next((i for i, (a, b) in enumerate(zip(rec_old.items, rec_new.items))
                                      if a != b), min(len(rec_old.items), len(rec_new.items)))
                    fails.append({
                        "region": key, "policy": label, "seed": int(seed),
                        "G1": g1, "G2": g2,
                        "n_events_old": len(rec_old), "n_events_new": len(rec_new),
                        "first_divergence_idx": first_div,
                        "old_at_div": rec_old.items[first_div] if first_div < len(rec_old.items) else None,
                        "new_at_div": rec_new.items[first_div] if first_div < len(rec_new.items) else None,
                        "metrics_old": dict(zip(METRIC_NAMES, m_old)),
                        "metrics_new": dict(zip(METRIC_NAMES, m_new)),
                    })
                    if verbose:
                        print(f"  FAIL {key} / {label} / seed={seed}  G1={g1} G2={g2}", flush=True)
        if verbose:
            print(f"[{key}] 완료 (누적 {n_pairs} 쌍, 실패 {len(fails)})", flush=True)

    return {"n_pairs": n_pairs, "n_fail": len(fails), "fails": fails[:20]}


def parse_args():
    ap = argparse.ArgumentParser(description="G0/G1/G2 구·신 코어 동치 게이트")
    ap.add_argument("--manifest", default="eval250",
                    choices=["eval250", "train1000", "natural"], help="좌표 출처")
    ap.add_argument("--n_regions", type=int, default=5)
    ap.add_argument("--n_eps", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--policies", default="full64:4,shin:2,lb3,capT3")
    ap.add_argument("--rule_core", default="new", choices=["new", "old"],
                    help="신 코어 env 와 짝지을 RuleManager 출처")
    ap.add_argument("--mask_only", default="none", choices=["none", "new", "both"],
                    help="MaskOnlyFeatureWrapper 적용 범위. new=신 코어만(=obs 생략이 무해함을 증명)")
    ap.add_argument("--gate", default="occ", choices=["occ", "psent"])
    ap.add_argument("--cared", default="1")
    ap.add_argument("--obs_variant", default="essential+load+valid")
    ap.add_argument("--h_pad", default="47", help="빈 문자열이면 미설정")
    ap.add_argument("--amb_num", default="", help="MCI_AMB_NUM (빈값=시나리오 기본)")
    ap.add_argument("--uav_num", default="", help="MCI_UAV_NUM (빈값=시나리오 기본)")
    ap.add_argument("--incident_size", default="", help="MCI_INCIDENT_SIZE")
    ap.add_argument("--capa_scale", default="", help="MCI_CAPA_SCALE")
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", default="none", choices=["none", "traj", "metric"],
                    help="게이트 민감도 자가진단 — 지정 시 FAIL 이 정상")
    ap.add_argument("--audit", action="store_true",
                    help="증분 카운터를 매 호출 전수스캔과 대조(느림). 카운터 도입 시 1회 필수")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    apply_env(args.gate, args.cared, args.obs_variant, args.h_pad,
              {"MCI_AMB_NUM": args.amb_num, "MCI_UAV_NUM": args.uav_num,
               "MCI_INCIDENT_SIZE": args.incident_size, "MCI_CAPA_SCALE": args.capa_scale})

    try:
        origin_sync.check()
        print("[G0] PASS")
    except Exception as exc:  # noqa: BLE001
        print(f"[G0] FAIL — {exc}")
        return 1

    path = {"eval250": EVAL_MANIFEST, "train1000": TRAIN_MANIFEST,
            "natural": NATURAL_MANIFEST}[args.manifest]
    with open(path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    keys = list(mani)
    step = max(1, len(keys) // args.n_regions)
    keys = keys[::step][:args.n_regions]
    configs = {k: (mani[k] if isinstance(mani[k], str) else mani[k]["path"]) for k in keys}

    policy_specs = build_policy_specs(args.policies, args.rule_core)
    seeds = list(range(args.seed0, args.seed0 + args.n_eps))
    if args.audit:
        from sim_src_upgrade.core import EventManager as new_evm
        new_evm.AUDIT_COUNTERS = True
        print("[AUDIT] 증분 카운터 상시 대조 ON")
    inject_selftest(args.selftest)

    print(f"[설정] gate={args.gate} cared={args.cared} obs={args.obs_variant} "
          f"h_pad={args.h_pad or '(unset)'} rule_core={args.rule_core} mask_only={args.mask_only}")
    print(f"[규모] 좌표 {len(configs)} × 정책 {len(policy_specs)} × 시드 {len(seeds)} "
          f"= {len(configs)*len(policy_specs)*len(seeds)} 쌍")

    t0 = time.time()
    res = run_gate(configs, policy_specs, seeds, args.rule_core,
                   args.mask_only, verbose=not args.quiet)
    res["elapsed_s"] = time.time() - t0
    res["config"] = vars(args)
    res["regions"] = list(configs)

    ok = res["n_fail"] == 0
    print(f"\n[G1/G2] {'PASS' if ok else 'FAIL'} — {res['n_pairs']}쌍 중 실패 {res['n_fail']} "
          f"({res['elapsed_s']:.1f}s)")
    if not ok:
        for f in res["fails"][:5]:
            print(f"  · {f['region']} / {f['policy']} / seed={f['seed']}: "
                  f"G1={f['G1']} G2={f['G2']} 최초분기 idx={f['first_divergence_idx']}")
            print(f"      old: {f['old_at_div']}")
            print(f"      new: {f['new_at_div']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        print(f"[저장] {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
