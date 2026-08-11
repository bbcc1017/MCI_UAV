"""G7-a 복제 동치 게이트 — `copy.deepcopy` 한 env 가 원본과 **똑같이** 굴러가는지 본다.

왜 별도 게이트인가
------------------
NCRP 플래너(`planner_policy`)와 오라클(`rollout_oracle`)은 결정마다 env 를 통째로
`copy.deepcopy` 하고 재시드해 상상 미래를 굴린다. 그런데 `deepcopy` 는 numpy **뷰**의
연결을 끊는다 — 뷰를 캐시하는 최적화는 비복제 경로에서 완벽히 통과하면서 복제 경로에서만
조용히 어긋난다(실제로 S2-11 이 그랬고, 이 게이트가 잡았다).

검사
----
1. **뷰 무결성** : 복제본에서 `_amb_t`/`_uav_t` 가 `amb_states[:,1]` 의 뷰인가
2. **전진 동치** : 같은 액션열을 원본·복제본에 같은 rng 로 먹여 상태·보상이 완전히 같은가
3. **원본 무접촉** : 복제본을 굴린 뒤 원본 상태가 안 변했는가(플래너 계약)
4. **pickle 왕복** : 직렬화 후에도 1~3 이 성립하는가

    python src/sim_src_upgrade/verify/verify_deepcopy.py
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import pickle
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

import numpy as np  # noqa: E402

from sim_src_upgrade._paths import REPO  # noqa: E402

EVAL_MANIFEST = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
RULE = "START, RedOnly, Red Both_AMBFirst, Yellow Both_AMBFirst"


def _state_digest(env) -> tuple:
    u = env.unwrapped
    s = u.en_manager.en_status
    ev = u.ev_manager
    return (
        s['patient']['p_states'].tobytes(),
        s['hospital']['h_states'].tobytes(),
        s['ambulance']['amb_states'].tobytes(),
        s['uav']['uav_states'].tobytes(),
        s['patient']['p_sent'].tobytes(),
        float.hex(float(ev.time)),
        len(ev.event_queue),
        int(getattr(ev, "_n_cared", -1)),
        bytes(getattr(ev, "_in_flight_cnt", np.zeros(0, np.int32))),
    )


def _views_ok(env) -> bool:
    ev = env.unwrapped.ev_manager
    amb = ev.status['ambulance']['amb_states']
    uav = ev.status['uav']['uav_states']
    a = getattr(ev, "_amb_t", None)
    u = getattr(ev, "_uav_t", None)
    if a is None or u is None:
        return True                      # 뷰 캐시를 안 쓰는 구현이면 해당 없음
    return (a.base is amb or np.shares_memory(a, amb)) and \
           (u.base is uav or np.shares_memory(u, uav))


def run_case(clone_fn, label: str, cfg: str, seed: int, n_pre: int, n_post: int) -> list[str]:
    """복제 전 n_pre 스텝 → 복제 → 양쪽에 같은 액션 n_post 스텝 → 대조."""
    from sim_src_upgrade.env_factory_fast import make_feature_env_fast
    from sim_src_upgrade.verify.policies import make_rule_policy

    fails: list[str] = []
    fac = make_feature_env_fast(cfg)
    pol = make_rule_policy(RULE)
    env = fac(seed=seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_pre):
        m = np.asarray(env.action_masks(), dtype=bool)
        a = int(pol(obs, m, env.unwrapped))
        obs, _r, t, tr, _i = env.step(a)
        if t or tr:
            return [f"{label}: 복제 전에 에피소드가 끝남(n_pre={n_pre}) — 케이스 무효"]

    before = _state_digest(env)
    clone = clone_fn(env)

    if not _views_ok(clone):
        fails.append(f"{label}: 복제본의 잔여시간 뷰가 차량 상태 배열과 메모리를 공유하지 않음")

    # 같은 rng 로 재시드해 양쪽을 동일 조건으로 만든다 (플래너가 하는 재시드와 같은 방식)
    rng_seed = 424242
    env.unwrapped.ev_manager.set_seed(np.random.default_rng(rng_seed))
    clone.unwrapped.ev_manager.set_seed(np.random.default_rng(rng_seed))

    obs_o, obs_c = obs, obs
    for i in range(n_post):
        m_o = np.asarray(env.action_masks(), dtype=bool)
        m_c = np.asarray(clone.action_masks(), dtype=bool)
        if not np.array_equal(m_o, m_c):
            fails.append(f"{label}: step{i} action mask 불일치")
            break
        a = int(pol(obs_o, m_o, env.unwrapped))
        obs_o, r_o, t_o, tr_o, i_o = env.step(a)
        obs_c, r_c, t_c, tr_c, i_c = clone.step(a)
        if not np.array_equal(np.asarray(obs_o), np.asarray(obs_c)):
            fails.append(f"{label}: step{i} obs 불일치")
            break
        if (float(r_o), t_o, tr_o, i_o.get("r_woG"), i_o.get("time")) != \
           (float(r_c), t_c, tr_c, i_c.get("r_woG"), i_c.get("time")):
            fails.append(f"{label}: step{i} 보상/종료 불일치 {r_o}/{r_c}")
            break
        if _state_digest(env) != _state_digest(clone):
            fails.append(f"{label}: step{i} 내부 상태 불일치")
            break
        if t_o or tr_o:
            break
    return fails


def run_isolation(cfg: str, seed: int, n_pre: int, n_post: int) -> list[str]:
    """복제본만 굴렸을 때 원본이 안 변하는지 — 플래너의 '원본 무접촉' 계약."""
    from sim_src_upgrade.env_factory_fast import make_feature_env_fast
    from sim_src_upgrade.verify.policies import make_rule_policy

    fac = make_feature_env_fast(cfg)
    pol = make_rule_policy(RULE)
    env = fac(seed=seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_pre):
        m = np.asarray(env.action_masks(), dtype=bool)
        obs, _r, t, tr, _i = env.step(int(pol(obs, m, env.unwrapped)))
        if t or tr:
            return []
    before = _state_digest(env)
    clone = copy.deepcopy(env)
    clone.unwrapped.ev_manager.set_seed(np.random.default_rng(999))
    o = obs
    for _ in range(n_post):
        m = np.asarray(clone.action_masks(), dtype=bool)
        o, _r, t, tr, _i = clone.step(int(pol(o, m, clone.unwrapped)))
        if t or tr:
            break
    return [] if _state_digest(env) == before else ["원본 무접촉 위반: 복제본 실행이 원본 상태를 바꿈"]


def main() -> int:
    ap = argparse.ArgumentParser(description="G7-a 복제 동치 게이트")
    ap.add_argument("--n_regions", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_pre", type=int, default=6)
    ap.add_argument("--n_post", type=int, default=12)
    args = ap.parse_args()

    os.environ.update(MCI_CAP_GATE="occ", MCI_OBS_VARIANT="essential+load+valid",
                      MCI_H_PAD="47", MCI_REWARD_MODE="woG")
    os.environ.setdefault("MCI_CARED_OBS", "1")

    with open(EVAL_MANIFEST, "r", encoding="utf-8") as f:
        mani = json.load(f)
    keys = list(mani)
    step = max(1, len(keys) // args.n_regions)
    keys = keys[::step][:args.n_regions]

    clone_fns = [("deepcopy", copy.deepcopy),
                 ("pickle", lambda e: pickle.loads(pickle.dumps(e)))]

    fails: list[str] = []
    n_case = 0
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull):
            for k in keys:
                cfg = mani[k] if isinstance(mani[k], str) else mani[k]["path"]
                for s in range(args.seeds):
                    for name, fn in clone_fns:
                        n_case += 1
                        try:
                            fails += [f"[{k}/s{s}] {m}" for m in
                                      run_case(fn, name, cfg, s, args.n_pre, args.n_post)]
                        except Exception as exc:  # noqa: BLE001
                            fails.append(f"[{k}/s{s}] {name}: 예외 {exc!r}")
                    n_case += 1
                    fails += [f"[{k}/s{s}] {m}" for m in
                              run_isolation(cfg, s, args.n_pre, args.n_post)]
    finally:
        devnull.close()

    ok = not fails
    print(f"[G7-a 복제] {'PASS' if ok else 'FAIL'} — {n_case}케이스 중 실패 {len(fails)}")
    for f in fails[:10]:
        print(f"  · {f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
