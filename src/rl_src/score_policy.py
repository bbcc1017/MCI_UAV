"""선형 스코어 정책 (플랜 v2 추출 트랙 B0-2) — dest 선택을 `argmax_h w·φ(h,ctx)` 로.

최강 RL(포인터 head)의 로짓 가법 구조 `L[c,d,m]=f_class(ctx)+S[d,m]+g_mode(ctx)` 에서 dest
랭킹 `S[d,m]` 를 **지역불변 선형 스코어** `w·φ(h,ctx)` 로 근사한다. φ 가 전부 지역불변·유계
(score_features)라 argmax 가 순열등변 → 전국 단일 스코어 정책이 성립한다.

정책 fn(ro, mask, env_unwrapped)->action. obs(ro) 비의존 — ctx 는 dict obs·en_properties 에서
재계산(⚠️평탄 obs 슬라이싱 금지). make_cap_policy/make_program_policy 와 동일 인터페이스.

구조(3축):
  class : program_policy 와 동일한 64룰 경로(rule.select). c<0 또는 dest=0 이면 stay 폴백.
  dest  : 적격(마스크 인코딩: tier3·헬기장·게이트) 중 argmax w·φ. T 있으면 정원제(p_sent<T).
  mode  : mode="timesave" 면 program 의 UAV 시간절감 규칙(raw 분 비교)으로 m 결정 후 h만 argmax;
          mode="joint" 면 (h,m) 후보를 동시에 argmax(φ11 is_uav·φ12 dt_uav 가 모드축을 표현).

파라미터
  w              : 길이 K_PHI 가중 벡터(score_features.PHI_NAMES 순서).
  class_rule     : 64룰 문자열(예 GENERIC_RULE). generic(범용) 룰도 허용.
  mode           : "timesave" | "joint".
  T_hard         : 고정 정원 T — pool={p_sent<T} 우선(비면 전체 적격서 최소발송 overflow, LB 재현).
  T_lookup       : f(rho, n_elig)->T 훅(있으면 T_hard 무시). 적응 정원.
  guard_n        : 적격 병원 < guard_n 이면 make_cap_policy(class_rule,4) 로 폴백(희소지역 안전).
  uav_time_factor: timesave UAV 채택 문턱(t_uav < factor·t_amb).
  uav_red_only   : timesave/joint 에서 UAV 를 Red(c==0)에만 허용.

--selftest: w=(−1,0,…,0)(eta만)+T_hard=4+timesave 가 make_program_policy(rule,4,0.8) 와
  서울·강원 3ep(seed 11000~)에서 **액션열 완전 일치**함을 검증(동치 봉인).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

sys.path.insert(0, os.path.dirname(__file__))
_SIM_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sim_src"))
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)

import numpy as np

from distill_policy import parse_rule
from loadbalance_heuristic import _codec_from_mask, make_cap_policy, H_DEFAULT
from score_features import build_ctx, build_phi, compute_static, K_PHI


def make_score_policy(w, class_rule, mode="timesave", T_hard=None, T_lookup=None,
                      guard_n=None, uav_time_factor=0.8, uav_red_only=True, H=H_DEFAULT):
    """선형 스코어 정책 팩토리 → fn(ro, mask, env_unwrapped)->flat action."""
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.shape[0] != K_PHI:
        raise ValueError(f"w 길이 {w.shape[0]} != K_PHI({K_PHI})")
    if mode not in ("timesave", "joint"):
        raise ValueError(f"mode 는 'timesave'|'joint' (got {mode})")
    rule = parse_rule(class_rule)
    guard_policy = make_cap_policy(class_rule, 4, H=H) if guard_n else None
    st = {"em": None, "encode": None, "static": None}

    def _sync(u, mask_len):
        # per-env(en_manager 아이덴티티) 캐시 — 멀티지역 재초기화(코덱·정적 ctx·rule init).
        if st["em"] is u.en_manager:
            return
        st["encode"] = _codec_from_mask(mask_len, H)
        st["static"] = compute_static(u)
        rule.set_seed(np.random.default_rng(0))
        rule.init_with_scenario({"EntityManager": u.en_manager})
        st["em"] = u.en_manager

    def fn(ro, mask, env):
        _sync(env, len(mask))
        enc = st["encode"]
        dobs = env.en_manager.get_full_obs()
        dobs["time"] = env.ev_manager.time
        c, d0, m0 = rule.select(dobs)
        base = enc(0, 0, 0) if c < 0 else enc(c, d0, m0)

        def _valid(a):
            return 0 <= a < len(mask) and mask[a]

        def fb():
            if _valid(base):
                return base
            v = np.flatnonzero(mask)
            return int(v[0]) if v.size else 0

        if c < 0 or d0 == 0:
            return fb()

        # 적격 병원(마스크 기반) — 모드별. AMB+UAV 형(192) 이면 UAV 후보도 존재.
        has_uav = (len(mask) == 2 * (H + 1) * 2)
        elig_amb = np.array([i for i in range(H) if _valid(enc(c, i + 1, 0))], dtype=int)
        elig_uav = (np.array([i for i in range(H) if _valid(enc(c, i + 1, 1))], dtype=int)
                    if has_uav else np.zeros(0, dtype=int))
        n_elig = max(elig_amb.size, elig_uav.size)
        if guard_policy is not None and n_elig < guard_n:
            return guard_policy(ro, mask, env)   # 희소지역 — 검증된 LB-T4 로 폴백

        ctx = build_ctx(env, static=st["static"], dobs=dobs)
        psent = ctx["p_sent"]
        # T 결정: T_lookup(적응) 우선, 없으면 T_hard(고정). 둘 다 None 이면 정원제 미적용.
        if T_lookup is not None:
            T = float(T_lookup(ctx["rho"], n_elig))
        elif T_hard is not None:
            T = float(T_hard)
        else:
            T = None

        def _score(elig, m):
            """적격 전체(elig)에 대한 φ·w 스코어 — 상대 φ(eta_rank 등)를 '적격 내'로 계산."""
            phi = build_phi(env, c, m, elig, ctx=ctx)      # (len(elig), K)
            return phi @ w

        def _pick_h(elig, m, overflow):
            """스코어 argmax 목적지. T 있으면 p_sent<T 풀 우선, overflow=True 면 초과 시 최소발송."""
            if elig.size == 0:
                return None
            scores = _score(elig, m)
            if T is not None:
                under = np.flatnonzero(psent[elig] < T)
                if under.size:
                    return int(elig[under[int(np.argmax(scores[under]))]])
                if not overflow:
                    return None
                # 전부 정원 초과 → 가장 덜 보낸 곳(LB/program overflow 재현)
                return int(elig[int(np.argmin(psent[elig]))])
            return int(elig[int(np.argmax(scores))])

        # ------------------------------------------------- joint: (h,m) 동시 argmax
        if mode == "joint":
            cand = [(i, 0) for i in elig_amb.tolist()]
            if has_uav and (c == 0 or not uav_red_only):
                cand += [(i, 1) for i in elig_uav.tolist()]
            if not cand:
                return fb()
            cand = np.asarray(cand, dtype=int)
            scores = build_phi(env, c, None, cand, ctx=ctx) @ w   # φ11·12 가 모드축 표현
            ph = psent[cand[:, 0]]
            if T is not None:
                under = np.flatnonzero(ph < T)
                j = int(under[int(np.argmax(scores[under]))]) if under.size \
                    else int(np.argmin(ph))                       # overflow: 최소발송
            else:
                j = int(np.argmax(scores))
            hi, mi = int(cand[j, 0]), int(cand[j, 1])
            a = enc(c, hi + 1, mi)
            return a if _valid(a) else fb()

        # ------------------------------------------------- timesave: m 결정 후 h argmax
        best_amb = _pick_h(elig_amb, 0, overflow=True)
        if best_amb is None:
            return fb()
        if has_uav and (c == 0 or not uav_red_only):
            best_uav = _pick_h(elig_uav, 1, overflow=False)   # 풀 비면 UAV 미채택(program 동일)
            if best_uav is not None:
                t_amb, t_uav = ctx["t_amb"], ctx["t_uav"]
                # UAV 최속이 AMB 최속보다 factor 배 이상 빠르면 UAV(raw 분 비교)
                if t_uav[best_uav] < uav_time_factor * t_amb[best_amb]:
                    a = enc(c, best_uav + 1, 1)
                    if _valid(a):
                        return a
        a = enc(c, best_amb + 1, 0)
        return a if _valid(a) else fb()

    fn.w = w
    fn.mode = mode
    return fn


# ----------------------------------------------------------------- selftest CLI
def _selftest(manifest_path, regions, n_eps, rule_name, seed_base=11000):
    """동치 봉인: eta-only w + T_hard=4 + timesave == make_program_policy(rule,4,0.8)."""
    import json

    from viper_distill import make_feature_env, _suppress_stdout
    from program_policy import make_program_policy

    os.environ.setdefault("MCI_OBS_VARIANT", "essential")   # 규칙류 convention(norm 불요)
    os.environ.setdefault("MCI_CAP_GATE", "occ")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    w_eta = np.zeros(K_PHI); w_eta[0] = -1.0   # score = −eta → argmax = 최근접(argmin eta)
    total_steps, total_dec, mism = 0, 0, []
    for region in regions:
        if region not in manifest:
            print(f"  [skip] {region} — 매니페스트에 없음", flush=True)
            continue
        cfg = manifest[region]
        prog = make_program_policy(rule_name, T=4, uav_time_factor=0.8)   # 기본 uav_red_only=True
        score = make_score_policy(w_eta, rule_name, mode="timesave",
                                  T_hard=4, uav_time_factor=0.8, uav_red_only=True)
        with _suppress_stdout():
            fac = make_feature_env(cfg, None)
            for ep in range(n_eps):
                env = fac(seed=seed_base + ep)
                obs, _ = env.reset(seed=seed_base + ep)
                done = False
                step = 0
                while not done:
                    mask = np.asarray(env.action_masks(), bool)
                    a_p = int(prog(obs, mask, env.unwrapped))
                    a_s = int(score(obs, mask, env.unwrapped))
                    total_steps += 1
                    H = env.unwrapped.en_manager.en_properties['hospital']['hos_num']
                    nd = H + 1
                    dp = (a_p % (nd * 2)) // 2 if len(mask) == 2 * nd * 2 else a_p % nd
                    if dp >= 1:
                        total_dec += 1
                    if a_p != a_s:
                        mism.append((region, ep, step, a_p, a_s))
                    obs, r, term, trunc, info = env.step(a_p)   # 기준(program)으로 진행
                    done = term or trunc
                    step += 1
        print(f"  [{region}] {n_eps}ep 완료 (누적 step={total_steps}, 이송결정={total_dec}, "
              f"불일치={len(mism)})", flush=True)

    print(f"\n=== selftest: rule='{rule_name}' regions={regions} n_eps={n_eps} ===", flush=True)
    print(f"총 step={total_steps}  이송결정={total_dec}  불일치={len(mism)}", flush=True)
    if mism:
        print("불일치 샘플(최대 20):", flush=True)
        for m in mism[:20]:
            print(f"  region={m[0]} ep={m[1]} step={m[2]} program={m[3]} score={m[4]}", flush=True)
        print("❌ 동치 실패 — 원인 조사 필요", flush=True)
        return 1
    print("✅ 액션열 완전 일치 (score[eta-only,T=4,timesave] == program[T=4,f=0.8])", flush=True)
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest",
                    default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                      os.pardir, os.pardir)),
                                         "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--regions", default="서울,강원")
    ap.add_argument("--n_eps", type=int, default=3)
    ap.add_argument("--rule", default="START, YellowNearest, Red Both_AMBFirst, Yellow Both_AMBFirst")
    A = ap.parse_args()
    if A.selftest:
        rc = _selftest(A.manifest, [r for r in A.regions.split(",") if r], A.n_eps, A.rule)
        sys.exit(rc)
    ap.error("현재는 --selftest 만 지원(정책 팩토리는 import 용).")


if __name__ == "__main__":
    main()
