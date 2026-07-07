"""스코어 정책 평가·Gate1 판정 (플랜 v2 추출 트랙 B5).

program_eval.py 골격 복제·확장. 같은 (region, seed=11000+ep) 실현에서 정책들을 각자 롤아웃
→ per-ep PDR_woG → 평균·paired 승/무/패·격차회수율. 내장 기준선(heur/lb_T4/prog(4:0.8))은
지역별 best rule(CSV) 로, 스코어 정책은 지역불변 GENERIC_RULE + w·φ argmax 로 실행한다.

  * 규칙류(heur/lb_T4/prog/score)는 essential env(정규화 없음, obs 비의존) 로 실행.
  * --with_rl 이면 RL 모델도 paired 에 포함(vecnorm 동결·pointer_policy/hospital_set_extractor
    import 후 MaskablePPO.load). RL 은 자기 obs_variant+자기 vecnorm 으로 env 를 따로 빌드하되
    reset(seed=s) 실현이 dynamics 와 무관하므로 규칙류와 paired 성립.
  * occ 게이트 고정(탐구단계). Pool maxtasksperchild=1 + OMP=1 핀.

--variants "이름:w_json:mode:T:guard,..."   (콤마 리스트)
    w_json : score_fit/score_cma json (w_vec + config.uav_time_factor/uav_red_only 재사용)
    mode   : timesave | joint
    T      : none(정원제 미적용) | 숫자(고정 T_hard) | lookup:<spec>(적응 정원, score_cma._make_T_lookup)
    guard  : none | 숫자(적격<guard 면 make_cap_policy(GENERIC,4) 폴백)
--with_rl "디렉터리=obs_variant" 또는 "이름=디렉터리=obs_variant" (상대경로는 REPO 기준)
--tune_pool40 : sigungu_osrm 매니페스트에서 score_cma.select_tune_regions(40) 지역·sigcd 매칭
    (CEM 과 동일 40지역 CRN — --n_eps 30 이면 CEM 평가 재현). heur_csv 도 sigungu 로 자동 전환.

격차회수율 = (lb_ref − PDR)/(lb_ref − rl_ref). Gate1 = (PDR < prog_ref) AND (회수율 ≥ gap_recover).
기본 앵커: lb_ref=0.1199(LB-T4) rl_ref=0.0923(L3) prog_ref=0.1162(프로그램4:0.8).

예 튜닝풀: python src/rl_src/score_eval.py --tune_pool40 --n_eps 30 --workers 20 \
  --variants "score_T4:results/rl/redesign/score_cma_v3wide.json:timesave:4:none" \
  --out results/rl/redesign/score_eval_tune40.csv
예 Gate1: python src/rl_src/score_eval.py --n_eps 1000 --workers 34 \
  --variants "score_T4:...v3wide.json:timesave:4:none,score_A1:...v3wide_A1.json:timesave:none:none" \
  --with_rl results/rl/redesign/v3_wide_s0=essential+load --out results/rl/redesign/score_gate1.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()


# ------------------------------------------------------------------ 스펙 파싱
def parse_variant(tok):
    """'이름:w_json:mode:T:guard' → dict. T 의 'lookup:<spec>' 콜론을 안전 파싱.

    parts 를 콜론으로 나눈 뒤 name/w_json/mode 는 앞 3개, guard 는 맨 뒤 1개,
    나머지(가운데)를 join 하여 T 로 복원한다(lookup:rho_step 대응).
    """
    p = tok.split(":")
    if len(p) < 5:
        raise ValueError(f"--variants 항목 형식 오류(이름:w_json:mode:T:guard): {tok!r}")
    name, w_json, mode = p[0], p[1], p[2]
    guard_spec = p[-1]
    T_spec = ":".join(p[3:-1])
    if not os.path.isabs(w_json):
        w_json = os.path.join(REPO, w_json)
    with open(w_json, encoding="utf-8") as f:
        fit = json.load(f)
    cfg = fit.get("config", {})
    # T 해석
    if T_spec == "none":
        T_hard, T_lookup_spec = None, None
    elif T_spec.startswith("lookup:"):
        T_hard, T_lookup_spec = None, T_spec.split(":", 1)[1]
    else:
        T_hard, T_lookup_spec = float(T_spec), None
    guard_n = None if guard_spec == "none" else int(guard_spec)
    return dict(name=name, w=list(fit["w_vec"]), mode=mode,
                T_hard=T_hard, T_lookup_spec=T_lookup_spec, guard_n=guard_n,
                uav_time_factor=float(cfg.get("uav_time_factor", 0.8)),
                uav_red_only=bool(int(cfg.get("uav_red_only", 1))), w_json=w_json)


def parse_rl(spec):
    """'디렉터리=variant' 또는 '이름=디렉터리=variant' → dict(name, mdir, variant)."""
    parts = spec.split("=")
    if len(parts) == 2:
        name, mdir, variant = "rl", parts[0], parts[1]
    elif len(parts) == 3:
        name, mdir, variant = parts
    else:
        raise ValueError(f"--with_rl 형식 오류(디렉터리=variant 또는 이름=디렉터리=variant): {spec!r}")
    if not os.path.isabs(mdir):
        mdir = os.path.join(REPO, mdir)
    return dict(name=name, mdir=mdir, variant=variant)


# ------------------------------------------------------------------ 롤아웃
def _pdr(factory, pol, seed):
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    w = 0.0
    while not done:
        m = env.action_masks()
        a = pol(obs, m, env.unwrapped)
        obs, r, te, tr, info = env.step(a)
        w += info.get("r_woG", 0.0)
        done = te or tr
    prev = env.unwrapped.preventable_woG
    return (1.0 - w / prev) if prev > 0 else 0.0


def worker(job):
    region, cfg, best_rule, score_specs, rl_specs, n_eps = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy
    from program_policy import make_program_policy
    from score_policy import make_score_policy
    from score_cma import GENERIC_RULE, _make_T_lookup
    from evaluate import ppo_policy
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    try:
        with _suppress_stdout():
            os.environ["MCI_OBS_VARIANT"] = "essential"
            rule_fac = make_feature_env(cfg, None)
            rule_fac(seed=SEED)                      # essential 로 캐시 고정
            entries = [
                ("heur", rule_fac, make_heuristic_policy(best_rule)),
                ("lb_T4", rule_fac, make_cap_policy(best_rule, 4)),
                ("prog", rule_fac, make_program_policy(best_rule, T=4, uav_time_factor=0.8,
                                                       uav_red_only=True)),
            ]
            for sc in score_specs:                   # 스코어 정책 = 지역불변 GENERIC_RULE
                entries.append((sc["name"], rule_fac, make_score_policy(
                    sc["w"], GENERIC_RULE, mode=sc["mode"], T_hard=sc["T_hard"],
                    T_lookup=_make_T_lookup(sc["T_lookup_spec"]), guard_n=sc["guard_n"],
                    uav_time_factor=sc["uav_time_factor"], uav_red_only=sc["uav_red_only"])))
            for rl in rl_specs:                      # RL — 자기 variant + 자기 vecnorm
                os.environ["MCI_OBS_VARIANT"] = rl["variant"]
                zip_path = os.path.join(rl["mdir"], "final_model.zip")
                model = MaskablePPO.load(zip_path, device="cpu")
                vn = os.path.join(rl["mdir"], "vecnormalize.pkl")
                norm = load_vecnorm(vn) if os.path.exists(vn) else None
                rl_fac = make_feature_env(cfg, norm)
                rl_fac(seed=SEED)
                entries.append((rl["name"], rl_fac, ppo_policy(model)))
            names = [e[0] for e in entries]
            P = {n: np.zeros(n_eps) for n in names}
            for ep in range(n_eps):
                s = SEED + ep
                for n, fac, pol in entries:
                    P[n][ep] = _pdr(fac, pol, s)
        out = dict(region=region, ok=True, _P={n: P[n].tolist() for n in names}, names=names)
        for n in names:
            out[f"PDR_{n}"] = float(P[n].mean())
        return out
    except Exception as e:
        import traceback
        return dict(region=region, ok=False, err=(str(e) + traceback.format_exc())[:400])


# ------------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--heur_csv", default=os.path.join(REPO, "results/sido_osrm_heuristic_best.csv"))
    ap.add_argument("--match", choices=["name", "sigcd"], default="name",
                    help="best_rule 매칭: name(시도) / sigcd(시군구·holdout)")
    ap.add_argument("--variants", default="", help="이름:w_json:mode:T:guard, ...")
    ap.add_argument("--with_rl", default="", help="디렉터리=variant 또는 이름=디렉터리=variant (콤마 여러개)")
    ap.add_argument("--regions", default="", help="쉼표구분 키 서브셋(기본 시도17 또는 매니페스트 전 키)")
    ap.add_argument("--tune_pool40", action="store_true",
                    help="sigungu_osrm 40지역(score_cma.select_tune_regions)·sigcd 매칭(CEM CRN 재현)")
    ap.add_argument("--tune_k", type=int, default=40)
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/score_eval.csv"))
    ap.add_argument("--dump_pe", default="",
                    help="설정 시 지역별 per-ep PDR 배열을 npz(regions,names,pdr[R,P,eps])로 덤프"
                         " — paired(변형 vs 임의 baseline) 오프라인 분석용, 기본 off(기존 동작 불변)")
    # 격차회수율 앵커
    ap.add_argument("--lb_ref", type=float, default=0.1199)
    ap.add_argument("--rl_ref", type=float, default=0.0923)
    ap.add_argument("--prog_ref", type=float, default=0.1162)
    ap.add_argument("--gap_recover", type=float, default=0.35)
    A = ap.parse_args()
    import numpy as np
    from score_cma import select_tune_regions

    # 튜닝풀 모드: 매니페스트/휴리CSV/매칭/지역 자동 전환
    if A.tune_pool40:
        A.manifest = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json")
        A.heur_csv = os.path.join(REPO, "results/sigungu_heuristic_best.csv")
        A.match = "sigcd"

    manifest = json.load(open(A.manifest, encoding="utf-8"))

    # 휴리 best_rule 룩업 (BOM 대응). match=name→region 키, sigcd→sigcd 키.
    best_by = {}
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            best_by[r["region"] if A.match == "name" else r["sigcd"]] = r["best_rule"]

    def _lookup(key):
        if A.match == "name":
            return best_by.get(key)
        digits = [t for t in key.split("_") if t.isdigit()]
        return best_by.get(digits[0]) if digits else None

    # 대상 키 목록
    if A.tune_pool40:
        keys = [rg for rg, _ in select_tune_regions(A.manifest, A.tune_k)]
    elif A.regions:
        keys = [k for k in A.regions.split(",") if k in manifest]
    elif A.match == "name":
        keys = [k for k in SIDO17 if k in manifest]
    else:
        keys = list(manifest.keys())

    score_specs = [parse_variant(t.strip()) for t in A.variants.split(",") if t.strip()]
    rl_specs = [parse_rl(t.strip()) for t in A.with_rl.split(",") if t.strip()]

    jobs = [(k, manifest[k], _lookup(k), score_specs, rl_specs, A.n_eps)
            for k in keys if k in manifest and _lookup(k) is not None]
    print(f"[score_eval] jobs={len(jobs)} match={A.match} tune_pool40={A.tune_pool40} "
          f"variants={[s['name'] for s in score_specs]} rl={[r['name'] for r in rl_specs]} "
          f"n_eps={A.n_eps} workers={A.workers}", flush=True)
    for s in score_specs:
        print(f"    variant {s['name']}: mode={s['mode']} T_hard={s['T_hard']} "
              f"T_lookup={s['T_lookup_spec']} guard={s['guard_n']} "
              f"uav_f={s['uav_time_factor']} uav_red_only={s['uav_red_only']} "
              f"({os.path.basename(s['w_json'])})", flush=True)

    res, t0 = [], time.time()
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            res.append(r)
            if r["ok"]:
                print(f"  [{k}/{len(jobs)}] {r['region']}: "
                      + " ".join(f"{n}={r['PDR_'+n]:.4f}" for n in r["names"])
                      + f"  ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{k}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}", flush=True)
    ok = [r for r in res if r["ok"]]
    if not ok:
        print("전부 실패", flush=True)
        return
    names = ok[0]["names"]

    with open(A.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region"] + [f"PDR_{n}" for n in names])
        w.writeheader()
        for r in ok:
            w.writerow({c: r.get(c) for c in ["region"] + [f"PDR_{n}" for n in names]})
    print(f"\n저장 {A.out} (지역 {len(ok)}) wall={time.time()-t0:.0f}s", flush=True)

    if A.dump_pe:  # 추가 기능(default off): 지역별 per-ep PDR 을 npz 로 — 임의 baseline paired 분석용
        regs = [r["region"] for r in ok]
        pdr = np.array([[r["_P"][n] for n in names] for r in ok], dtype=float)  # (R, P, eps)
        np.savez_compressed(A.dump_pe, regions=np.array(regs), names=np.array(names), pdr=pdr)
        print(f"저장(per-ep) {A.dump_pe}  shape={pdr.shape} (지역×정책×ep)", flush=True)

    means = {n: float(np.mean([r[f"PDR_{n}"] for r in ok])) for n in names}
    print("\n=== 평균 PDR_woG (낮을수록 좋음) ===", flush=True)
    for n in names:
        print(f"  {n:>16}: {means[n]:.4f}", flush=True)

    def paired(a, b):  # a=대상, b=baseline; 양수=a 우수(PDR 낮음)
        vals = []
        for r in ok:
            d = np.array(r["_P"][b]) - np.array(r["_P"][a])
            md = d.mean()
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            vals.append((md, "win" if md > ci else "loss" if md < -ci else "tie"))
        return (np.mean([v[0] for v in vals]),
                sum(v[1] == "win" for v in vals), sum(v[1] == "tie" for v in vals),
                sum(v[1] == "loss" for v in vals))

    targets = [s["name"] for s in score_specs] + [r["name"] for r in rl_specs]
    for base in ("prog", "lb_T4"):
        if base not in names:
            continue
        print(f"\n=== vs {base} (양수=대상 우수, 승/무/패 across 지역) ===", flush=True)
        for n in targets:
            if n == base:
                continue
            md, wi, ti, lo = paired(n, base)
            print(f"  {n:>16}: {md:+.4f} ({wi}/{ti}/{lo})", flush=True)

    # ---- 격차회수율 + Gate1 판정 ----
    denom = A.lb_ref - A.rl_ref
    print(f"\n=== 격차회수율·Gate1 (앵커: lb_ref={A.lb_ref} rl_ref={A.rl_ref} "
          f"prog_ref={A.prog_ref} 임계={A.gap_recover:.0%}) ===", flush=True)
    print(f"    Gate1 통과 조건: PDR < {A.prog_ref} AND 회수율 ≥ {A.gap_recover:.0%} "
          f"(즉 PDR ≤ {A.lb_ref - A.gap_recover*denom:.4f})", flush=True)
    # 인런 측정 앵커(교차확인용)
    lb_in = means.get("lb_T4")
    rl_in = None
    for r in rl_specs:
        rl_in = means.get(r["name"])
    for n in targets:
        pdr = means[n]
        rec_fixed = (A.lb_ref - pdr) / denom if denom else float("nan")
        gate = "PASS" if (pdr < A.prog_ref and rec_fixed >= A.gap_recover) else "fail"
        extra = ""
        if lb_in is not None and rl_in is not None and (lb_in - rl_in) != 0:
            extra = f"  [인런회수율={(lb_in - pdr)/(lb_in - rl_in):+.1%}]"
        print(f"  {n:>16}: PDR={pdr:.4f}  고정앵커회수율={rec_fixed:+.1%}  Gate1={gate}{extra}",
              flush=True)


if __name__ == "__main__":
    main()
