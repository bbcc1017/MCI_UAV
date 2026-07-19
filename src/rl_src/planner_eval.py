"""P1 NCRP 플래너 판정 드라이버 (계획 §4.1 표 #3) — rollout_oracle CLI 관례 승계.

같은 시드(CRN)에서 ①챔피언 greedy 베이스라인 → ②플래너(TruncatedRolloutPlanner) 에피소드를
paired 측정한다. pdr_base 는 재계산이 아니라 오라클(lookahead_episode)과 동일하게 **같은
에피소드를 greedy 로 한 번 더 돌려** 얻는다(reset(seed) 가 ev_manager 까지 재시드 →
paired 성립, 2026-07-03 재현성 계약).

재현성 앵커: `--clairvoyant --h -1 --leaf none --K 8` 은 rollout_oracle 과 행동·수치가
비트 단위로 동일해야 한다(oracle_headroom_sido17_v4.csv 재현 = 구현 합격선; 단 그 CSV 는
model_dir=v4_plr2_s0 으로 생성됐으므로 앵커 실행도 같은 model_dir 전제).

관례(오라클 승계): 시도17=판정 전용(튜닝 금지), 튜닝풀=--tune_pool40(시군구 sigcd 균등 40,
score_cma.select_tune_regions 재사용), seed0=11000 CRN, Pool(maxtasksperchild=1)·OMP=1 핀·
워커 th.set_num_threads(1)·_suppress_stdout, CSV (region,ep) 재개, --validate(validate_clone).

CSV 컬럼: region,ep,pdr_planner,pdr_base,n_dec,n_switch,ms_per_dec,sec
  n_dec=lookahead 수행 결정 수, n_switch=greedy 이탈 수, ms_per_dec=lookahead 결정당 평균 ms
  (배포 지연 논거), sec=에피소드 전체(베이스라인 포함) 소요.

예(앵커):   PYTHONIOENCODING=utf-8 python src/rl_src/planner_eval.py \
    --clairvoyant --h -1 --leaf none --K 8 --regions 서울 --n_eps 3 --workers 3 \
    --out /tmp/planner_anchor.csv
예(판정):   ... --manifest scenarios/manifests/sido_osrm_manifest.json --n_eps 1000 \
    --K 8 --h 10 --m 2 --leaf reg --workers 32 --out results/rl/redesign/planner_sido17.csv
예(튜닝풀): ... --tune_pool40 --n_eps 50 --K 8 --h 10 --m 2 --leaf reg --workers 32
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
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # 평가는 info['r_woG'] 직접 읽음(모드 무관)

import numpy as np
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*action_masks.*")
_warnings.filterwarnings("ignore", category=UserWarning, module=r"gymnasium.*")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED0_DEFAULT = 11000
COLS = ["region", "ep", "pdr_planner", "pdr_base", "n_dec", "n_switch", "ms_per_dec", "sec"]


def _log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def planner_episode(fac, model, seed, K, h, m, leaf_fn, clairvoyant, reseed_base,
                    switch_margin=0.0):
    """한 에피소드: (1) 챔피언 greedy 베이스라인 → (2) 같은 시드에서 플래너 에피소드.
    (1)(2) 순서·수식은 rollout_oracle.lookahead_episode 와 동일(앵커 비트 동일성).
    반환 (pdr_base, pdr_planner, n_dec, n_switch, ms_per_dec)."""
    from planner_policy import TruncatedRolloutPlanner
    env = fac(seed=seed)

    # ---- (1) baseline: 순수 greedy (paired 기준 — 재계산 아님, 같은 CRN 재주행) ----
    obs, _ = env.reset(seed=seed)
    done, w_base = False, 0.0
    while not done:
        mask = env.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _r, term, trunc, info = env.step(int(a))
        w_base += info.get("r_woG", 0.0)
        done = term or trunc
    prev = env.unwrapped.preventable_woG
    pdr_base = 1.0 - w_base / prev if prev > 0 else 0.0

    # ---- (2) planner: 매 결정 TruncatedRolloutPlanner.act ----
    planner = TruncatedRolloutPlanner(model, K=K, h=h, m=m, leaf_fn=leaf_fn,
                                      clairvoyant=clairvoyant, reseed_base=reseed_base,
                                      switch_margin=switch_margin)
    obs, _ = env.reset(seed=seed)
    prev_p = env.unwrapped.preventable_woG
    done, w = False, 0.0
    n_dec = n_switch = 0
    ms_list = []
    while not done:
        a = planner.act(env, ep_seed=seed, obs=obs)
        li = planner.last_info
        if li["lookahead"]:
            n_dec += 1
            ms_list.append(li["ms"])
            if li["switched"]:
                n_switch += 1
        obs, _r, term, trunc, info = env.step(int(a))
        w += info.get("r_woG", 0.0)
        done = term or trunc
    pdr_planner = 1.0 - w / prev_p if prev_p > 0 else 0.0
    ms_per_dec = float(np.mean(ms_list)) if ms_list else 0.0
    return pdr_base, pdr_planner, n_dec, n_switch, ms_per_dec


# ---------------------------------------------------------------- 병렬 워커
def worker(job):
    """(region, cfg, model_dir, seed0, ep 리스트, K, h, m, leaf_path, clairvoyant,
    reseed_base) → per-ep 행 목록."""
    (region, cfg, model_dir, seed0, eps, K, h, m,
     leaf_path, clairvoyant, reseed_base, switch_margin) = job
    from rollout_oracle import _set_env_vars
    _set_env_vars()                                  # essential+load · occ (env 빌드 전)
    import torch as th
    th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    try:
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        leaf_fn = None
        if leaf_path:
            from leaf_value import load_leaf
            leaf_fn = load_leaf(leaf_path, device="cpu")
        rows = []
        with _suppress_stdout():
            fac = make_feature_env(cfg, norm)
            for ep in eps:
                t0 = time.time()
                pdr_b, pdr_p, nd, ns, mspd = planner_episode(
                    fac, model, seed0 + ep, K, h, m, leaf_fn, clairvoyant, reseed_base,
                                switch_margin=switch_margin)
                rows.append({"region": region, "ep": ep,
                             "pdr_planner": pdr_p, "pdr_base": pdr_b,
                             "n_dec": nd, "n_switch": ns,
                             "ms_per_dec": round(mspd, 1),
                             "sec": round(time.time() - t0, 2)})
        return {"ok": True, "region": region, "rows": rows}
    except Exception as e:
        import traceback
        return {"ok": False, "region": region, "err": (str(e) + traceback.format_exc())[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(REPO, "results/rl/redesign/v4_plr2_s0"),
                    help="챔피언 디렉터리(final_model.zip + vecnormalize.pkl)")
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--regions", default="", help="쉼표구분 매니페스트 키 서브셋(생략시 전체)")
    ap.add_argument("--tune_pool40", action="store_true",
                    help="시군구 매니페스트 sigcd 균등 40지역 튜닝풀 사용(시도17 판정과 분리)")
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--K", type=int, default=8, help="masked-prob 상위 후보 수")
    ap.add_argument("--h", type=int, default=10, help="롤아웃 결정 지평(h<0=무한=종단까지)")
    ap.add_argument("--m", type=int, default=2, help="비천리안 몬테카를로 롤아웃 수")
    ap.add_argument("--leaf", choices=["none", "reg"], default="none",
                    help="절단 리프 부트스트랩: none=0, reg=leaf_value 회귀망")
    ap.add_argument("--leaf_path", default=os.path.join(REPO, "results/rl/redesign/leaf_value.pt"))
    ap.add_argument("--clairvoyant", action="store_true",
                    help="재시드 생략(=기존 오라클 천리안 — 앵커/격차분해용)")
    ap.add_argument("--reseed_base", type=int, default=777000)
    ap.add_argument("--switch_margin", type=float, default=0.0,
                    help="스위치 마진 ε(pdrwog 단위) — 평균 개선>ε×preventable 일 때만 이탈")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=5, help="잡당 에피소드 수(부하 균형)")
    ap.add_argument("--seed0", type=int, default=SEED0_DEFAULT)
    # (v6) 패딩 env 제어: 미지정=환경변수 상속(구 동작). 지정 시 main 에서 os.environ 설정 →
    # Pool 워커가 fork 로 상속(단 worker 의 _set_env_vars 가 MCI_OBS_VARIANT 를 essential+load
    # 로 고정하므로 obs_variant 는 essential+load 계열 챔피언에만 유효; MCI_H_PAD 는 무간섭 보존).
    ap.add_argument("--obs_variant", default=None,
                    help="MCI_OBS_VARIANT 명시(기본 None=상속). 자연-H 챔피언 판정=essential+load")
    ap.add_argument("--h_pad", default=None,
                    help="MCI_H_PAD 명시(기본 None=상속). 자연-H 시나리오를 고정 레이아웃(예 47)으로 패딩")
    ap.add_argument("--validate", action="store_true", help="deepcopy 결정론 검증만 하고 종료")
    ap.add_argument("--validate_steps", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/planner_eval.csv"))
    A = ap.parse_args()

    # (v6) obs variant/H_pad 를 Pool fork 전에 설정 → 워커 상속. 미지정 시 미설정(구 동작 불변).
    if A.obs_variant:
        os.environ["MCI_OBS_VARIANT"] = A.obs_variant
    if A.h_pad:
        os.environ["MCI_H_PAD"] = str(A.h_pad)

    # ---- 대상 지역: 시도17 판정(기본) 또는 시군구40 튜닝풀 ----
    if A.tune_pool40:
        from score_cma import select_tune_regions
        sig = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json")
        pairs = select_tune_regions(sig, 40)
        if A.regions:
            want = set(A.regions.split(","))
            pairs = [(k, c) for k, c in pairs if k in want]
    else:
        manifest = json.load(open(A.manifest, encoding="utf-8"))
        keys = [k for k in A.regions.split(",") if k in manifest] if A.regions else list(manifest.keys())
        pairs = [(k, manifest[k]) for k in keys]
    if not pairs:
        raise SystemExit("대상 지역 0개 — --regions/--tune_pool40 확인")

    leaf_path = A.leaf_path if A.leaf == "reg" else ""
    if leaf_path and not os.path.exists(leaf_path):
        raise SystemExit(f"--leaf reg 인데 리프 모델 없음: {leaf_path}")

    # ---- 검증 모드(오라클 validate_clone 재사용) ----
    if A.validate:
        from rollout_oracle import validate_clone
        all_ok = True
        for k, cfg in pairs:
            for ep in range(A.n_eps):
                r = validate_clone(cfg, A.seed0 + ep, A.validate_steps, A.model_dir)
                print(f"[validate] {k} seed={A.seed0+ep}: ok={r['ok']} n_cmp={r['n_cmp']} "
                      f"deepcopy={r.get('deepcopy_sec', float('nan')):.3f}s {r['note']}", flush=True)
                all_ok &= r["ok"]
        print(f"[validate] 종합: {'PASS' if all_ok else 'FAIL — deepcopy 결정론 깨짐'}", flush=True)
        return

    # ---- 재개: 기존 CSV 의 (region, ep) 스킵 ----
    done = set()
    if os.path.exists(A.out):
        with open(A.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["region"], int(r["ep"])))
        _log(f"[planner] 재개 — 기존 {len(done)}행 스킵")

    jobs = []
    for k, cfg in pairs:
        todo = [ep for ep in range(A.n_eps) if (k, ep) not in done]
        for i in range(0, len(todo), A.chunk):
            jobs.append((k, cfg, A.model_dir, A.seed0, todo[i:i + A.chunk],
                         A.K, A.h, A.m, leaf_path, A.clairvoyant, A.reseed_base,
                         A.switch_margin))
    _log(f"[planner] regions={len(pairs)} n_eps={A.n_eps} K={A.K} h={A.h} m={A.m} "
         f"leaf={A.leaf} clairvoyant={A.clairvoyant} margin={A.switch_margin} jobs={len(jobs)} "
         f"variant={A.obs_variant or '(상속)'} h_pad={A.h_pad or '(상속)'} "
         f"workers={A.workers} out={A.out}")
    if not jobs:
        _log("[planner] 할 일 없음(전부 완료)")
        return

    new_file = not os.path.exists(A.out)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)) or ".", exist_ok=True)
    fout = open(A.out, "a", newline="", encoding="utf-8")
    wcsv = csv.DictWriter(fout, fieldnames=COLS)
    if new_file:
        wcsv.writeheader()
        fout.flush()

    t0, n_rows, n_fail = time.time(), 0, 0
    with Pool(min(A.workers, len(jobs)), maxtasksperchild=1) as pool:
        for j, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                for row in r["rows"]:
                    wcsv.writerow(row)
                    n_rows += 1
                fout.flush()
                d = np.mean([row["pdr_base"] - row["pdr_planner"] for row in r["rows"]])
                s = np.mean([row["sec"] for row in r["rows"]])
                ms = np.mean([row["ms_per_dec"] for row in r["rows"]])
                _log(f"  [{j}/{len(jobs)}] {r['region']} +{len(r['rows'])}ep Δ={d:+.4f} "
                     f"{s:.0f}s/ep {ms:.0f}ms/dec (누적 {n_rows}행, {time.time()-t0:.0f}s)")
            else:
                n_fail += 1
                _log(f"  [{j}/{len(jobs)}] FAIL {r['region']}: {r['err'][:200]}")
    fout.close()
    _log(f"[planner] 완료 rows={n_rows} fail_jobs={n_fail} wall={time.time()-t0:.0f}s → {A.out}")

    # 요약(Δ = pdr_base − pdr_planner, 양수 = 플래너가 PDR 낮춤 = 개선)
    per, ms_all = {}, []
    with open(A.out, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per.setdefault(r["region"], []).append(float(r["pdr_base"]) - float(r["pdr_planner"]))
            ms_all.append(float(r["ms_per_dec"]))
    ds = [d for v in per.values() for d in v]
    _log(f"[planner] Δ 전체 mean={np.mean(ds):+.4f} ms/dec 평균={np.mean(ms_all):.0f} "
         f"(지역 {len(per)}, 에피소드 {len(ds)})")
    for k in sorted(per, key=lambda x: -np.mean(per[x])):
        _log(f"    {k}: Δ={np.mean(per[k]):+.4f} (n={len(per[k])})")


if __name__ == "__main__":
    main()
