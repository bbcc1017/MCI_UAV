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


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_meta(A, pairs, milp_kw):
    """(v11) 산출 CSV 옆에 재현 메타(.meta.json) — 입력 해시·env·시드·하이퍼·git_sha."""
    import subprocess
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    inputs = {}
    for p in (A.manifest,
              os.path.join(A.model_dir, "final_model.zip"),
              os.path.join(A.model_dir, "vecnormalize.pkl"),
              os.path.join(REPO, "src/rl_src/planner_policy.py"),
              os.path.join(REPO, "src/rl_src/planner_eval.py"),
              os.path.join(REPO, "src/rl_src/milp_policy.py")):
        if p and os.path.exists(p):
            inputs[os.path.relpath(p, REPO)] = {"sha256": _sha256(p),
                                                "bytes": os.path.getsize(p)}
    meta = {
        "schema_version": 1, "tag": A.tag, "policy": A.policy,
        "protocol": "v10_random4_train__representative250_eval",
        "metric": "PDR_woG", "lower_is_better": True,
        "model_dir": A.model_dir, "manifest": A.manifest,
        "regions": [k for k, _ in pairs], "n_regions": len(pairs),
        "seed0": A.seed0, "n_eps": A.n_eps,
        "seeds": [A.seed0, A.seed0 + A.n_eps - 1],
        "planner": {"K": A.K, "h": A.h, "m": A.m, "leaf": A.leaf,
                    "clairvoyant": bool(A.clairvoyant), "reseed_base": A.reseed_base,
                    "switch_margin": A.switch_margin, "alloc": A.alloc,
                    "switch_z": A.switch_z, "cand_source": A.cand_source},
        "milp": milp_kw if (A.policy == "milp" or A.cand_source == "ppo+milp") else None,
        "environment": {k: os.environ.get(k) for k in
                        ("MCI_CAP_GATE", "MCI_OBS_VARIANT", "MCI_H_PAD", "MCI_REWARD_MODE")},
        "inputs": inputs, "git_sha": sha, "output": A.out,
    }
    with open(A.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def planner_episode(fac, model, seed, K, h, m, leaf_fn, clairvoyant, reseed_base,
                    switch_margin=0.0, alloc="uniform", switch_z=0.0,
                    policy="planner", milp_kw=None, cand_source="ppo"):
    """한 에피소드: (1) 챔피언 greedy 베이스라인 → (2) 같은 시드에서 플래너 에피소드.
    (1)(2) 순서·수식은 rollout_oracle.lookahead_episode 와 동일(앵커 비트 동일성).
    반환 (pdr_base, pdr_planner, n_dec, n_switch, ms_per_dec).

    (v11) policy="milp" 면 (2)를 MILP 단독 정책으로 대체한다(같은 CRN·같은 base 라
    NCRP 팔들과 paired 비교가 그대로 성립). cand_source="ppo+milp" 면 NCRP 후보집합에
    MILP 액션을 주입한다."""
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
    milp_kw = dict(milp_kw or {})
    h_pad = int(os.environ.get("MCI_H_PAD", "47"))
    if policy == "milp":
        from milp_policy import MilpPlanner
        planner = MilpPlanner(model=model, h_pad=h_pad, **milp_kw)
    else:
        extra = None
        if cand_source == "ppo+milp":
            from milp_policy import MilpProposer
            prop = MilpProposer(h_pad=h_pad, **milp_kw)
            extra = prop.propose
        planner = TruncatedRolloutPlanner(model, K=K, h=h, m=m, leaf_fn=leaf_fn,
                                          clairvoyant=clairvoyant, reseed_base=reseed_base,
                                          switch_margin=switch_margin, alloc=alloc,
                                          switch_z=switch_z, extra_cand_fn=extra)
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
    """job dict(region·cfg·model_dir·seed0·eps + 플래너 하이퍼) → per-ep 행 목록."""
    region, cfg, model_dir = job["region"], job["cfg"], job["model_dir"]
    seed0, eps = job["seed0"], job["eps"]
    K, h, m = job["K"], job["h"], job["m"]
    leaf_path, clairvoyant = job["leaf_path"], job["clairvoyant"]
    reseed_base, switch_margin = job["reseed_base"], job["switch_margin"]
    alloc, switch_z = job.get("alloc", "uniform"), job.get("switch_z", 0.0)
    policy, cand_source = job.get("policy", "planner"), job.get("cand_source", "ppo")
    milp_kw = job.get("milp_kw", {})
    from rollout_oracle import _set_env_vars
    # (v6) 상속 존중: main 의 --obs_variant/--h_pad 설정(fork 상속)을 워커가 덮어쓰지 않음.
    # 미설정 시 기본(essential+load·occ)은 setdefault 로 동일 유지.
    _set_env_vars(respect_existing=True)
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
                    switch_margin=switch_margin, alloc=alloc, switch_z=switch_z,
                    policy=policy, milp_kw=milp_kw, cand_source=cand_source)
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
    # ---- (v11) 조건탐색 축 ----
    ap.add_argument("--regions_file", default="",
                    help="지역키 목록 파일(줄바꿈/쉼표 구분) — dev40 등 고정 풀 지정용")
    ap.add_argument("--alloc", choices=["uniform", "sh"], default="uniform",
                    help="롤아웃 할당: uniform(기존) | sh(successive halving, 같은 예산)")
    ap.add_argument("--switch_z", type=float, default=0.0,
                    help="페어드 SE 의 z배를 스위치 마진으로(0=기존 엄격개선)")
    ap.add_argument("--policy", choices=["planner", "milp"], default="planner",
                    help="planner=NCRP 롤아웃 | milp=MILP 단독(OR 기준선)")
    ap.add_argument("--cand_source", choices=["ppo", "ppo+milp"], default="ppo",
                    help="NCRP 후보집합 원천(ppo+milp=MILP 액션 주입)")
    ap.add_argument("--milp_n_opp", type=int, default=3, help="MILP 병원별 지연 기회 수")
    ap.add_argument("--milp_topk", type=int, default=0, help="MILP 후보 병원 상한(0=전체)")
    ap.add_argument("--milp_second_wave", action="store_true", help="MILP 2차 왕복 슬롯+체인")
    ap.add_argument("--milp_future", action="store_true",
                    help="MILP 수요에 예상 구조환자 포함(정적 구조시간 분포만 사용)")
    ap.add_argument("--milp_future_groups", type=int, default=2, help="예상 구조환자 분위 그룹수")
    ap.add_argument("--milp_queue_model", choices=["fluid", "timed"], default="fluid",
                    help="MILP 큐 모형: fluid=이송중 머릿수 | timed=도착시각 인식")
    ap.add_argument("--milp_force_dispatch", action="store_true",
                    help="MILP 가 stay 를 택할 때 현장 차량 최대가치 배정을 강제")
    ap.add_argument("--milp_n_propose", type=int, default=2, help="주입 후보 수(cand_source=ppo+milp)")
    ap.add_argument("--tag", default="", help="meta.json 에 남길 실험 팔 이름")
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
        sig = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json")
        pairs = select_tune_regions(sig, 40)
        if A.regions:
            want = set(A.regions.split(","))
            pairs = [(k, c) for k, c in pairs if k in want]
    else:
        manifest = json.load(open(A.manifest, encoding="utf-8"))
        want = []
        if A.regions_file:
            # 주석(#)은 줄 단위로 먼저 제거한 뒤 쉼표 분리(주석 안 쉼표가 키로 새는 것 방지)
            for line in open(A.regions_file, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                want += [x.strip() for x in line.split(",") if x.strip()]
        elif A.regions:
            want = [x for x in A.regions.split(",") if x]
        if want:
            missing = [k for k in want if k not in manifest]
            if missing:
                raise SystemExit(f"매니페스트에 없는 지역키 {len(missing)}개: {missing[:5]}")
            keys = want
        else:
            keys = list(manifest.keys())
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

    milp_kw = {"n_opp": A.milp_n_opp, "topk_hosp": A.milp_topk,
               "second_wave": A.milp_second_wave, "future_patients": A.milp_future,
               "n_future_groups": A.milp_future_groups,
               "force_dispatch": A.milp_force_dispatch,
               "queue_model": A.milp_queue_model}
    if A.cand_source == "ppo+milp":
        milp_kw["n_propose"] = A.milp_n_propose
    jobs = []
    for k, cfg in pairs:
        todo = [ep for ep in range(A.n_eps) if (k, ep) not in done]
        for i in range(0, len(todo), A.chunk):
            jobs.append({"region": k, "cfg": cfg, "model_dir": A.model_dir,
                         "seed0": A.seed0, "eps": todo[i:i + A.chunk],
                         "K": A.K, "h": A.h, "m": A.m, "leaf_path": leaf_path,
                         "clairvoyant": A.clairvoyant, "reseed_base": A.reseed_base,
                         "switch_margin": A.switch_margin, "alloc": A.alloc,
                         "switch_z": A.switch_z, "policy": A.policy,
                         "cand_source": A.cand_source, "milp_kw": milp_kw})
    _log(f"[planner] tag={A.tag or '-'} policy={A.policy} cand={A.cand_source} "
         f"regions={len(pairs)} n_eps={A.n_eps} K={A.K} h={A.h} m={A.m} alloc={A.alloc} "
         f"z={A.switch_z} leaf={A.leaf} clairvoyant={A.clairvoyant} margin={A.switch_margin} "
         f"jobs={len(jobs)} variant={A.obs_variant or '(상속)'} h_pad={A.h_pad or '(상속)'} "
         f"workers={A.workers} out={A.out}")
    _write_meta(A, pairs, milp_kw)
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
