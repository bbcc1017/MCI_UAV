"""L사다리 paired 평가 — RL 4런(L0~L3) + 규칙 3종(적응T-LB·LB-T4·휴리best)을 같은 시드에서 비교.

플랜 v2 Phase 1 판정 하네스. 같은 (region, seed) 실현에서 7개 정책을 각자 롤아웃 →
per-episode woG·PDR_woG 배열 → 사다리 기여(L1−L0, L2−L1, L3−L2)와 baseline 대비
승/무/패·평균차·95%CI. 지표는 PDR_woG(규모 불변) 주, woG 보조.

핵심 주의:
  - 모델별 obs variant/vecnorm 상이: L0/L1=essential(209), L2/L3=essential+load(355).
    env 는 정책마다 자기 variant+자기 vecnorm 으로 빌드(빌드 전 MCI_OBS_VARIANT 설정).
    dynamics 는 obs/norm 과 무관 → reset(seed=s) 실현 동일 → paired 성립.
  - 규칙 정책은 obs 비의존(en_manager·get_static_eta) → norm 없는 essential env 로 실행.
  - occ 게이트 고정(플랜 탐구단계). MCI_CAP_GATE 는 step 시점에 읽히므로 워커 전역 설정.
  - --use_ckpt: 최신 checkpoint + norm=None 로 배관 스모크(정규화 없어 성능 무의미, 형상만 검증).

예(정식): PYTHONIOENCODING=utf-8 python src/rl_src/paired_eval_ladder.py --n_eps 1000 --workers 17
예(스모크): ... --use_ckpt --n_eps 3 --regions 서울,강원
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import hashlib
import json
import sys
import time
import subprocess
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_REWARD_MODE", "woG")  # eval 은 info['r_woG'] 를 직접 읽음(모드 무관)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SEED = 11000
SIDO17 = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()

# 모델 → obs variant (학습 시와 동일해야 로드/forward 정합) — 레거시 단축명용
MODEL_VARIANT = {
    "L0_base": "essential", "L1_hygiene": "essential",
    "L2_loadobs": "essential+load", "L3_pointer": "essential+load",
}


def parse_model_specs(spec: str, model_root: str):
    """--models 파싱 → [(name, mdir, variant, algo)].

    항목 2형식(쉼표구분 혼용 가능):
      * 레거시 단축명 "L3_pointer"        → (model_root/L3_pointer_s0, MODEL_VARIANT 참조)
      * 일반형 "이름=디렉터리=obs_variant[=algo]" → 임의 모델 디렉터리(상대경로는 REPO 기준).
        algo(v5 zoo): ppo|dqn|qrdqn|sacd|reinforce — 4번째 토큰 명시가 우선, 생략 시
        mdir/meta.json(train_zoo 저장 관례)의 "algo" 자동 감지, 둘 다 없으면 ppo.
    기본값(L0~L3)은 전부 레거시 형식 → 기존 동작 불변(algo="ppo").
    """
    entries = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            parts = tok.split("=")
            if len(parts) not in (3, 4):
                raise ValueError(f"--models 항목 형식 오류(이름=디렉터리=obs_variant[=algo]): {tok!r}")
            name, mdir, variant = parts[:3]
            algo = parts[3] if len(parts) == 4 else None
            if not os.path.isabs(mdir):
                mdir = os.path.join(REPO, mdir)
        else:
            if tok not in MODEL_VARIANT:
                raise ValueError(f"미지 단축명 {tok!r} — 일반형 '이름=디렉터리=obs_variant[=algo]' 사용")
            name, mdir, variant = tok, os.path.join(model_root, f"{tok}_s0"), MODEL_VARIANT[tok]
            algo = None
        if algo is None:  # meta.json(v5 zoo 저장 관례) 자동 감지 — 없으면 기존 기본 ppo
            meta_p = os.path.join(mdir, "meta.json")
            if os.path.exists(meta_p):
                try:
                    with open(meta_p, encoding="utf-8") as f:
                        algo = json.load(f).get("algo", "ppo")
                except Exception:
                    algo = "ppo"
            else:
                algo = "ppo"
        entries.append((name, mdir, variant, algo))
    return entries


def _load_policy(algo: str, path: str):
    """algo별 모델 로드 → policy_fn(obs, mask, unwrapped)->int (v5 zoo 로더 디스패치).

    ppo=기존 경로 그대로(MaskablePPO.load + ppo_policy — pointer/deepsets 역직렬화 import 는
    worker 상단이 담당). 나머지는 지연 import(파일 부재 시 ppo 경로 무영향) 후
    공통 계약 `predict_masked` 를 evaluate.masked_model_policy 로 래핑."""
    if algo == "ppo":
        from sb3_contrib import MaskablePPO
        from evaluate import ppo_policy
        return ppo_policy(MaskablePPO.load(path, device="cpu"))
    from evaluate import masked_model_policy
    if algo == "dqn":
        from masked_dqn import MaskedDQN
        return masked_model_policy(MaskedDQN.load(path, device="cpu"))
    if algo == "qrdqn":
        from masked_qrdqn import MaskedQRDQN
        return masked_model_policy(MaskedQRDQN.load(path, device="cpu"))
    if algo == "sacd":
        from masked_sac_discrete import SACDiscrete
        return masked_model_policy(SACDiscrete.load(path, device="cpu"))
    if algo == "reinforce":
        from reinforce_vec import ReinforceVec
        return masked_model_policy(ReinforceVec.load(path, device="cpu"))
    raise ValueError(f"미지 algo {algo!r} (ppo|dqn|qrdqn|sacd|reinforce)")


def _rollout_woG(factory, policy_fn, seed):
    """1 에피소드 롤아웃 → (woG 합, PDR_woG). factory(seed) 는 캐시된 env 재사용."""
    env = factory(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    w = 0.0
    while not done:
        mask = env.action_masks()
        a = policy_fn(obs, mask, env.unwrapped)
        obs, r, term, trunc, info = env.step(a)
        w += info.get("r_woG", 0.0)
        done = term or trunc
    prev = env.unwrapped.preventable_woG
    pdr = 1.0 - w / prev if prev > 0 else 0.0
    return w, pdr


def worker(job):
    region, cfg, best_rule, model_entries, baselines, n_eps, seed, use_ckpt, env_variant = job
    import numpy as np
    import torch as th
    th.set_num_threads(1)
    os.environ["MCI_CAP_GATE"] = "occ"  # 탐구단계 고정
    from sb3_contrib import MaskablePPO
    from hospital_set_extractor import HospitalSetExtractor  # noqa: F401 (deepsets 역직렬화)
    from pointer_policy import HospitalTokenExtractor, PointerMaskablePolicy  # noqa: F401
    import pad_vecnorm  # noqa: F401 (v6 valid: PadAwareVecNormalize pickle 해석용 — pointer 전례)
    from viper_distill import make_feature_env, load_vecnorm, _suppress_stdout
    from evaluate import ppo_policy
    from distill_policy import make_heuristic_policy
    from loadbalance_heuristic import make_cap_policy, make_adaptive_cap_policy

    def build_factory(variant, norm):
        os.environ["MCI_OBS_VARIANT"] = variant
        # (v6) valid variant 는 병원 패딩 레이아웃 필수 — HospitalFeatureWrapper 가
        # "valid 인데 MCI_H_PAD 없음"이면 에러. 미설정 시만 47(고정 fixed_hos_num=H_DEFAULT)
        # 로 설정하고 외부 지정은 존중. 이후 비-valid 빌드(규칙 essential 등)도 이 패딩을 타
        # 므로 make_cap_policy 의 H_DEFAULT=47 코덱과 자연 정합(자연-H 판정 시 규칙 무크래시).
        if "valid" in variant:
            os.environ.setdefault("MCI_H_PAD", "47")
        fac = make_feature_env(cfg, norm)
        fac(seed=seed)  # 강제 빌드(현재 variant 로 캐시 고정) — 이후 env var 바뀌어도 무관
        return fac

    try:
        with _suppress_stdout():
            # ---- 정책별 (factory, policy_fn) 구성 ----
            entries = []  # (name, factory, policy_fn)
            for m, mdir, variant, algo in model_entries:
                ext = ".pt" if algo == "reinforce" else ".zip"  # reinforce=torch dict, 그외 SB3 zip
                if use_ckpt:
                    cks = sorted([f for f in os.listdir(os.path.join(mdir, "checkpoints"))
                                  if f.endswith(ext)],
                                 key=lambda f: int(f.split("_")[-2]))
                    if not cks:
                        continue
                    zip_path = os.path.join(mdir, "checkpoints", cks[-1])
                    norm = None  # 체크포인트엔 vecnorm 없음 → 스모크(형상만)
                else:
                    zip_path = os.path.join(mdir, f"final_model{ext}")
                    if not os.path.exists(zip_path):
                        continue
                    vn = os.path.join(mdir, "vecnormalize.pkl")
                    norm = load_vecnorm(vn) if os.path.exists(vn) else None
                pol = _load_policy(algo, zip_path)  # v5 zoo 디스패치(ppo=기존 경로 동일)
                fac = build_factory(variant, norm)
                entries.append((m, fac, pol))

            # 규칙 정책은 obs 비의존. v10은 RL과 같은 valid/pad 배관을 명시해 환경 설정도 통일한다.
            rule_fac = build_factory(env_variant, None)
            if "heur" in baselines:
                entries.append(("heur", rule_fac, make_heuristic_policy(best_rule)))
            if "lb_T4" in baselines:
                entries.append(("lb_T4", rule_fac, make_cap_policy(best_rule, 4)))
            if "lb_adaptT" in baselines:
                entries.append(("lb_adaptT", rule_fac, make_adaptive_cap_policy(best_rule)))

            names = [e[0] for e in entries]
            W = {n: np.zeros(n_eps) for n in names}
            P = {n: np.zeros(n_eps) for n in names}
            for ep in range(n_eps):
                s = seed + ep
                for name, fac, pol in entries:
                    w, pdr = _rollout_woG(fac, pol, s)
                    W[name][ep] = w
                    P[name][ep] = pdr

        out = {"region": region, "n_eps": n_eps, "ok": True, "names": names}
        for n in names:
            out[f"woG_{n}"] = float(W[n].mean())
            out[f"PDR_{n}"] = float(P[n].mean())
        # paired 배열 보존(집계용) — PDR_woG 기준 승/무/패는 main 에서
        out["_P"] = {n: P[n].tolist() for n in names}
        out["_W"] = {n: W[n].tolist() for n in names}
        return out
    except Exception as e:
        import traceback
        return {"region": region, "ok": False, "err": (str(e) + traceback.format_exc())[:400]}


def _paired(a, b):
    """a,b: per-ep PDR_woG 배열. PDR 은 낮을수록 좋음 → 개선 = b−a(a가 모델). 반환 (mean_impr, ci, sig)."""
    import numpy as np
    d = np.asarray(b) - np.asarray(a)  # >0 = a(모델)가 baseline b 보다 PDR 낮음(우수)
    md = float(d.mean())
    n = len(d)
    ci = 1.96 * float(np.std(d, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    sig = "win" if md > ci else "loss" if md < -ci else "tie"
    return md, ci, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json"))
    ap.add_argument("--heur_csv", default=os.path.join(REPO, "results/sido_osrm_heuristic_best.csv"))
    ap.add_argument("--regions", default="", help="쉼표구분 키 서브셋(기본 시도17 전체 또는 매니페스트 전 키)")
    ap.add_argument("--match", choices=["name", "sigcd"], default="name",
                    help="best_rule 매칭: name(시도) / sigcd(시군구·holdout, 키의 숫자토큰→best CSV sigcd)")
    ap.add_argument("--key_filter", default="", help="매니페스트 키 부분문자열 필터(예: _p0 = 시군구당 1점)")
    ap.add_argument("--model_root", default=os.path.join(REPO, "results/rl/redesign"))
    ap.add_argument("--models", default="L0_base,L1_hygiene,L2_loadobs,L3_pointer",
                    help="쉼표구분. 레거시 단축명(L0_base 등, model_root/<명>_s0) 또는 "
                         "일반형 '이름=디렉터리=obs_variant[=algo]'(신규 모델 평가용) 혼용 가능. "
                         "algo 생략 시 meta.json 자동 감지→ppo(v5 zoo: dqn|qrdqn|sacd|reinforce).")
    ap.add_argument("--n_eps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--baselines", default="heur,lb_T4,lb_adaptT",
                    help="쉼표구분: heur,lb_T4,lb_adaptT. baseline-only는 --models '' 사용")
    ap.add_argument("--env_variant", default="essential",
                    help="규칙 평가 env obs 배관. v10은 essential+load+valid")
    ap.add_argument("--dataset_role", choices=["generic", "train1000", "eval250"], default="generic",
                    help="v10 데이터셋 구조·개수 엄격 검증")
    ap.add_argument("--strict", action="store_true",
                    help="누락 규칙·실패 job·출력 수 불일치 시 즉시 실패")
    ap.add_argument("--use_ckpt", action="store_true", help="스모크: 최신 ckpt+norm없음")
    ap.add_argument("--out", default=os.path.join(REPO, "results/rl/redesign/paired_ladder.csv"))
    ap.add_argument("--dump_pe", default="", help="per-episode PDR/woG NPZ 저장(공정 paired 재분석용)")
    ap.add_argument("--meta_out", default="", help="평가 provenance JSON(기본 <out>.meta.json)")
    A = ap.parse_args()

    import numpy as np  # noqa
    manifest = json.load(open(A.manifest, encoding="utf-8"))
    model_entries = parse_model_specs(A.models, A.model_root)
    models = [e[0] for e in model_entries]  # 이하 요약/사다리 로직은 이름 기준(기존 유지)
    baselines_requested = [x.strip() for x in A.baselines.split(",") if x.strip()]
    unknown = set(baselines_requested) - {"heur", "lb_T4", "lb_adaptT"}
    if unknown:
        raise ValueError(f"미지 baseline: {sorted(unknown)}")

    if A.dataset_role == "train1000":
        import re
        pat = re.compile(r"^.+_(\d{5})_p([0-3])$")
        groups = {}
        for key in manifest:
            match = pat.match(key)
            if not match:
                raise ValueError(f"train1000 키 형식 오류: {key!r}")
            groups.setdefault(match.group(1), set()).add(int(match.group(2)))
        bad = {k: sorted(v) for k, v in groups.items() if v != {0, 1, 2, 3}}
        if len(manifest) != 1000 or len(groups) != 250 or bad:
            raise ValueError(f"train1000 구조 오류: N={len(manifest)} groups={len(groups)} bad={list(bad.items())[:5]}")
    elif A.dataset_role == "eval250":
        if len(manifest) != 250 or any(k.endswith(("_p0", "_p1", "_p2", "_p3")) for k in manifest):
            raise ValueError(f"eval250 구조 오류: N={len(manifest)}")

    # 휴리 best_rule 룩업 (BOM 대응). match=name→region 키, sigcd→sigcd 키.
    best_by = {}
    heuristic_rows = []
    with open(A.heur_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lookup_key = r["region"] if A.match == "name" else r["sigcd"]
            if lookup_key in best_by and A.strict:
                raise ValueError(f"heur_csv 중복 키: {lookup_key}")
            best_by[lookup_key] = r["best_rule"]
            heuristic_rows.append(r)
    if A.strict and A.dataset_role in {"train1000", "eval250"}:
        if len(best_by) != 250:
            raise ValueError(f"v10 heur_csv는 250개 시군구 규칙이어야 함: {len(best_by)}")
        if not heuristic_rows or "selection_manifest" not in heuristic_rows[0]:
            raise ValueError("v10 strict 평가에는 학습 1,000좌표에서 적합한 selection_manifest 열이 필요함")
        expected_fit_manifest = os.path.realpath(os.path.join(
            REPO, "scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json"
        ))
        fit_sources = {os.path.realpath(r["selection_manifest"]) for r in heuristic_rows}
        if fit_sources != {expected_fit_manifest}:
            raise ValueError(f"휴리스틱 규칙 선택 데이터가 v10 train1000이 아님: {sorted(fit_sources)}")

    def _lookup(key):
        if A.match == "name":
            return best_by.get(key)
        digits = [t for t in key.split("_") if t.isdigit()]  # 키의 sigcd 토큰
        return best_by.get(digits[0]) if digits else None

    # 대상 키 목록
    if A.regions:
        keys = [k for k in A.regions.split(",") if k in manifest]
    elif A.match == "name":
        keys = [k for k in SIDO17 if k in manifest]
    else:
        keys = list(manifest.keys())
    if A.key_filter:
        keys = [k for k in keys if A.key_filter in k]

    missing_rules = [k for k in keys if _lookup(k) is None]
    if missing_rules and A.strict:
        raise ValueError(f"휴리스틱 규칙 누락 {len(missing_rules)}개: {missing_rules[:5]}")
    jobs = [(k, manifest[k], _lookup(k), model_entries, baselines_requested, A.n_eps,
             A.seed, A.use_ckpt, A.env_variant)
            for k in keys if _lookup(k) is not None]
    print(f"[paired] jobs={len(jobs)} match={A.match} filter={A.key_filter!r} models={models} "
          f"baselines={baselines_requested} n_eps={A.n_eps} seed={A.seed} "
          f"use_ckpt={A.use_ckpt} workers={A.workers}", flush=True)

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
    failed = [r for r in res if not r["ok"]]
    if A.strict and (failed or len(ok) != len(jobs)):
        raise RuntimeError(f"평가 실패: failed={len(failed)}, ok={len(ok)}, jobs={len(jobs)}")
    if not ok:
        print("전부 실패", flush=True); return
    order = {key: i for i, key in enumerate(keys)}
    ok.sort(key=lambda r: order[r["region"]])
    names = ok[0]["names"]
    if A.strict:
        for r in ok:
            if r["names"] != names:
                raise RuntimeError(f"정책 목록 불일치: {r['region']} {r['names']} != {names}")
            for name in names:
                arr = np.asarray(r["_P"][name], dtype=float)
                if arr.shape != (A.n_eps,) or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
                    raise RuntimeError(f"PDR 원자료 오류: {r['region']} {name}")
    rl_models = [m for m in models if f"PDR_{m}" in ok[0]]
    baselines = [b for b in ("heur", "lb_T4", "lb_adaptT") if b in names]

    # 절대 PDR_woG (낮을수록 좋음) 저장
    with open(A.out, "w", newline="", encoding="utf-8") as f:
        cols = ["region", "n_eps"] + [f"PDR_{n}" for n in names] + [f"woG_{n}" for n in names]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ok:
            w.writerow({c: r.get(c) for c in cols})
    print(f"\n저장 {A.out}  wall={time.time()-t0:.0f}s", flush=True)

    # per-episode 원자료: 별도 실행한 정책도 같은 manifest/seed/n_eps이면 정확한 paired 결합 가능.
    if A.dump_pe:
        pdr = np.asarray([[r["_P"][name] for name in names] for r in ok], dtype=np.float64)
        wog = np.asarray([[r["_W"][name] for name in names] for r in ok], dtype=np.float64)
        np.savez_compressed(
            A.dump_pe,
            regions=np.asarray([r["region"] for r in ok]),
            names=np.asarray(names),
            seeds=np.arange(A.seed, A.seed + A.n_eps, dtype=np.int64),
            pdr=pdr,
            wog=wog,
        )
        print(f"저장(per-ep) {A.dump_pe} shape={pdr.shape}", flush=True)

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _scenario_bundle_sha256(data):
        h = hashlib.sha256()
        for key in sorted(data):
            path = os.path.realpath(data[key])
            h.update(key.encode("utf-8"))
            h.update(b"\0")
            h.update(path.encode("utf-8"))
            h.update(b"\0")
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            h.update(b"\0")
        return h.hexdigest()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = "unknown"
    meta_path = A.meta_out or A.out + ".meta.json"
    meta = {
        "protocol": "same-seed-paired-pdrwog",
        "manifest": os.path.abspath(A.manifest),
        "manifest_sha256": _sha256(A.manifest),
        "scenario_bundle_sha256": _scenario_bundle_sha256({k: manifest[k] for k in keys}),
        "heur_csv": os.path.abspath(A.heur_csv),
        "heur_csv_sha256": _sha256(A.heur_csv),
        "dataset_role": A.dataset_role,
        "seed": A.seed,
        "n_eps": A.n_eps,
        "n_regions": len(ok),
        "models": [{"name": n, "dir": d, "variant": v, "algo": a}
                   for n, d, v, a in model_entries],
        "baselines": baselines_requested,
        "environment": {
            "MCI_CAP_GATE": "occ",
            "MCI_OBS_VARIANT": A.env_variant,
            "MCI_H_PAD": os.environ.get("MCI_H_PAD", "47" if "valid" in A.env_variant else ""),
            "MCI_REWARD_MODE": "woG",
        },
        "metric": "PDR_woG",
        "lower_is_better": True,
        "git_sha": git_sha,
        "outputs": {"summary_csv": os.path.abspath(A.out),
                    "per_episode_npz": os.path.abspath(A.dump_pe) if A.dump_pe else None},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"저장(meta) {meta_path}", flush=True)

    # paired 요약: 각 RL vs 각 baseline (PDR_woG 개선 = baseline−RL, 승=RL이 유의 낮음)
    print("\n=== paired PDR_woG (양수=RL 우수, 승/무/패 across 지역) ===", flush=True)
    for m in rl_models:
        line = f"[{m}]"
        for b in baselines:
            diffs = [_paired(r["_P"][m], r["_P"][b]) for r in ok]
            md = np.mean([d[0] for d in diffs])
            win = sum(d[2] == "win" for d in diffs); tie = sum(d[2] == "tie" for d in diffs)
            loss = sum(d[2] == "loss" for d in diffs)
            line += f"  vs {b}: {md:+.4f} ({win}/{tie}/{loss})"
        print(line, flush=True)

    # 임의 모델 일반형의 첫 항목을 control로 보고 나머지 RL과 직접 paired 비교. 기존 L사다리는
    # 아래 전용 블록을 그대로 유지하고, v9처럼 이름이 자유로운 아키텍처 실험도 에피소드 배열
    # 95%CI 기준 W/T/L을 잃지 않도록 한다. 양수 = 비교 모델(m)이 첫 모델(base)보다 PDR 낮음.
    if len(rl_models) >= 2:
        base = rl_models[0]
        print(f"\n=== RL 직접 비교 (양수=비교 모델이 {base}보다 우수) ===", flush=True)
        for m in rl_models[1:]:
            diffs = [_paired(r["_P"][m], r["_P"][base]) for r in ok]
            md = np.mean([d[0] for d in diffs])
            win = sum(d[2] == "win" for d in diffs)
            tie = sum(d[2] == "tie" for d in diffs)
            loss = sum(d[2] == "loss" for d in diffs)
            print(f"  {m} vs {base}: {md:+.4f} ({win}/{tie}/{loss})", flush=True)
    # 사다리 기여 (인접 단계 PDR 개선)
    print("\n=== 사다리 기여 (양수=상위단계가 PDR 낮춤) ===", flush=True)
    order = [m for m in ("L0_base", "L1_hygiene", "L2_loadobs", "L3_pointer") if m in rl_models]
    for i in range(1, len(order)):
        diffs = [_paired(r["_P"][order[i]], r["_P"][order[i-1]]) for r in ok]
        md = np.mean([d[0] for d in diffs])
        win = sum(d[2] == "win" for d in diffs); loss = sum(d[2] == "loss" for d in diffs)
        print(f"  {order[i]} − {order[i-1]}: {md:+.4f}  승{win}/패{loss}", flush=True)


if __name__ == "__main__":
    main()
