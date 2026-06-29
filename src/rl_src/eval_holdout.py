"""10M 모델 일반화 평가 — hold-out 좌표(학습과 분리)에서 결정론 정책 평가.

대상: (A) 전국 단일정책 @ eval_holdout_A(1000점) + (B) 시도17 모델 @ 각 시도 부분집합.
각 (모델,게이트,config)마다 결정론 정책 1000ep → woG reward + PDR_woG (eval_policy 재사용).
vecnorm 동결·게이트별 envvar(occ | psent[+CARED_OBS=0])·essential·green-mask·woG 학습과 일치.

병렬: 전역 작업목록(≈4000)을 Pool 로 분산. 모델은 워커당 1회 로드(model_dir 정렬+캐시).
출력: results/rl_holdout_eval.csv  (scope,region,sido,gate,key,R_woG,R,PDR_woG,PDR,n_eps)

예: PYTHONIOENCODING=utf-8 python src/rl_src/eval_holdout.py --workers 64 --n_eps 1000
"""
import os
# ★스레드 핀 필수(numpy/torch import 전) — 안 하면 워커당 전코어 스레드 폭주로 ~20배 느림.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, json, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MCI_OBS_VARIANT", "essential")
os.environ.setdefault("MCI_GREEN_MASK", "1")
os.environ.setdefault("MCI_REWARD_MODE", "woG")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
N_EPS = 1000
SEED_BASE = 7000
_CACHE = {}  # 워커 프로세스별 모델 캐시: model_dir -> (model, norm)


def _set_gate(gate):
    if gate == "occ":
        os.environ["MCI_CAP_GATE"] = "occ"
        os.environ["MCI_CARED_OBS"] = "1"
    else:  # psent = siteonly
        os.environ["MCI_CAP_GATE"] = "psent"
        os.environ["MCI_CARED_OBS"] = "0"


def _load(model_dir):
    if model_dir in _CACHE:
        return _CACHE[model_dir]
    import torch
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from viper_distill import load_vecnorm
    model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
    vn = os.path.join(model_dir, "vecnormalize.pkl")
    norm = load_vecnorm(vn) if os.path.exists(vn) else None
    _CACHE[model_dir] = (model, norm)
    return model, norm


def worker(job):
    scope, region, sido, gate, key, cfg, model_dir, n_eps = job
    _set_gate(gate)  # env build 전에 게이트 설정 (래퍼가 os.environ 읽음)
    from viper_distill import make_feature_env, _suppress_stdout
    from evaluate import eval_policy, ppo_policy
    try:
        model, norm = _load(model_dir)
        factory = make_feature_env(cfg, norm)
        with _suppress_stdout():
            m = eval_policy(factory, ppo_policy(model), n_eps, SEED_BASE)
        return dict(scope=scope, region=region, sido=sido, gate=gate, key=key,
                    R_woG=round(m["mean_R_woG"], 4), R=round(m["mean_R"], 4),
                    PDR_woG=round(m["mean_PDR_woG"], 4), PDR=round(m["mean_PDR"], 4),
                    n_eps=n_eps, ok=True)
    except Exception as e:
        return dict(scope=scope, region=region, sido=sido, gate=gate, key=key,
                    err=str(e)[:200], ok=False)


def build_jobs(n_eps, scopes, gates, limit):
    jobs = []
    A = json.load(open(os.path.join(REPO, "scenarios", "manifests", "eval_holdout_A_manifest.json")))
    # sido of each key (key=name_sigcd_pIdx)
    pts = json.load(open(os.path.join(REPO, "scenarios", "manifests", "eval_holdout_points.json")))
    sido_of = {k: v["sido"] for k, v in pts.items()}
    for gate in gates:
        suf = "occ" if gate == "occ" else "siteonly"
        if "A" in scopes:
            md = os.path.join(REPO, "results", "rl", "sigungu_nat", f"ds_ess_woG_{suf}_s0")
            for key, cfg in A.items():
                jobs.append(("A_전국", "전국", sido_of.get(key, "?"), gate, key, cfg, md, n_eps))
        if "B" in scopes:
            bdir = os.path.join(REPO, "scenarios", "manifests", "eval_holdout_sido")
            for fn in sorted(os.listdir(bdir)):
                sido = fn[:-5]
                md = os.path.join(REPO, "results", "rl", "sido", f"{sido}_ds_ess_woG_{suf}_s0")
                for key, cfg in json.load(open(os.path.join(bdir, fn))).items():
                    jobs.append(("B_시도", sido, sido, gate, key, cfg, md, n_eps))
    jobs.sort(key=lambda j: j[6])  # model_dir 정렬 → 워커 캐시 적중↑
    if limit:
        jobs = jobs[:limit]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n_eps", type=int, default=N_EPS)
    ap.add_argument("--scopes", default="A,B")
    ap.add_argument("--gates", default="occ,psent")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N잡만")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "rl_holdout_eval.csv"))
    args = ap.parse_args()

    fields = ["scope", "region", "sido", "gate", "key", "R_woG", "R", "PDR_woG", "PDR", "n_eps"]
    # 재개: 기존 CSV 로드 → 완료된 (scope,gate,key) skip
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["scope"], row["gate"], row["key"]))
    jobs = build_jobs(args.n_eps, args.scopes.split(","), args.gates.split(","), args.limit)
    jobs = [j for j in jobs if (j[0], j[3], j[4]) not in done]
    print(f"[eval_holdout] jobs={len(jobs)} 남음 (done={len(done)}, n_eps={args.n_eps}, workers={args.workers})", flush=True)
    if not jobs:
        print("[eval_holdout] 전부 완료됨(재개 불필요)", flush=True); return
    # 증분 기록(append, 매 결과 flush) — 크래시 시에도 보존
    new_file = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new_file:
        w.writeheader(); fout.flush()
    t0 = time.time(); n_ok = n_fail = 0
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                w.writerow(r); fout.flush(); n_ok += 1
            else:
                n_fail += 1
            if k % 50 == 0 or not r["ok"]:
                el = time.time() - t0
                print(f"  [{k}/{len(jobs)}] {r['scope']} {r['region']} {r['gate']} "
                      f"{'R_woG='+str(r.get('R_woG')) if r['ok'] else 'FAIL:'+str(r.get('err'))} "
                      f"ok={n_ok} fail={n_fail} ({el:.0f}s, eta {el/k*(len(jobs)-k):.0f}s)", flush=True)
    fout.close()
    print(f"\n[eval_holdout] 완료: 성공 {n_ok}, 실패 {n_fail}, wall={time.time()-t0:.0f}s, out={args.out}", flush=True)


if __name__ == "__main__":
    main()
