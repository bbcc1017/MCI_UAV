"""VIPER 재증류 (10M 오라클) — 기존 5M/2M 트리를 10M 모델로 갱신·성능 강화.

개선점(기존 viper_distill 대비):
  1) **10M 오라클**(기존 시군구 5M·시도 2M → 전부 10M final_model.zip).
  2) **best-iter 선택을 woG 기준**(viper(select_metric='woG')) — 기존 raw(Green 포화·둔감)로
     골라 woG 최적이 아닐 수 있던 문제 교정. 배치 지표(woG)에 정합.
  3) **롤아웃 데이터 증량** — 특히 A 전국(250지역 단일트리, fidelity 0.40 약점)은 rollout_eps↑·
     트리 용량(depth/min_leaf)↑로 지역 커버리지 확보.
  4) 최종 평가 woG 1000ep(휴리스틱과 동일 — 공정), PPO·휴리스틱과 동시 비교.

대상 36오라클 = (A 전국 occ/psent) + (B 시도17 × occ/psent). 각 잡 = 독립 프로세스
(maxtasksperchild=1)로 게이트 env var 깨끗이 세팅. 산출: results/viper/v2_10m/.

예: PYTHONIOENCODING=utf-8 python src/rl_src/viper_redistill_10m.py --workers 32
"""
import os
# 스레드 핀(numpy/torch import 전) — 36 프로세스 병렬, 미설정시 코어 폭주.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, json, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

REGIONS = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
FINAL_EPS = 1000   # 최종 woG 평가 에피소드(휴리스틱·이전 RESIM1000 과 동일)
SEED_BASE = 2000


def _set_gate(gate):
    os.environ["MCI_OBS_VARIANT"] = "essential"
    os.environ["MCI_GREEN_MASK"] = "1"
    os.environ["MCI_REWARD_MODE"] = "woG"
    if gate == "occ":
        os.environ["MCI_CAP_GATE"] = "occ"; os.environ["MCI_CARED_OBS"] = "1"
    else:  # psent = siteonly
        os.environ["MCI_CAP_GATE"] = "psent"; os.environ["MCI_CARED_OBS"] = "0"


def worker(job):
    scope, region, gate, model_dir, src, hp = job
    _set_gate(gate)  # env build 전에 게이트 설정(래퍼가 os.environ 읽음)
    import torch as th
    th.set_num_threads(1)
    import numpy as np
    from sklearn.tree import export_text
    from sb3_contrib import MaskablePPO
    from viper_distill import (viper, make_feature_env, load_vecnorm,
                               make_tree_policy, _suppress_stdout)
    from evaluate import eval_policy, ppo_policy
    try:
        model = MaskablePPO.load(os.path.join(model_dir, "final_model.zip"), device="cpu")
        vn = os.path.join(model_dir, "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        factory = make_feature_env(src, norm)
        # ── 증류 (woG 기준 best-iter) ──
        best, history = viper(factory, model, hp["n_iter"], hp["rollout_eps"], hp["eval_eps"],
                              hp["max_depth"], hp["min_samples_leaf"], "loggap",
                              SEED_BASE, select_metric="woG")
        tree = best["tree"]
        # ── 최종 woG 1000ep (tree + PPO) ──
        with _suppress_stdout():
            m_tree = eval_policy(factory, make_tree_policy(tree), FINAL_EPS, SEED_BASE + 9000)
            m_ppo = eval_policy(factory, ppo_policy(model), FINAL_EPS, SEED_BASE + 9000)
        # 충실도(best iter 의 누적 D 재현율 최대값)
        fid = max((h.get("fidelity", 0) for h in history), default=0.0)
        # ── 저장 ──
        suf = "occ" if gate == "occ" else "psent"
        tag = ("sigungu_nat" if scope.startswith("A") else region) + f"_{suf}"
        odir = os.path.join(REPO, "results", "viper", "v2_10m", tag)
        os.makedirs(odir, exist_ok=True)
        with open(os.path.join(odir, "viper_v2.pkl"), "wb") as f:
            pickle.dump({"tree": tree, "history": history, "crit": "loggap",
                         "select_metric": "woG", "hp": hp, "best_iter": best["iter"]}, f)
        with open(os.path.join(odir, "viper_v2_rules.txt"), "w", encoding="utf-8") as f:
            f.write(export_text(tree, max_depth=min(hp["max_depth"], 12)))
        return dict(ok=True, scope=scope, region=region, gate=gate,
                    fidelity=round(fid, 4), tree_woG=round(m_tree["mean_R_woG"], 3),
                    ppo_woG=round(m_ppo["mean_R_woG"], 3), tree_raw=round(m_tree["mean_R"], 2),
                    leaves=int(tree.get_n_leaves()), depth=int(tree.get_depth()),
                    best_iter=best["iter"], n_iter=hp["n_iter"], rollout_eps=hp["rollout_eps"])
    except Exception as e:
        import traceback
        return dict(ok=False, scope=scope, region=region, gate=gate,
                    err=(str(e) + " | " + traceback.format_exc())[:400])


def build_jobs(scopes, gates):
    sido_manifest = json.load(open(os.path.join(REPO, "scenarios/manifests/sido_osrm_manifest.json")))
    sg_manifest = os.path.join(REPO, "scenarios/manifests/sigungu_osrm_manifest.json")
    # 하이퍼파라미터: 전국=고용량(250지역 단일트리), 시도=기존 양호 설정 강화
    HP_NAT = dict(n_iter=12, rollout_eps=80, eval_eps=50, max_depth=14, min_samples_leaf=10)
    HP_SIDO = dict(n_iter=12, rollout_eps=30, eval_eps=40, max_depth=12, min_samples_leaf=20)
    jobs = []
    for gate in gates:
        suf = "occ" if gate == "occ" else "siteonly"
        if "A" in scopes:
            md = os.path.join(REPO, "results/rl/sigungu_nat", f"ds_ess_woG_{suf}_s0")
            jobs.append(("A_시군구전국", "전국", gate, md, sg_manifest, HP_NAT))
        if "B" in scopes:
            for rg in REGIONS:
                md = os.path.join(REPO, "results/rl/sido", f"{rg}_ds_ess_woG_{suf}_s0")
                jobs.append(("B_시도", rg, gate, md, sido_manifest[rg], HP_SIDO))
    return jobs


def load_heur():
    import pandas as pd
    h = {"occ": {}, "psent": {}}
    for g, f in [("occ", "results/sido_osrm_heuristic_best.csv"),
                 ("psent", "results/sido_osrm_heuristic_psent_best.csv")]:
        df = pd.read_csv(os.path.join(REPO, f))
        h[g] = dict(zip(df.region, df.reward_wog))
    sg = {"occ": float(pd.read_csv(os.path.join(REPO, "results/sigungu_heuristic_best.csv")).reward_wog.mean()),
          "psent": float(pd.read_csv(os.path.join(REPO, "results/sigungu_heuristic_psent_best.csv")).reward_wog.mean())}
    return h, sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--scopes", default="A,B")
    ap.add_argument("--gates", default="occ,psent")
    ap.add_argument("--out", default=os.path.join(REPO, "results/viper/V2_10M_REDISTILL.csv"))
    args = ap.parse_args()

    heur, sg_heur = load_heur()
    jobs = build_jobs(args.scopes.split(","), args.gates.split(","))
    # 재개: 완료된 (scope,region,gate) skip
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["scope"], r["region"], r["gate"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"[redistill] jobs={len(jobs)} (done={len(done)}, workers={args.workers}, final_eps={FINAL_EPS})", flush=True)
    if not jobs:
        print("[redistill] 전부 완료"); return

    fields = ["scope", "region", "gate", "fidelity", "tree_woG", "ppo_woG", "heur_woG",
              "tree_vs_heur", "tree_vs_ppo", "tree_raw", "leaves", "depth", "best_iter",
              "n_iter", "rollout_eps"]
    new_file = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if new_file:
        w.writeheader(); fout.flush()
    t0 = time.time(); n_ok = n_fail = 0
    # 전국 잡(무거움)을 먼저 띄우기 위해 정렬
    jobs.sort(key=lambda j: 0 if j[0].startswith("A") else 1)
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
            if r["ok"]:
                g = r["gate"]
                hw = sg_heur[g] if r["region"] == "전국" else heur[g].get(r["region"], float("nan"))
                r["heur_woG"] = round(hw, 3)
                r["tree_vs_heur"] = round(r["tree_woG"] - hw, 3)
                r["tree_vs_ppo"] = round(r["tree_woG"] - r["ppo_woG"], 3)
                w.writerow(r); fout.flush(); n_ok += 1
                el = time.time() - t0
                print(f"  [{k}/{len(jobs)}] {r['scope']} {r['region']} {g}: "
                      f"tree_woG={r['tree_woG']} ppo={r['ppo_woG']} heur={r['heur_woG']} "
                      f"(vs_heur {r['tree_vs_heur']:+}, fid {r['fidelity']}, "
                      f"leaves {r['leaves']}) ok={n_ok} fail={n_fail} "
                      f"({el:.0f}s, eta {el/k*(len(jobs)-k):.0f}s)", flush=True)
            else:
                n_fail += 1
                print(f"  [{k}/{len(jobs)}] FAIL {r['scope']} {r['region']} {r['gate']}: {r['err']}", flush=True)
    fout.close()
    print(f"\n[redistill] 완료: ok={n_ok} fail={n_fail} wall={time.time()-t0:.0f}s out={args.out}", flush=True)


if __name__ == "__main__":
    main()
