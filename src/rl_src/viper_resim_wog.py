"""4개 RL 실험({A 시군구전국, B 시도17} × {occ, psent}) 증류 트리를 woG 로 재시뮬레이션.

§4.1 에서 증류한 트리(viper_loggap_d12.pkl)들은 vecnorm 적용해 학습됐으나 평가를 raw 로 했음.
raw 는 Green 고정생존(~60)이 지배해 정책차에 둔감 → 학습목표 woG(=info['r_woG'] 합산, eval_policy
가 반환)로 재평가한다. RL 추가 학습/재증류 없음 — 고정 트리를 sim 에 돌려 woG 만 다시 잰다.

게이트는 MCI_CAP_GATE 로 지정(occ | psent[=siteonly]). psent 면 MCI_CARED_OBS=0 도 필요.
출력: results/viper/RESIM_<gate>.csv (지역별 tree_woG / ppo_woG / heur_woG / fidelity).
"""
import argparse, csv, glob, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from viper_distill import make_feature_env, load_vecnorm, make_tree_policy, _suppress_stdout
from evaluate import eval_policy, ppo_policy


def best_fidelity(pkl):
    h = pkl.get("history", [])
    return max((x.get("fidelity", 0) for x in h), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["occ", "psent"])
    ap.add_argument("--eval_eps", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--only", default="all", help="단일 지역명(또는 '전국') — 병렬용. 'all'=전체")
    ap.add_argument("--out_dir", default="results/viper", help="출력 디렉터리")
    args = ap.parse_args()
    gate = args.gate
    suf = "occ" if gate == "occ" else "siteonly"

    from sb3_contrib import MaskablePPO
    heur = pd.read_csv("results/sido_osrm_heuristic_best.csv" if gate == "occ"
                       else "results/sido_osrm_heuristic_psent_best.csv")
    heur_wog = dict(zip(heur.region, heur.reward_wog))
    sg_heur = pd.read_csv("results/sigungu_heuristic_best.csv" if gate == "occ"
                          else "results/sigungu_heuristic_psent_best.csv")
    sg_heur_wog = float(sg_heur.reward_wog.mean())

    REGIONS = "서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
    # (scope, region, tree_pkl, model, src)
    jobs = []
    sg_tree = (f"results/viper/sigungu_nat_{'occ' if gate=='occ' else 'psent'}_hq/viper_loggap_d12.pkl")
    jobs.append(("A_시군구전국", "전국", sg_tree,
                 f"results/rl/sigungu_nat/ds_ess_woG_{suf}_s0/final_model.zip",
                 "scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"))
    sido_manifest = json.load(open("scenarios/manifests/sido_osrm_manifest.json"))
    for rg in REGIONS:
        jobs.append(("B_시도", rg, f"results/viper/sido/{rg}_{gate}/viper_loggap_d12.pkl",
                     f"results/rl/sido/{rg}_ds_ess_woG_{suf}_s0/final_model.zip",
                     sido_manifest[rg]))

    if args.only != "all":
        jobs = [j for j in jobs if j[1] == args.only]

    rows = []
    for scope, rg, treep, modelp, src in jobs:
        if not (os.path.exists(treep) and os.path.exists(modelp)):
            sys.stderr.write(f"SKIP {scope} {rg} (파일없음)\n"); continue
        d = pickle.load(open(treep, "rb"))
        tree = d["tree"]
        model = MaskablePPO.load(modelp)
        vn = os.path.join(os.path.dirname(modelp), "vecnormalize.pkl")
        norm = load_vecnorm(vn) if os.path.exists(vn) else None
        factory = make_feature_env(src, norm)
        with _suppress_stdout():
            m_tree = eval_policy(factory, make_tree_policy(tree), args.eval_eps, args.seed_base + 9000)
            m_ppo = eval_policy(factory, ppo_policy(model), args.eval_eps, args.seed_base + 9000)
        hw = sg_heur_wog if rg == "전국" else heur_wog.get(rg, float("nan"))
        row = {"scope": scope, "region": rg, "gate": gate,
               "tree_woG": round(m_tree["mean_R_woG"], 2),
               "ppo_woG": round(m_ppo["mean_R_woG"], 2),
               "heur_woG": round(hw, 2),
               "tree_vs_ppo": round(m_tree["mean_R_woG"] - m_ppo["mean_R_woG"], 2),
               "tree_vs_heur": round(m_tree["mean_R_woG"] - hw, 2),
               "tree_raw": round(m_tree["mean_R"], 2),
               "fidelity": round(best_fidelity(d) or 0, 3)}
        rows.append(row)
        sys.stderr.write(f"[{scope} {rg} {gate}] tree_woG={row['tree_woG']} ppo={row['ppo_woG']} "
                         f"heur={row['heur_woG']} (vs_heur {row['tree_vs_heur']:+})\n"); sys.stderr.flush()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.only != "all":
        out = f"{args.out_dir}/{gate}__{args.only}.csv"   # 병렬: 지역별 1파일(레이스 방지)
    else:
        out = f"{args.out_dir}/RESIM_{gate}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[저장] {out} ({len(rows)}행)")


if __name__ == "__main__":
    main()
