"""공정 paired 평가 — 시도17×{occ,site} 같은 시드 1000ep 에서 휴리best vs 트리(d4/d6/d12) vs RL.
ep별 woG → paired diff(정책−휴리)·95%CI·승률·유의성. unpaired 평균비교의 통계적 정정.
같은 eval_policy 경로·같은 시드(SEED_BASE). 휴리는 make_heuristic_policy(같은 경로).

출력: results/viper/fair_paired_17.csv (region,gate,policy,woG,heur_woG,diff,ci95,win_rate,sig,best_rule)
예: PYTHONIOENCODING=utf-8 python src/rl_src/fair_paired_eval.py --workers 17 --n_ep 1000
"""
import os, sys, argparse, csv, time, pickle, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))
REGIONS="서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
SEED_BASE=11000

def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential",MCI_GREEN_MASK="1",MCI_REWARD_MODE="woG")
    if g=="occ": os.environ.update(MCI_CAP_GATE="occ",MCI_CARED_OBS="1")
    else:        os.environ.update(MCI_CAP_GATE="psent",MCI_CARED_OBS="0")

def eval_perep(factory, policy, n, seed_base):
    from viper_distill import _suppress_stdout
    out=np.empty(n)
    with _suppress_stdout():
        for ep in range(n):
            env=factory(seed=seed_base+ep); obs,_=env.reset(seed=seed_base+ep); done=False; rw=0.0
            while not done:
                m=env.action_masks(); a=policy(obs,m,env.unwrapped)
                obs,r,te,tr,info=env.step(a); rw+=info.get("r_woG",0.0); done=te or tr
            out[ep]=rw
    return out

def worker(job):
    region,gate,n_ep=job; setgate(gate)
    import json, torch as th; th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from viper_distill import make_feature_env, load_vecnorm, make_tree_policy
    from evaluate import ppo_policy
    from distill_policy import make_heuristic_policy
    import pandas as pd
    try:
        suf="occ" if gate=="occ" else "siteonly"
        md=os.path.join(REPO,f"results/rl/sido/{region}_ds_ess_woG_{suf}_s0")
        cfg=json.load(open(os.path.join(REPO,"scenarios/manifests/sido_osrm_manifest.json")))[region]
        hcsv="results/sido_osrm_heuristic_best.csv" if gate=="occ" else "results/sido_osrm_heuristic_psent_best.csv"
        hrow=pd.read_csv(os.path.join(REPO,hcsv)).set_index("region").loc[region]
        best_rule=hrow["best_rule"]
        model=MaskablePPO.load(os.path.join(md,"final_model.zip"),device="cpu")
        norm=load_vecnorm(os.path.join(md,"vecnormalize.pkl"))
        fac=make_feature_env(cfg,norm)
        zoo=os.path.join(REPO,f"results/viper/zoo/B시도_{region}_{gate}")
        trees={d:pickle.load(open(os.path.join(zoo,f"tree_d{d}.pkl"),"rb"))["tree"] for d in [4,6,12]}
        pols={"heur":make_heuristic_policy(best_rule),"rl":ppo_policy(model)}
        for d,t in trees.items(): pols[f"tree_d{d}"]=make_tree_policy(t)
        W={k:eval_perep(fac,p,n_ep,SEED_BASE) for k,p in pols.items()}
        h=W["heur"]; rows=[]
        for k in ["tree_d4","tree_d6","tree_d12","rl"]:
            d=W[k]-h; sd=d.std(ddof=1); ci=1.96*sd/np.sqrt(len(d)); md_=d.mean()
            sig="win" if md_>ci else ("loss" if md_<-ci else "tie")
            rows.append(dict(region=region,gate=gate,policy=k,woG=round(W[k].mean(),3),
                             heur_woG=round(h.mean(),3),diff=round(md_,3),ci95=round(ci,3),
                             win_rate=round(float((d>0).mean()),3),sig=sig,best_rule=best_rule))
        return dict(ok=True,rows=rows,region=region,gate=gate)
    except Exception as e:
        import traceback
        return dict(ok=False,region=region,gate=gate,err=(str(e)+"|"+traceback.format_exc())[:300])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=17); ap.add_argument("--n_ep",type=int,default=1000)
    ap.add_argument("--gates",default="occ,site")
    ap.add_argument("--out",default=os.path.join(REPO,"results/viper/fair_paired_17.csv"))
    A=ap.parse_args()
    jobs=[(r,g,A.n_ep) for g in A.gates.split(",") for r in REGIONS]
    done=set()
    if os.path.exists(A.out):
        for r in csv.DictReader(open(A.out,encoding="utf-8")): done.add((r["region"],r["gate"]))
    jobs=[j for j in jobs if (j[0],j[1]) not in done]
    print(f"[fair17] jobs={len(jobs)} (done={len(done)}, n_ep={A.n_ep})",flush=True)
    if not jobs: print("[fair17] 완료"); return
    fields=["region","gate","policy","woG","heur_woG","diff","ci95","win_rate","sig","best_rule"]
    new=(not os.path.exists(A.out)) or os.path.getsize(A.out)==0
    fout=open(A.out,"a",newline="",encoding="utf-8"); w=csv.DictWriter(fout,fieldnames=fields,extrasaction="ignore")
    if new: w.writeheader(); fout.flush()
    t0=time.time(); nok=nf=0
    with Pool(A.workers,maxtasksperchild=1) as pool:
        for k,r in enumerate(pool.imap_unordered(worker,jobs),1):
            if r["ok"]:
                for row in r["rows"]: w.writerow(row)
                fout.flush(); nok+=1
                d6=next(x for x in r["rows"] if x["policy"]=="tree_d6")
                print(f"  [{k}/{len(jobs)}] {r['region']} {r['gate']}: d6 {d6['diff']:+.2f}±{d6['ci95']:.2f}[{d6['sig']}] 승률{d6['win_rate']*100:.0f}% ({time.time()-t0:.0f}s)",flush=True)
            else:
                nf+=1; print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['gate']}: {r['err'][:150]}",flush=True)
    fout.close(); print(f"\n[fair17] 완료 ok={nok} fail={nf} wall={time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
