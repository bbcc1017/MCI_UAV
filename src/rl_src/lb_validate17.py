"""부하균형 휴리(발송상한 T) 17지역 paired 검증 — 같은시드(11000) 1000ep, occ+site.
RL·휴리best 는 simlog/episodes.csv 에 같은시드로 이미 있으니 재사용(LB만 재실행=효율).
T 여러 값 동시 평가 → 지역별 best-T + 고정 T 둘 다 보고. ep별 woG paired diff±95%CI·승무패.
"""
import os, sys, json, argparse, csv, time, collections
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"src/rl_src")
from multiprocessing import Pool
import numpy as np
REGIONS="서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
H=46; SEED=11000

def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential",MCI_GREEN_MASK="1",MCI_REWARD_MODE="woG")
    if g=="occ": os.environ.update(MCI_CAP_GATE="occ",MCI_CARED_OBS="1")
    else: os.environ.update(MCI_CAP_GATE="psent",MCI_CARED_OBS="0")

def load_simlog_woG(region,gate,policy,n_ep):
    """simlog/episodes.csv 에서 같은시드 ep별 woG 배열(ep오름차순)."""
    arr=np.full(n_ep,np.nan)
    for r in csv.DictReader(open("results/viper/simlog/episodes.csv",encoding="utf-8")):
        if r["region"]==region and r["gate"]==gate and r["policy"]==policy:
            e=int(r["ep"])
            if e<n_ep: arr[e]=float(r["woG"])
    return arr

def worker(job):
    region,gate,Ts,n_ep=job; setgate(gate)
    import torch as th; th.set_num_threads(1)
    from viper_distill import make_feature_env,_suppress_stdout
    from loadbalance_heuristic import make_cap_policy
    import pandas as pd
    try:
        cfg=json.load(open("scenarios/manifests/sido_osrm_manifest.json"))[region]
        hc="results/sido_osrm_heuristic_best.csv" if gate=="occ" else "results/sido_osrm_heuristic_psent_best.csv"
        br=pd.read_csv(hc).set_index("region").loc[region,"best_rule"]
        fac=make_feature_env(cfg,None)
        pols={t:make_cap_policy(br,t,H) for t in Ts}
        W={t:np.zeros(n_ep) for t in Ts}
        with _suppress_stdout():
            for ep in range(n_ep):
                for t in Ts:
                    env=fac(seed=SEED+ep); ro,_=env.reset(seed=SEED+ep); done=False; w=0.0
                    pol=pols[t]
                    while not done:
                        mask=np.asarray(env.action_masks(),bool); a=pol(ro,mask,env.unwrapped)
                        ro,r,te,tr,info=env.step(a); w+=info.get("r_woG",0.0); done=te or tr
                    W[t][ep]=w
        Wh=load_simlog_woG(region,gate,"heur",n_ep)
        Wr=load_simlog_woG(region,gate,"rl",n_ep)
        def stat(A,B):
            d=A-B; md=float(np.nanmean(d)); n=np.sum(~np.isnan(d)); ci=1.96*float(np.nanstd(d,ddof=1))/np.sqrt(n)
            return md,ci,float(np.nanmean(d>0)),("win" if md>ci else "loss" if md<-ci else "tie")
        out=dict(ok=True,region=region,gate=gate,heur=float(np.nanmean(Wh)),rl=float(np.nanmean(Wr)))
        rl_md,rl_ci,_,rl_sig=stat(Wr,Wh); out["rl_diff"]=rl_md; out["rl_sig"]=rl_sig
        for t in Ts:
            md,ci,wr,sig=stat(W[t],Wh)
            out[f"lb_T{t}"]=float(W[t].mean()); out[f"diff_T{t}"]=md; out[f"ci_T{t}"]=ci; out[f"sig_T{t}"]=sig; out[f"wr_T{t}"]=wr
        return out
    except Exception as e:
        import traceback; return dict(ok=False,region=region,gate=gate,err=(str(e)+traceback.format_exc())[:300])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--Ts",default="3,4,5")
    ap.add_argument("--workers",type=int,default=34); ap.add_argument("--n_ep",type=int,default=1000)
    ap.add_argument("--gates",default="occ,site"); ap.add_argument("--out",default="results/viper/lb_paired_17.csv")
    A=ap.parse_args()
    Ts=[float(x) for x in A.Ts.split(",")]
    jobs=[(r,g,Ts,A.n_ep) for g in A.gates.split(",") for r in REGIONS]
    print(f"[lb17] Ts={Ts} jobs={len(jobs)} n_ep={A.n_ep}",flush=True)
    res=[]; t0=time.time()
    with Pool(A.workers,maxtasksperchild=1) as pool:
        for k,r in enumerate(pool.imap_unordered(worker,jobs),1):
            res.append(r)
            if r["ok"]:
                bt=max(Ts,key=lambda t:r[f"diff_T{t}"]); print(f"  [{k}/{len(jobs)}] {r['region']} {r['gate']}: bestT{int(bt)} {r[f'diff_T{bt}']:+.2f}({r[f'sig_T{bt}']}) | T4 {r['diff_T4.0']:+.2f}({r['sig_T4.0']}) RL {r['rl_diff']:+.2f} ({time.time()-t0:.0f}s)",flush=True)
            else: print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['gate']}: {r['err'][:120]}",flush=True)
    ok=[r for r in res if r["ok"]]
    cols=sorted({k for r in ok for k in r if k!="ok"})
    with open(A.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["region","gate"]+[c for c in cols if c not in("region","gate")]); w.writeheader()
        for r in ok: w.writerow({k:v for k,v in r.items() if k!="ok"})
    print(f"\n=== 17지역 paired: 부하균형 휴리(발송상한) vs 휴리best ===",flush=True)
    for g in A.gates.split(","):
        gg=[r for r in ok if r["gate"]==g]
        print(f"[{g}]")
        for t in Ts:
            win=sum(r[f"sig_T{t}"]=="win" for r in gg); tie=sum(r[f"sig_T{t}"]=="tie" for r in gg); loss=sum(r[f"sig_T{t}"]=="loss" for r in gg)
            md=np.mean([r[f"diff_T{t}"] for r in gg])
            print(f"  고정 T={int(t)}: diff평{md:+.2f}  승{win}/무{tie}/패{loss}")
        # 지역별 best-T
        bestdiffs=[max(r[f"diff_T{t}"] for t in Ts) for r in gg]
        bw=sum(1 for r in gg if max((r[f"diff_T{t}"],r[f"ci_T{t}"]) for t in Ts) and max(r[f"diff_T{t}"]-r[f"ci_T{t}"] for t in Ts)>0)
        print(f"  지역별 best-T: diff평{np.mean(bestdiffs):+.2f}")
        print(f"  RL상한: diff평{np.mean([r['rl_diff'] for r in gg]):+.2f}  승{sum(r['rl_sig']=='win' for r in gg)}/무{sum(r['rl_sig']=='tie' for r in gg)}/패{sum(r['rl_sig']=='loss' for r in gg)}")
    print(f"저장 {A.out}  wall={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
