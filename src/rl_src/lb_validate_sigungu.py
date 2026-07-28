"""부하균형 휴리(발송상한 T=4) 시군구250 paired 검증 — 같은시드(11000) 1000ep, occ+site.
RL(전국정책)·휴리best(시군구별) 는 simlog_sigungu/episodes.csv 에 같은시드로 존재 → 재사용(LB만 재실행).
17지역 결과(LB가 휴리·RL 모두 능가)의 전국 250 일반화 확인.
"""
import os, sys, json, argparse, csv, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"src/rl_src")
from multiprocessing import Pool
import numpy as np
MANIFEST="scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json"
REGIONS=sorted(json.load(open(MANIFEST)).keys())
H=47; SEED=11000  # 2026-07-02 성남 정정: H 46→47
EP_CSV="results/viper/simlog_sigungu/episodes.csv"

def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential",MCI_GREEN_MASK="1",MCI_REWARD_MODE="woG")
    if g=="occ": os.environ.update(MCI_CAP_GATE="occ",MCI_CARED_OBS="1")
    else: os.environ.update(MCI_CAP_GATE="psent",MCI_CARED_OBS="0")

# simlog 한번만 읽어 인덱싱(워커마다 재파싱 방지 위해 전역 캐시)
_SIM=None
def simlog():
    global _SIM
    if _SIM is None:
        d={}
        for r in csv.DictReader(open(EP_CSV,encoding="utf-8")):
            d.setdefault((r["region"],r["gate"],r["policy"]),{})[int(r["ep"])]=float(r["woG"])
        _SIM=d
    return _SIM

def woG_arr(region,gate,policy,n_ep):
    m=simlog().get((region,gate,policy),{}); a=np.full(n_ep,np.nan)
    for e,v in m.items():
        if e<n_ep: a[e]=v
    return a

def worker(job):
    region,gate,T,n_ep,Wh,Wr=job; setgate(gate)
    import torch as th; th.set_num_threads(1)
    from viper_distill import make_feature_env,_suppress_stdout
    from loadbalance_heuristic import make_cap_policy
    import pandas as pd
    try:
        cfg=json.load(open(MANIFEST))[region]
        sig=region.rsplit("_",1)[1]
        hc="results/sigungu_heuristic_best.csv" if gate=="occ" else "results/sigungu_heuristic_psent_best.csv"
        br=pd.read_csv(hc,dtype={"sigcd":str},encoding="utf-8-sig").set_index("sigcd").loc[sig,"best_rule"]
        fac=make_feature_env(cfg,None); pol=make_cap_policy(br,T,H)
        Wl=np.zeros(n_ep)
        with _suppress_stdout():
            for ep in range(n_ep):
                env=fac(seed=SEED+ep); ro,_=env.reset(seed=SEED+ep); done=False; w=0.0
                while not done:
                    mask=np.asarray(env.action_masks(),bool); a=pol(ro,mask,env.unwrapped)
                    ro,r,te,tr,info=env.step(a); w+=info.get("r_woG",0.0); done=te or tr
                Wl[ep]=w
        def stat(A,B):
            d=A-B; md=float(np.nanmean(d)); n=np.sum(~np.isnan(d)); ci=1.96*float(np.nanstd(d,ddof=1))/np.sqrt(n)
            return md,ci,("win" if md>ci else "loss" if md<-ci else "tie")
        lm,lc,ls=stat(Wl,Wh); rm,rc,rs=stat(Wr,Wh)
        return dict(ok=True,region=region,gate=gate,heur=float(np.nanmean(Wh)),rl=float(np.nanmean(Wr)),lb=float(Wl.mean()),
                    lb_diff=lm,lb_ci=lc,lb_sig=ls,rl_diff=rm,rl_sig=rs)
    except Exception as e:
        import traceback; return dict(ok=False,region=region,gate=gate,err=(str(e)+traceback.format_exc())[:200])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--T",type=float,default=4)
    ap.add_argument("--workers",type=int,default=40); ap.add_argument("--n_ep",type=int,default=1000)
    ap.add_argument("--gates",default="occ,site"); ap.add_argument("--out",default="results/viper/lb_paired_sigungu.csv")
    ap.add_argument("--limit",type=int,default=0); A=ap.parse_args()
    regs=REGIONS[:A.limit] if A.limit>0 else REGIONS
    S=simlog()
    jobs=[(r,g,A.T,A.n_ep,woG_arr(r,g,"heur",A.n_ep),woG_arr(r,g,"rl",A.n_ep)) for g in A.gates.split(",") for r in regs]
    print(f"[lb-sgg] T={A.T} jobs={len(jobs)} n_ep={A.n_ep}",flush=True)
    res=[]; t0=time.time()
    with Pool(A.workers,maxtasksperchild=1) as pool:
        for k,r in enumerate(pool.imap_unordered(worker,jobs),1):
            res.append(r)
            if not r["ok"]: print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['gate']}: {r['err'][:120]}",flush=True)
            elif k%50==0: print(f"  [{k}/{len(jobs)}] {r['region']} {r['gate']}: LB{r['lb_diff']:+.2f}({r['lb_sig']}) RL{r['rl_diff']:+.2f} ({time.time()-t0:.0f}s)",flush=True)
    ok=[r for r in res if r["ok"]]
    with open(A.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=[k for k in ok[0] if k!="ok"]); w.writeheader()
        for r in ok: w.writerow({k:v for k,v in r.items() if k!="ok"})
    print(f"\n=== 시군구250 paired: 발송상한 T={A.T} vs 휴리best ===",flush=True)
    for g in A.gates.split(","):
        gg=[r for r in ok if r["gate"]==g]
        lw=sum(r["lb_sig"]=="win" for r in gg); lt=sum(r["lb_sig"]=="tie" for r in gg); ll=sum(r["lb_sig"]=="loss" for r in gg)
        rw=sum(r["rl_sig"]=="win" for r in gg); rt=sum(r["rl_sig"]=="tie" for r in gg); rl=sum(r["rl_sig"]=="loss" for r in gg)
        print(f"  [{g}] LB diff평{np.mean([r['lb_diff'] for r in gg]):+.2f} 승{lw}/무{lt}/패{ll}  |  RL diff평{np.mean([r['rl_diff'] for r in gg]):+.2f} 승{rw}/무{rt}/패{rl} (n={len(gg)})",flush=True)
        # LB가 RL 능가한 지역수
        lbwin=sum(1 for r in gg if r["lb_diff"]>r["rl_diff"]); print(f"       LB diff > RL diff 지역: {lbwin}/{len(gg)}",flush=True)
    print(f"저장 {A.out}  wall={time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
