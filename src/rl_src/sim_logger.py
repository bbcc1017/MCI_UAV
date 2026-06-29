"""시뮬레이션 로그 저장 — RL/휴리best/트리d6 를 시도17×{occ,site} 같은시드 1000ep 재실행하며
결정·병원부하·에피소드 로그를 저장(결정론이라 평가와 동일 시뮬 재현). 사후 부하분산/대기/포화 분석 기반.

산출(results/viper/simlog/):
  decisions_<region>_<gate>_<policy>.csv.gz  — 결정 long: ep,step,time,cls,dest,mode,hosp,cap_remain,eta_amb,eta_uav,eta_rank,n_avail,r_woG
  episodes.csv      — ep요약: region,gate,policy,ep,woG,raw,time_end,n_transport,n_used_hosp,gini,max_share,mean_eta_rank,n_uav,n_amb,q_peak,n_sat_steps
  hospital_loads.csv— 병원집계: region,gate,policy,hosp,tier3,arrivals,peak_queue,peak_occ,sat_steps,max_capa

정책 입력: RL/트리는 VecNorm 정규화 obs, 휴리는 env.unwrapped(무관). rollout은 raw feature obs.
예: PYTHONIOENCODING=utf-8 python src/rl_src/sim_logger.py --workers 24 --n_ep 1000 --policies rl,heur,tree_d6
"""
import os, sys, argparse, csv, gzip, time, pickle, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))
REGIONS="서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충북 충남 전북 전남 경북 경남 제주".split()
SEED=11000; H=46; ND=H+1; NM=2
OUT=os.path.join(REPO,"results/viper/simlog")
# 시나리오축 오버라이드(Kakao 등). main()에서 인자로 교체, fork 로 워커에 전파. 기본=시도 OSRM.
MODEL_BASE="results/rl/sido"; CFG_MANIFEST="scenarios/manifests/sido_osrm_manifest.json"; TREE_TAG="B시도"
HEUR_OCC="results/sido_osrm_heuristic_best.csv"; HEUR_PSENT="results/sido_osrm_heuristic_psent_best.csv"

def setgate(g):
    os.environ.update(MCI_OBS_VARIANT="essential",MCI_GREEN_MASK="1",MCI_REWARD_MODE="woG")
    if g=="occ": os.environ.update(MCI_CAP_GATE="occ",MCI_CARED_OBS="1")
    else:        os.environ.update(MCI_CAP_GATE="psent",MCI_CARED_OBS="0")

def gini(x):
    x=np.sort(np.asarray(x,float)); n=len(x); s=x.sum()
    return 0.0 if s==0 else float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s))

def worker(job):
    region,gate,policy,n_ep=job; setgate(gate)
    import json, torch as th; th.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from viper_distill import make_feature_env, load_vecnorm, make_tree_policy, _suppress_stdout
    from evaluate import ppo_policy
    from distill_policy import make_heuristic_policy
    import pandas as pd
    try:
        suf="occ" if gate=="occ" else "siteonly"
        md=os.path.join(REPO,f"{MODEL_BASE}/{region}_ds_ess_woG_{suf}_s0")
        cfg=json.load(open(os.path.join(REPO,CFG_MANIFEST)))[region]
        mean,std,clip=load_vecnorm(os.path.join(md,"vecnormalize.pkl"))
        fac=make_feature_env(cfg,None)  # raw feature obs
        # 정책 빌드
        if policy=="rl":
            model=MaskablePPO.load(os.path.join(md,"final_model.zip"),device="cpu")
            def act(ro,mask,env): return int(model.predict(np.clip((np.asarray(ro,np.float32)-mean)/std,-clip,clip),action_masks=mask,deterministic=True)[0])
        elif policy=="heur":
            br=pd.read_csv(os.path.join(REPO,HEUR_OCC if gate=="occ" else HEUR_PSENT)).set_index("region").loc[region,"best_rule"]
            hp=make_heuristic_policy(br)
            def act(ro,mask,env): return hp(ro,mask,env)
        else:  # tree_dN
            dN=policy.split("_d")[1]
            tr=pickle.load(open(os.path.join(REPO,f"results/viper/zoo/{TREE_TAG}_{region}_{gate}/tree_d{dN}.pkl"),"rb"))["tree"]
            tp=make_tree_policy(tr)
            def act(ro,mask,env): return tp(np.clip((np.asarray(ro,np.float32)-mean)/std,-clip,clip),mask,env)
        # 병원 정적속성
        with _suppress_stdout():
            env0=fac(seed=SEED); env0.reset(seed=SEED)
        hp_props=env0.unwrapped.en_manager.en_properties['hospital']
        max_capa=np.asarray(hp_props['hos_max_capa'],float)
        tier3=(np.asarray(hp_props.get('hos_tier',np.zeros(H)))==3).astype(int)
        # 누적
        os.makedirs(OUT,exist_ok=True)
        dec_path=os.path.join(OUT,f"decisions_{region}_{gate}_{policy}.csv.gz")
        arrivals=np.zeros(H); peak_q=np.zeros(H); peak_occ=np.zeros(H); sat_steps=np.zeros(H)
        ep_rows=[]
        with gzip.open(dec_path,"wt",newline="",encoding="utf-8") as fdec, _suppress_stdout():
            wd=csv.writer(fdec); wd.writerow(["ep","step","time","cls","dest","mode","hosp","cap_remain","eta_amb","eta_uav","eta_rank","n_avail","r_woG"])
            for ep in range(n_ep):
                env=fac(seed=SEED+ep); ro,_=env.reset(seed=SEED+ep); done=False
                wog=0.0; raw=0.0; step=0; ntr=0; nuav=0; namb=0; eta_ranks=[]; ep_arr=np.zeros(H)
                while not done:
                    mask=np.asarray(env.action_masks(),bool)
                    a=act(ro,mask,env.unwrapped)
                    c=a//(ND*NM); rem=a%(ND*NM); dst=rem//NM; m=rem%NM
                    rr=np.asarray(ro,np.float32); HF=rr[:H*4].reshape(H,4)
                    erk=-1; navail=0; hosp=-1; capr=eta_a=eta_u=-1
                    if dst>0:
                        hosp=dst-1; capr=float(HF[hosp,1]); eta_a=float(HF[hosp,2]); eta_u=float(HF[hosp,3])
                        cap=HF[:,1]; tier=HF[:,0]; eta=HF[:,2] if m==0 else HF[:,3]
                        av=cap>0
                        if c==0: av=av&(tier>0.5)
                        navail=int(av.sum())
                        if navail>0: erk=int((eta[av]<eta[hosp]).sum())+1; eta_ranks.append(erk)
                        arrivals[hosp]+=1; ep_arr[hosp]+=1; ntr+=1
                        if m==1: nuav+=1
                        else: namb+=1
                    ro,r,te,tr,info=env.step(a); rw=info.get("r_woG",0.0); wog+=rw; raw+=r; done=te or tr; step+=1
                    # 병원 큐/포화 (step 후 상태)
                    hs=env.unwrapped.en_manager.get_full_obs()['h_states']  # (H,3) idle,queue,occ
                    q=hs[:,1]; occ=hs[:,2]
                    peak_q=np.maximum(peak_q,q); peak_occ=np.maximum(peak_occ,occ)
                    sat_steps+=(occ>=max_capa).astype(float)
                    wd.writerow([ep,step,round(float(info.get('time',0)),2),int(c),int(dst),int(m),int(hosp),
                                 round(capr,3),round(eta_a,3),round(eta_u,3),erk,navail,round(rw,4)])
                used=int((ep_arr>0).sum())
                ep_rows.append(dict(region=region,gate=gate,policy=policy,ep=ep,woG=round(wog,3),raw=round(raw,2),
                    time_end=round(float(info.get('time',0)),2),n_transport=ntr,n_used_hosp=used,
                    gini=round(gini(ep_arr),3),max_share=round(float(ep_arr.max()/max(ep_arr.sum(),1)),3),
                    mean_eta_rank=round(float(np.mean(eta_ranks)) if eta_ranks else 0,2),n_uav=nuav,n_amb=namb))
        hosp_rows=[dict(region=region,gate=gate,policy=policy,hosp=h,tier3=int(tier3[h]),
                        arrivals=int(arrivals[h]),peak_queue=int(peak_q[h]),peak_occ=int(peak_occ[h]),
                        sat_steps=int(sat_steps[h]),max_capa=int(max_capa[h])) for h in range(H)]
        return dict(ok=True,region=region,gate=gate,policy=policy,ep_rows=ep_rows,hosp_rows=hosp_rows,dec_path=dec_path)
    except Exception as e:
        import traceback
        return dict(ok=False,region=region,gate=gate,policy=policy,err=(str(e)+"|"+traceback.format_exc())[:300])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=24); ap.add_argument("--n_ep",type=int,default=1000)
    ap.add_argument("--gates",default="occ,site"); ap.add_argument("--policies",default="rl,heur,tree_d6")
    # 시나리오축 오버라이드(Kakao 등; 기본=시도 OSRM). RL모델base·증류config·휴리·트리태그·출력.
    ap.add_argument("--model_base",default=""); ap.add_argument("--manifest",default="")
    ap.add_argument("--heur_occ",default=""); ap.add_argument("--heur_psent",default="")
    ap.add_argument("--tree_tag",default=""); ap.add_argument("--out",default="")
    A=ap.parse_args()
    global MODEL_BASE, CFG_MANIFEST, HEUR_OCC, HEUR_PSENT, TREE_TAG, OUT
    if A.model_base: MODEL_BASE=A.model_base
    if A.manifest: CFG_MANIFEST=A.manifest if A.manifest.endswith(".json") else os.path.join("scenarios/manifests",A.manifest)
    if A.heur_occ: HEUR_OCC=A.heur_occ
    if A.heur_psent: HEUR_PSENT=A.heur_psent
    if A.tree_tag: TREE_TAG=A.tree_tag
    if A.out: OUT=os.path.join(REPO,"results/viper",A.out) if not os.path.isabs(A.out) else A.out
    os.makedirs(OUT,exist_ok=True)
    ep_csv=os.path.join(OUT,"episodes.csv"); hl_csv=os.path.join(OUT,"hospital_loads.csv")
    done=set()
    if os.path.exists(ep_csv):
        for r in csv.DictReader(open(ep_csv,encoding="utf-8")): done.add((r["region"],r["gate"],r["policy"]))
    jobs=[(r,g,p,A.n_ep) for g in A.gates.split(",") for p in A.policies.split(",") for r in REGIONS if (r,g,p) not in done]
    print(f"[simlog] jobs={len(jobs)} (done={len(done)}, n_ep={A.n_ep}, out={OUT})",flush=True)
    if not jobs: print("[simlog] 완료"); return
    ep_f=open(ep_csv,"a",newline="",encoding="utf-8"); hl_f=open(hl_csv,"a",newline="",encoding="utf-8")
    epw=hlw=None
    if os.path.getsize(ep_csv)==0: epw=None
    t0=time.time(); nok=nf=0
    with Pool(A.workers,maxtasksperchild=1) as pool:
        for k,r in enumerate(pool.imap_unordered(worker,jobs),1):
            if r["ok"]:
                if epw is None:
                    epw=csv.DictWriter(ep_f,fieldnames=list(r["ep_rows"][0].keys()));
                    if os.path.getsize(ep_csv)==0: epw.writeheader()
                    hlw=csv.DictWriter(hl_f,fieldnames=list(r["hosp_rows"][0].keys()))
                    if os.path.getsize(hl_csv)==0: hlw.writeheader()
                epw.writerows(r["ep_rows"]); hlw.writerows(r["hosp_rows"]); ep_f.flush(); hl_f.flush(); nok+=1
                eps=r["ep_rows"]; mg=np.mean([x["gini"] for x in eps]); mu=np.mean([x["n_used_hosp"] for x in eps])
                print(f"  [{k}/{len(jobs)}] {r['region']} {r['gate']} {r['policy']}: gini平{mg:.2f} 사용병원平{mu:.0f} ({time.time()-t0:.0f}s)",flush=True)
            else:
                nf+=1; print(f"  [{k}/{len(jobs)}] FAIL {r['region']} {r['gate']} {r['policy']}: {r['err'][:150]}",flush=True)
    ep_f.close(); hl_f.close()
    print(f"\n[simlog] 완료 ok={nok} fail={nf} wall={time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
