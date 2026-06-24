import glob, os, csv, json, statistics as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
RL="/home/ryu/MCI_UAV/results/rl"; BEST="/home/ryu/MCI_UAV/results/sigungu_heuristic_best.csv"; SIDO="/home/ryu/MCI_UAV/scenarios/manifests/sido"
rows=list(csv.DictReader(open(BEST,encoding='utf-8-sig'))); by_sig={r['sigcd']:float(r['reward_wog']) for r in rows}
nat_mean=st.mean(by_sig.values())
sido_base={}
for jp in glob.glob(f"{SIDO}/*.json"):
    s=os.path.splitext(os.path.basename(jp))[0]; man=json.load(open(jp,encoding='utf-8'))
    sido_base[s]=st.mean([by_sig[k.split('_')[-1]] for k in man if k.split('_')[-1] in by_sig])

def series(rd,tag):
    ev=sorted(glob.glob(f"{rd}/tb/ppo_feature_1/events*"))
    if not ev: return []
    a=EventAccumulator(ev[-1],size_guidance={'scalars':0}); a.Reload()
    return [(s.step,s.value) for s in a.Scalars(tag)] if tag in a.Tags().get('scalars',[]) else []
def wmean(ser,lo,hi):
    v=[x for s,x in ser if lo<=s<=hi]; return st.mean(v) if v else None
def slope(ser,frm):
    p=[(s,x) for s,x in ser if s>=frm]
    if len(p)<3: return None
    n=len(p);sx=sum(s for s,_ in p);sy=sum(x for _,x in p);sxx=sum(s*s for s,_ in p);sxy=sum(s*x for s,x in p);d=n*sxx-sx*sx
    return None if d==0 else (n*sxy-sx*sy)/d*1e6
def analyze(rd,base,tot,cohort,scope,region):
    rew=series(rd,'rollout/ep_rew_mean')
    if not rew: return {"region":region,"cohort":cohort,"scope":scope,"ok":False}
    last=rew[-1][0]
    final=wmean(rew,last*0.95,last); q3=wmean(rew,last*0.70,last*0.80)
    ev=wmean(series(rd,'train/explained_variance'),last*0.9,last)
    ent=wmean(series(rd,'train/entropy_loss'),last*0.9,last)
    vl=wmean(series(rd,'train/value_loss'),last*0.9,last)
    nz=st.pstdev([x for s,x in rew if s>=last*0.75]) if sum(1 for s,_ in rew if s>=last*0.75)>=5 else None
    return {"region":region,"cohort":cohort,"scope":scope,"ok":True,
            "target":tot,"last_step":last,"pct":round(100*last/tot,1),"n_pts":len(rew),
            "ep_rew_final":round(final,3) if final else None,"ep_rew_max":round(max(x for _,x in rew),3),
            "ep_rew_q3":round(q3,3) if q3 else None,
            "slope_per_M":round(slope(rew,last*0.75),3) if slope(rew,last*0.75) is not None else None,
            "noise":round(nz,3) if nz else None,
            "ev_final":round(ev,3) if ev is not None else None,"ent_final":round(ent,3) if ent is not None else None,
            "vloss_final":round(vl,3) if vl is not None else None,
            "heur_base":round(base,3),"margin":round(final-base,3) if final else None}
out=[]
for coh,tok in [("occ","occ"),("siteonly","siteonly")]:
    out.append(analyze(f"{RL}/sigungu_nat/ds_ess_woG_{tok}_s0",nat_mean,5_000_000,coh,"national","전국"))
    for s in sorted(sido_base):
        out.append(analyze(f"{RL}/sido/{s}_ds_ess_woG_{tok}_s0",sido_base[s],2_000_000,coh,"sido",s))
p="/tmp/claude-1002/-home-ryu-MCI-UAV/33461d9f-42e2-4552-b697-fae131f3e6db/scratchpad/review36.json"
json.dump(out,open(p,'w'),ensure_ascii=False,indent=1)
ok=sum(1 for r in out if r.get('ok')); print(f"추출 완료 {ok}/{len(out)}런 → {p}")
# 빠른 요약 출력
print(f"\n{'cohort':<9}{'region':<7}{'pct':>6}{'final':>8}{'heur':>7}{'margin':>8}{'EV':>6}{'slope':>7}")
for r in out:
    if not r.get('ok'): print(f"{r['cohort']:<9}{r['region']:<7}  데이터없음"); continue
    print(f"{r['cohort']:<9}{r['region']:<7}{r['pct']:>5.0f}%{r['ep_rew_final']:>8}{r['heur_base']:>7}{r['margin']:>+8}{r['ev_final']:>6}{r['slope_per_M']:>7}")
