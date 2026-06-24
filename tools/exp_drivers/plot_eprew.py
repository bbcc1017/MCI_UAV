import glob, os, csv, json, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
REPO="/home/ryu/MCI_UAV"; RL=f"{REPO}/results/rl"; OUT=f"{REPO}/results/plots"; os.makedirs(OUT,exist_ok=True)
# ── 한글 폰트(NanumGothic) 명시 등록 ──
NF=os.path.expanduser("~/.fonts/NanumGothic-Regular.ttf")
fm.fontManager.addfont(NF)
plt.rcParams['font.family']='NanumGothic'; plt.rcParams['axes.unicode_minus']=False
def curve(rd):
    ev=sorted(glob.glob(f"{rd}/tb/ppo_feature_1/events*"))
    if not ev: return None,None
    a=EventAccumulator(ev[-1],size_guidance={'scalars':0}); a.Reload()
    if 'rollout/ep_rew_mean' not in a.Tags().get('scalars',[]): return None,None
    s=a.Scalars('rollout/ep_rew_mean'); return np.array([x.step for x in s]),np.array([x.value for x in s])
def ema(y,a=0.1):
    o=np.zeros_like(y); o[0]=y[0]
    for i in range(1,len(y)): o[i]=a*y[i]+(1-a)*o[i-1]
    return o
def lh(p): return {r['sigcd']:float(r['reward_wog']) for r in csv.DictReader(open(p,encoding='utf-8-sig'))}
occ_h=lh(f"{REPO}/results/sigungu_heuristic_best.csv"); psent_h=lh(f"{REPO}/results/sigungu_heuristic_psent_best.csv")
nat_occ=st.mean(occ_h.values()); nat_psent=st.mean(psent_h.values())
sido_occ={}; sido_psent={}
for jp in sorted(glob.glob(f"{REPO}/scenarios/manifests/sido/*.json")):
    s=os.path.splitext(os.path.basename(jp))[0]; man=json.load(open(jp,encoding='utf-8')); sigs=[k.split('_')[-1] for k in man]
    sido_occ[s]=st.mean(occ_h[g] for g in sigs if g in occ_h); sido_psent[s]=st.mean(psent_h[g] for g in sigs if g in psent_h)
SIDO=sorted(sido_occ)
def nat_fig(cohort,hv,hl,fname,title):
    x,y=curve(f"{RL}/sigungu_nat/ds_ess_woG_{cohort}_s0"); fig,ax=plt.subplots(figsize=(10,6))
    if x is not None:
        ax.plot(x,y,color='C0',alpha=0.25,lw=1); ax.plot(x,ema(y),color='C0',lw=2,label='RL ep_rew_mean (woG, EMA평활)')
    ax.axhline(hv,color='crimson',ls='--',lw=2,label=f'{hl} = {hv:.2f}')
    ax.set_xlabel('학습 step'); ax.set_ylabel('ep_rew_mean (woG)'); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}",dpi=110); plt.close(fig); print("저장:",fname)
def sido_fig(cohort,hmap,hl,fname,title):
    fig,axs=plt.subplots(5,4,figsize=(20,18)); axs=axs.flatten()
    for i,s in enumerate(SIDO):
        ax=axs[i]; x,y=curve(f"{RL}/sido/{s}_ds_ess_woG_{cohort}_s0")
        if x is not None: ax.plot(x,y,color='C0',alpha=0.25,lw=0.8); ax.plot(x,ema(y),color='C0',lw=1.6)
        ax.axhline(hmap[s],color='crimson',ls='--',lw=1.4)
        marg=(ema(y)[-1]-hmap[s]) if x is not None else 0
        ax.set_title(f"{s}  (휴리스틱 {hmap[s]:.1f}, 마진 {marg:+.2f})",fontsize=11)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    for j in range(len(SIDO),len(axs)): axs[j].axis('off')
    fig.suptitle(title,fontsize=16,y=0.997); fig.supxlabel('학습 step'); fig.supylabel('ep_rew_mean (woG)')
    fig.tight_layout(rect=[0,0,1,0.99]); fig.savefig(f"{OUT}/{fname}",dpi=100); plt.close(fig); print("저장:",fname)
nat_fig('occ',nat_occ,'occ-휴리스틱 평균(250)','eprew_sigungu_occ.png','OSRM 시군구 (전국, 250 랜덤샘플) — RL occ vs occ-휴리스틱')
nat_fig('siteonly',nat_psent,'psent-휴리스틱 평균(250)','eprew_sigungu_psent.png','OSRM 시군구 (전국, 250 랜덤샘플) — RL psent(현장한정) vs psent-휴리스틱')
sido_fig('occ',sido_occ,'occ-휴리스틱','eprew_sido_occ.png','OSRM 시도별 17개 모델 — RL occ vs occ-휴리스틱 (지역별)')
sido_fig('siteonly',sido_psent,'psent-휴리스틱','eprew_sido_psent.png','OSRM 시도별 17개 모델 — RL psent(현장한정) vs psent-휴리스틱 (지역별)')
print("완료(한글)")
