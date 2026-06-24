import json, os, csv, yaml, sys
REPO="/home/ryu/MCI_UAV"
MANIFEST=sys.argv[1]; SUFFIX=sys.argv[2]; OUTPREFIX=sys.argv[3]; OUTDIR=sys.argv[4] if len(sys.argv)>4 else f"{REPO}/results"
def find_out(d,acc=[None]):
    if isinstance(d,dict):
        for k,v in d.items():
            if k=='output_path': acc[0]=v
            find_out(v,acc)
    return acc[0]
def stat_path(cfgp):
    outp=find_out(yaml.safe_load(open(cfgp,encoding='utf-8')),[None])
    coord=os.path.basename(os.path.dirname(cfgp))
    return os.path.join(REPO, outp.lstrip('./'), coord, f"results_{coord}{SUFFIX}_stat.txt")
def parse_stat(p):
    rows=[l for l in open(p,encoding='utf-8') if l.strip()]
    if len(rows)!=320: return None
    names=[];rew=[];pdr=[];rewog=[];pdrwog=[]
    for i,l in enumerate(rows):
        t=l.split(); mean=float(t[-3]); name=' '.join(t[:-3]); b=i//64
        if b==0: names.append(name);rew.append(mean)
        elif b==2: pdr.append(mean)
        elif b==3: rewog.append(mean)
        elif b==4: pdrwog.append(mean)
    return names,rew,pdr,rewog,pdrwog
def splitkey(k):
    if '_' in k:
        r,s=k.rsplit('_',1)
        if s.isdigit(): return r,s
    return k,''
m=json.load(open(MANIFEST,encoding='utf-8'))
best=[];full=[];miss=[]
for key,cfgp in m.items():
    region,sigcd=splitkey(key)
    coord=os.path.basename(os.path.dirname(cfgp)).strip('()'); lat,lon=coord.split(',')
    sp=stat_path(cfgp)
    if not os.path.exists(sp): miss.append(key); continue
    pr=parse_stat(sp)
    if not pr: miss.append(key+'(bad)'); continue
    names,rew,pdr,rewog,pdrwog=pr; bi=max(range(len(rew)),key=lambda i:rew[i])
    best.append([region,sigcd,lat,lon,names[bi],rew[bi],pdr[bi],rewog[bi],pdrwog[bi]])
    for i in range(len(names)): full.append([region,sigcd,names[i],rew[i],pdr[i],rewog[i],pdrwog[i]])
os.makedirs(OUTDIR,exist_ok=True)
bp=f"{OUTDIR}/{OUTPREFIX}{SUFFIX}_best.csv"; fp=f"{OUTDIR}/{OUTPREFIX}{SUFFIX}_full.csv"
csv.writer(open(bp,'w',encoding='utf-8-sig',newline='')).writerows([['region','sigcd','lat','lon','best_rule','reward','pdr','reward_wog','pdr_wog']]+best)
csv.writer(open(fp,'w',encoding='utf-8-sig',newline='')).writerows([['region','sigcd','best_rule','reward','pdr','reward_wog','pdr_wog']]+full)
print(f"  best={bp}({len(best)}) full=({len(full)}) missing={len(miss)} {miss[:5]}")
