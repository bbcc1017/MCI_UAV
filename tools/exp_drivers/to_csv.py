import json, csv
J="/tmp/claude-1002/-home-ryu-MCI-UAV/33461d9f-42e2-4552-b697-fae131f3e6db/scratchpad/review36.json"
data=json.load(open(J,encoding='utf-8'))
# 1) 런별 36행 통합 지표
cols=["cohort","scope","region","target","last_step","pct","n_pts",
      "ep_rew_final","ep_rew_max","ep_rew_q3","slope_per_M","noise",
      "ev_final","ent_final","vloss_final","heur_base","margin"]
p1="/home/ryu/MCI_UAV/results/rl_occ_siteonly_metrics.csv"
with open(p1,'w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader()
    for r in sorted(data,key=lambda x:(x['cohort'],x['scope']!='national',x['region'])):
        w.writerow(r)
# 2) occ↔siteonly 비교 18행 (region 페어)
idx={}
for r in data:
    idx.setdefault(r['region'],{})[r['cohort']]=r
p2="/home/ryu/MCI_UAV/results/rl_cost_of_comms.csv"
with open(p2,'w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(["region","scope","occ_margin","siteonly_margin","cost_of_comms","occ_ep_rew","siteonly_ep_rew","heur_base"])
    rows=[]
    for reg,d in idx.items():
        o=d.get('occ'); s=d.get('siteonly')
        if not(o and s and o.get('ok') and s.get('ok')): continue
        cost=round(o['margin']-s['margin'],3)
        rows.append([reg,o['scope'],o['margin'],s['margin'],cost,o['ep_rew_final'],s['ep_rew_final'],o['heur_base']])
    for row in sorted(rows,key=lambda x:-x[4]): w.writerow(row)
print("작성:")
print("  ",p1,f"({len(data)}행)")
print("  ",p2,f"({len(rows)}행)")
