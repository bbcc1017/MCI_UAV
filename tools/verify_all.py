# -*- coding: utf-8 -*-
"""v5·v6 중간정리 문서용 — 전 수치 원본 CSV 재검증. 문서 표는 이 출력만 인용."""
import csv, json, os
import numpy as np
from collections import defaultdict
R = "/home/ryu/MCI_UAV/results/rl/redesign"
Z = "/home/ryu/MCI_UAV/results/rl/zoo"
LB_S, OR_S = 0.1199, 0.06841   # 시도17 LB-T4 / 완전천리안 오라클
LB_H = 0.1711                  # holdout250 LB-T4

def jmean(p, ms):
    rows = list(csv.DictReader(open(p)))
    return {m: np.mean([float(r[f"PDR_{m}"]) for r in rows]) for m in ms}, len(rows)

def planner(p, LB):
    d = defaultdict(list)
    for r in csv.DictReader(open(p)):
        d[r["region"]].append((float(r["pdr_base"]), float(r["pdr_planner"])))
    ne = sum(len(v) for v in d.values())
    base = np.mean([np.mean([x[0] for x in v]) for v in d.values()])
    plan = np.mean([np.mean([x[1] for x in v]) for v in d.values()])
    W=T=L=0
    for reg,v in d.items():
        a=np.array([x[0]-x[1] for x in v]); m=a.mean(); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
        W+=m>ci; L+=m<-ci; T+=(-ci<=m<=ci)
    return len(d), ne, base, plan, (W,T,L), (LB-base)/(LB-OR_S)*100, (LB-plan)/(LB-OR_S)*100

print("="*80)
print("【v5 저번주】")
print("- zoo 알고리즘 (시도17 PDR_woG, 다시드=평균):")
rows = list(csv.DictReader(open(f"{Z}/v5_sido17_judgment.csv")))
def colavg(cols):  # 여러 컬럼(시드)을 지역평균 후 시드평균
    per = [np.mean([float(r[c]) for r in rows]) for c in cols]
    return np.mean(per)
groups = {"heur":["PDR_heur"], "lb_T4":["PDR_lb_T4"],
          "dqn":["PDR_dqn_s0","PDR_dqn_s1","PDR_dqn_s2"],
          "qrdqn":["PDR_qrdqn_s0","PDR_qrdqn_s1","PDR_qrdqn_s2"],
          "reinforce":["PDR_reinf_s0","PDR_reinf_s1","PDR_reinf_s2"],
          "dqn_smdp":["PDR_dqn_smdp"],
          "v3_wide":["PDR_v3_wide_s0","PDR_v3_wide_s1","PDR_v3_wide_s2"],
          "v4_plr2":["PDR_v4_plr2_s0","PDR_v4_plr2_s1","PDR_v4_plr2_s2"]}
for k,cols in groups.items(): print(f"    {k:10s}={colavg(cols):.4f}")
srows = list(csv.DictReader(open(f"{Z}/v5_sido17_sacd.csv")))
sac = np.mean([np.mean([float(r[c]) for r in srows]) for c in ["PDR_sacd_s0","PDR_sacd_s1","PDR_sacd_s2"]])
print(f"    {'sacd':10s}={sac:.4f}")
print("- NCRP m8 (v5 채택):")
for p,lab,LB in [(f"{R}/planner_sido17_h10m8.csv","시도17",LB_S),(f"{R}/planner_holdout250_h10m8.csv","holdout250",LB_H)]:
    ng,ne,b,pl,(W,T,L),rb,rp = planner(p,LB)
    print(f"    {lab}: 지역{ng} ep{ne} greedy {b:.4f} → 스택 {pl:.4f} Δ+{b-pl:.4f} W/T/L={W}/{T}/{L}"
          + (f" (도달 {rb:.1f}%→{rp:.1f}%)" if LB==LB_S else ""))

print("="*80)
print("【v6 이번주 — Track A 차원비의존】 시도17 1000ep")
for p,lab in [(f"{R}/v6_sido17_judgment.csv","고정47"),(f"{R}/v6_sido17_natural_judgment.csv","자연-H")]:
    m,n = jmean(p, ["v4plr2","v6pad_s0","v6pad_s1","v6pad_s2","heur","lb_T4"])
    print(f"  [{lab} n={n}] " + " ".join(f"{k}={m[k]:.4f}" for k in ["v4plr2","v6pad_s0","v6pad_s1","v6pad_s2","lb_T4"]))
print("  holdout250 300ep:")
for p,lab in [(f"{R}/v6_holdout250_judgment.csv","고정47"),(f"{R}/v6_holdout250_natural_judgment.csv","자연-H")]:
    m,n = jmean(p, ["v4plr2","v6pad_s0","v6pad_s1","v6pad_s2","lb_T4"])
    print(f"  [{lab} n={n}] " + " ".join(f"{k}={m[k]:.4f}" for k in ["v4plr2","v6pad_s0","v6pad_s1","v6pad_s2","lb_T4"]))

print("="*80)
print("【v6 — Track A5 랜덤화 stress】")
MS=["v4plr2","v6pad_s0","v6rand_s0","v6rand_s1","v6rand_s2","lb_T4"]
for p,lab in [(f"{R}/v6_rand_sido17_judgment.csv","표준 시도17"),(f"{R}/v6_stress_n200.csv","incident200"),
              (f"{R}/v6_stress_capa05.csv","capa0.5"),(f"{R}/v6_stress_uav5.csv","uav5")]:
    m,n=jmean(p,MS); print(f"  [{lab} n={n}] " + " ".join(f"{k}={m[k]:.4f}" for k in MS))

print("="*80)
print("【v6 — Track B0 NCRP 스케일링】 시군구40 CRN 30ep")
def psumm(p):
    d=defaultdict(list)
    for r in csv.DictReader(open(p)):
        d[r["region"]].append((float(r["pdr_base"]),float(r["pdr_planner"]),float(r["ms_per_dec"])))
    de=[np.mean([x[0]-x[1] for x in v]) for v in d.values()]; ms=[x[2] for v in d.values() for x in v]
    W=T=L=0
    for v in d.values():
        a=np.array([x[0]-x[1] for x in v]); mm=a.mean(); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
        W+=mm>ci; L+=mm<-ci; T+=(-ci<=mm<=ci)
    return np.mean(de),(W,T,L),np.mean(ms),len(d)
for tag,f in [("K8h10m8(v5)","planner_tune2_h10_m8_e0.csv"),("K8h10m16★","planner_tune3_K8h10m16.csv"),
              ("K8h20m8","planner_tune3_K8h20m8.csv"),("K8h20m16","planner_tune3_K8h20m16.csv"),
              ("K16h10m16","planner_tune3_K16h10m16.csv")]:
    fp=f"{R}/{f}"
    if not os.path.exists(fp): continue
    dm,(W,T,L),ms,ng=psumm(fp); print(f"  {tag:12s} 지역{ng} Δ={dm:+.4f} W/T/L={W}/{T}/{L} {ms:.0f}ms/dec")
print("  헤드라인 시도17 100ep + A×B:")
for p,lab in [(f"{R}/planner_sido17_h10m16.csv","챔피언+m16"),(f"{R}/planner_v6pad_sido17_h10m16.csv","v6_pad+m16")]:
    ng,ne,b,pl,(W,T,L),rb,rp=planner(p,LB_S)
    print(f"    {lab}: greedy {b:.4f}({rb:.1f}%) → {pl:.4f}({rp:.1f}%) Δ+{b-pl:.4f} W/T/L={W}/{T}/{L}")
print("  홀드아웃 30ep:")
for p,lab,LB in [(f"{R}/planner_v6pad_holdout250nat_h10m16.csv","v6_pad+m16 자연H",LB_H),
                 (f"{R}/planner_champ_holdout250_h10m16.csv","챔피언+m16 고정47",LB_H)]:
    ng,ne,b,pl,(W,T,L),rb,rp=planner(p,LB)
    print(f"    {lab}: 지역{ng} ep{ne} greedy {b:.4f} → {pl:.4f} Δ+{b-pl:.4f} W/T/L={W}/{T}/{L}")

print("="*80)
print("【v6 — Track B1 ExIt 게이트】")
j=json.load(open(f"{R}/ncrp_probe_v6_m16.json")); bc=j["bc_probe"]
print(f"  N={j['n_samples']} switch율={j['switch_rate_overall']:.3f} 평균유효행동={j['mean_n_valid']:.1f}")
print(f"  held-out: 전체 {bc['acc_overall_pre']:.3f}→{bc['acc_overall_post']:.3f} | switched {bc['acc_switched_post']:.3f}(chance {bc['chance_acc_switched_val']:.3f}={bc['switched_acc_over_chance']:.2f}배) | non-sw {bc['acc_nonswitched_post']:.3f}")

print("="*80)
print("【오라클 손실분해 시도17】")
for p,lab in [(f"{R}/oracle_headroom_sido17_v4.csv","v4 오라클(greedy vs 완전천리안)")]:
    rows=list(csv.DictReader(open(p)))
    cols=rows[0].keys()
    # region별 평균 후 전체 평균(ep 여러개)
    db=defaultdict(list); do=defaultdict(list)
    for r in rows: db[r["region"]].append(float(r["pdr_base"])); do[r["region"]].append(float(r["pdr_oracle"]))
    b=np.mean([np.mean(v) for v in db.values()]); o=np.mean([np.mean(v) for v in do.values()])
    print(f"  {lab}: greedy {b:.4f}({(LB_S-b)/(LB_S-o)*100:.1f}%) 완전천리안 {o:.4f}(100%)")
for p,lab in [(f"{R}/planner_sido17_h10_clair.csv","h10 천리안")]:
    if os.path.exists(p):
        ng,ne,b,pl,wtl,rb,rp=planner(p,LB_S)
        print(f"  {lab}: {pl:.4f} (도달 {rp:.1f}%)")
