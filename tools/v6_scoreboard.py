# -*- coding: utf-8 -*-
"""v6 사이클 마스터 집계 — 보고서용 스코어보드. 확정 CSV/JSON 전부 읽어 표 출력."""
import csv, json, os
import numpy as np
from collections import defaultdict

R = "/home/ryu/MCI_UAV/results/rl/redesign"
LB_SIDO, LB_HOLD, ORACLE = 0.1199, 0.1711, 0.06841

def judg(p, models):
    rows = list(csv.DictReader(open(p)))
    return {m: np.mean([float(r[f"PDR_{m}"]) for r in rows]) for m in models}, len(rows)

def wtl(p, a, b):  # PDR_a - PDR_b per region... judgment CSV엔 에피배열 없음 → region-mean 부호만
    rows = list(csv.DictReader(open(p)))
    d = np.array([float(r[f"PDR_{a}"]) - float(r[f"PDR_{b}"]) for r in rows])
    return int((d < 0).sum()), int((d > 0).sum())  # a<b 우세, a>b 열세

MS = ["v4plr2", "v6pad_s0", "v6pad_s1", "v6pad_s2", "heur", "lb_T4"]
print("="*78)
print("【Track A — 차원 비의존】 시도17 PDR_woG (낮을수록 우수, 판정전용 seed11000)")
for p, lab in [("v6_sido17_judgment.csv","고정47 1000ep"),
               ("v6_sido17_natural_judgment.csv","자연-H 1000ep")]:
    fp = f"{R}/{p}"
    if not os.path.exists(fp): print(f"  {lab}: 없음"); continue
    mean, n = judg(fp, MS)
    print(f"  [{lab}] " + " ".join(f"{m}={mean[m]:.4f}" for m in MS))

print("\n  holdout250 300ep:")
for p, lab in [("v6_holdout250_judgment.csv","고정47"),
               ("v6_holdout250_natural_judgment.csv","자연-H")]:
    fp = f"{R}/{p}"
    if not os.path.exists(fp): continue
    mean, n = judg(fp, MS)
    print(f"  [{lab} n={n}] " + " ".join(f"{m}={mean[m]:.4f}" for m in MS))

print("="*78)
print("【Track A5 — 규모·자원 랜덤화】 stress 스위트 PDR_woG")
MS5 = ["v4plr2","v6pad_s0","v6rand_s0","v6rand_s1","v6rand_s2","lb_T4"]
for p, lab in [("v6_rand_sido17_judgment.csv","표준 시도17 1000ep"),
               ("v6_stress_n200.csv","incident200 300ep"),
               ("v6_stress_capa05.csv","capa0.5 300ep"),
               ("v6_stress_uav5.csv","uav5 300ep")]:
    fp = f"{R}/{p}"
    if not os.path.exists(fp): continue
    mean, n = judg(fp, MS5)
    print(f"  [{lab}] " + " ".join(f"{m}={mean[m]:.4f}" for m in MS5))

print("="*78)
print("【Track B0 — NCRP 스케일링】 시군구40 CRN 30ep (Δ=base−planner, 높을수록 우수)")
def planner_summ(p):
    d = defaultdict(list)
    for r in csv.DictReader(open(p)):
        d[r["region"]].append((float(r["pdr_base"]), float(r["pdr_planner"]), float(r["ms_per_dec"])))
    dels, ms = [], []
    W=T=L=0
    for reg,v in d.items():
        a=np.array([x[0]-x[1] for x in v]); m=a.mean(); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
        W+=m>ci; L+=m<-ci; T+=(-ci<=m<=ci); dels.append(m); ms+=[x[2] for x in v]
    return np.mean(dels), (W,T,L), np.mean(ms), len(d)
for tag, f in [("K8h10m8(v5)","planner_tune2_h10_m8_e0.csv"),("K8h10m16","planner_tune3_K8h10m16.csv"),
               ("K8h20m8","planner_tune3_K8h20m8.csv"),("K8h20m16","planner_tune3_K8h20m16.csv"),
               ("K16h10m16","planner_tune3_K16h10m16.csv")]:
    fp=f"{R}/{f}"
    if not os.path.exists(fp): continue
    dm,(W,T,Lx),ms,ng = planner_summ(fp)
    print(f"  {tag:12s} 지역{ng} Δ={dm:+.4f} W/T/L={W}/{T}/{Lx} {ms:.0f}ms/dec")

print("="*78)
print("【Track B0 헤드라인 + A×B 통합】 시도17 100ep 도달률 사다리")
def stack(p, lab, LB):
    fp=f"{R}/{p}"
    if not os.path.exists(fp): print(f"  {lab}: 없음"); return
    d=defaultdict(list)
    for r in csv.DictReader(open(fp)):
        d[r["region"]].append((float(r["pdr_base"]),float(r["pdr_planner"])))
    base=np.mean([np.mean([x[0] for x in v]) for v in d.values()])
    plan=np.mean([np.mean([x[1] for x in v]) for v in d.values()])
    W=T=L=0
    for reg,v in d.items():
        a=np.array([x[0]-x[1] for x in v]); m=a.mean(); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
        W+=m>ci; L+=m<-ci; T+=(-ci<=m<=ci)
    rb=(LB-base)/(LB-ORACLE)*100; rp=(LB-plan)/(LB-ORACLE)*100
    print(f"  {lab}: greedy {base:.4f}({rb:.1f}%) → +NCRP-m16 {plan:.4f}({rp:.1f}%) Δ+{base-plan:.4f} W/T/L={W}/{T}/{L}")
stack("planner_sido17_h10m16.csv","챔피언 v4_plr2", LB_SIDO)
stack("planner_v6pad_sido17_h10m16.csv","v6_pad_s0(차원비의존)", LB_SIDO)
print(f"  [기준] LB-T4=0.1199(0%) · 완전천리안 오라클=0.0684(100%) · v5 m8스택=0.0862(65.4%)")

print("="*78)
print("【Track B1 — ExIt 게이트(no-go)】 NCRP 라벨 예측가능성")
jp=f"{R}/ncrp_probe_v6_m16.json"
if os.path.exists(jp):
    j=json.load(open(jp)); bc=j.get("bc_probe",{})
    print(f"  N={j['n_samples']} switch율={j['switch_rate_overall']:.3f} chance={j['chance_acc_overall']:.3f} agreement={j['greedy_label_agreement']:.3f}")
    print(f"  held-out acc: 전체 pre {bc['acc_overall_pre']:.3f}→post {bc['acc_overall_post']:.3f}")
    print(f"  switched한정 post={bc['acc_switched_post']:.3f}(chance {bc['chance_acc_switched_val']:.3f}, {bc['switched_acc_over_chance']:.2f}배) / non-switched post={bc['acc_nonswitched_post']:.3f}")
    print(f"  → v3 천리안 0.19 → v6 비천리안 {bc['acc_switched_post']:.3f}: 반응형 학습 불가 3중 확증")
