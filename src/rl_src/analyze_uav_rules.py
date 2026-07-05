"""UAV 운용규칙 분석 (Phase 3-A) — 67만 결정 → 언제·누구를·어디로 UAV."""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import gzip, csv
import numpy as np
from collections import defaultdict

REPO="/home/ryu/MCI_UAV"
URBAN=set("서울 부산 대구 인천 광주 대전 울산 경기".split())
rows=[]
with gzip.open(f"{REPO}/results/rl/redesign/uav_decisions.csv.gz","rt",encoding="utf-8") as f:
    for r in csv.DictReader(f): rows.append(r)
print(f"총 {len(rows)} 결정\n")

def fnum(r,k): return float(r[k])
def inum(r,k): return int(r[k])

# 1) 대수별 UAV 사용률 + UAV 이송의 특성
print("=== 1) 대수별 UAV 사용률 및 UAV 이송 특성 ===")
print(f"{'UAV':>4} {'결정수':>8} {'UAV%':>6} {'UAV_tier3%':>10} {'AMB_tier3%':>10} {'UAV_etarank':>11} {'AMB_etarank':>11} {'UAV_etagap':>10}")
for lv in [0,5,10,15,26]:
    sub=[r for r in rows if inum(r,'level')==lv]
    if not sub: continue
    uav=[r for r in sub if inum(r,'mode')==1]; amb=[r for r in sub if inum(r,'mode')==0]
    def rate(s,k): return np.mean([inum(r,k) for r in s]) if s else 0
    def avg(s,k): return np.mean([fnum(r,k) for r in s]) if s else 0
    print(f"{lv:>4} {len(sub):>8} {100*len(uav)/len(sub):>5.1f}% "
          f"{100*rate(uav,'dest_tier3'):>9.0f}% {100*rate(amb,'dest_tier3'):>9.0f}% "
          f"{avg(uav,'eta_amb_rank'):>11.2f} {avg(amb,'eta_amb_rank'):>11.2f} {avg(uav,'eta_gap'):>+10.3f}")

# 2) 중증도별 UAV 사용률(레벨별)
print("\n=== 2) 중증도(cls)별 UAV 사용률 ===")
print(f"{'UAV':>4} {'Red_UAV%':>9} {'Yellow_UAV%':>11}")
for lv in [5,10,15,26]:
    sub=[r for r in rows if inum(r,'level')==lv]
    red=[r for r in sub if inum(r,'cls')==0]; yel=[r for r in sub if inum(r,'cls')==1]
    ru=np.mean([inum(r,'mode') for r in red]) if red else 0
    yu=np.mean([inum(r,'mode') for r in yel]) if yel else 0
    print(f"{lv:>4} {100*ru:>8.1f}% {100*yu:>10.1f}%")

# 3) 로지스틱 회귀: P(mode=UAV) ~ 피처 (uav>0만, 표준화 계수)
print("\n=== 3) '언제 UAV' 로지스틱 회귀 (uav5~26 통합, 표준화 계수, +=UAV 선택↑) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sub=[r for r in rows if inum(r,'level')>0]
feats=['eta_amb','eta_gap','dest_tier3','cls','eta_amb_rank','p_sent_dest','rho','n_red_wait','n_yellow_wait']
X=np.array([[fnum(r,k) for k in feats] for r in sub]); y=np.array([inum(r,'mode') for r in sub])
Xs=StandardScaler().fit_transform(X)
lr=LogisticRegression(max_iter=1000,C=1.0).fit(Xs,y)
order=np.argsort(-np.abs(lr.coef_[0]))
for i in order:
    print(f"  {feats[i]:>14}: {lr.coef_[0][i]:+.3f}")
print(f"  (UAV 선택률 {100*y.mean():.1f}%, 정확도 {lr.score(Xs,y):.3f})")

# 4) 도심 vs 농촌 UAV 사용률(uav26)
print("\n=== 4) 도심 vs 농촌 UAV 사용률 (uav26) ===")
for grp,name in [(URBAN,'도심'),(None,'농촌')]:
    sub=[r for r in rows if inum(r,'level')==26 and ((r['region'] in URBAN) == (name=='도심'))]
    if sub:
        print(f"  {name}: UAV {100*np.mean([inum(r,'mode') for r in sub]):.1f}% "
              f"(UAV이송 tier3율 {100*np.mean([inum(r,'dest_tier3') for r in sub if inum(r,'mode')==1] or [0]):.0f}%, "
              f"eta_rank {np.mean([fnum(r,'eta_amb_rank') for r in sub if inum(r,'mode')==1] or [0]):.2f})")

# 5) UAV 이송이 최근접이 아닌 경우(원거리 상급 이송) 비율
print("\n=== 5) UAV 이송의 목적지 선택 (uav26) ===")
sub=[r for r in rows if inum(r,'level')==26 and inum(r,'mode')==1]
if sub:
    far=[r for r in sub if inum(r,'eta_amb_rank')>=3]
    print(f"  UAV이송 {len(sub)}건: 최근접(rank1) {100*np.mean([inum(r,'eta_amb_rank')==1 for r in sub]):.0f}%, "
          f"3위이상 원거리 {100*len(far)/len(sub):.0f}%, tier3 {100*np.mean([inum(r,'dest_tier3') for r in sub]):.0f}%")
