# MCI_UAV

대규모 사상자 사고(**MCI**, Mass-Casualty Incident)에서 **구급차(AMB) + 드론(UAV)** 혼합 자원의
환자 triage·이송 의사결정을 **강화학습(RL)** 으로 학습하고, RL을 **① 도달가능 성능의 증인 +
② 운용규칙 발견 엔진**으로 사용해 **UAV 도입의 효용가치를 정량화**하고 해석가능 규칙을 추출하는 연구용 코드.

- **시뮬레이터**: MCI_ADV 기반 이산사건 시뮬레이터(`src/sim_src/`) — 환자 구조·이송·병원 처치를
  event-driven 으로 모델링하고, 보상 = 병원 도착(처치 시작) 시점의 생존확률.
- **에이전트**: MaskablePPO(주력 — essential+load obs + 포인터 head + PPO 위생 + pdrwog 보상),
  DQN/REINFORCE 비교. baseline = **64종 휴리스틱 룰** + **부하균형(발송상한) 규칙 LB-T4/적응T**(비교 기준선).
- **일반화**: 단일 좌표 → 17개 광역시도 → **시군구 250 전국 단일정책** → hold-out 무작위 좌표 평가.
- **현재 상태 (2026-07-05)**: 재설계 v2 완주 — **RL이 최강 규칙(LB-T4)을 역전**(시도17 16/17, PDR 27.7%↓),
  **UAV 한계가치 곡선**("첫 5대가 이득의 75%")과 **해석 산출물 3종**(운용규칙·프로그램·T-메타) 확보.
  다음 = **논문 라이팅** + 이월 검증(psent·Kakao·시드3).

> Unity 디지털트윈 시각화(C#)는 같은 시나리오 데이터를 3D 한국 지도에 렌더링한다. 본 저장소(`origin`)는
> RL/시뮬 연구 코드(Python)만 버전관리하며, Unity 자산은 로컬(Windows)에만 존재한다 → [§ Unity 디지털트윈](#unity).

---

<a id="toc"></a>
## 목차

1. [현재 상태 & 연구 여정](#status)
2. [환경 셋업](#setup)
3. [실험 파이프라인 (메인 경로)](#pipeline)
   - [시나리오 생성](#pl-scenario) · [휴리스틱 baseline](#pl-heur) · [RL 학습](#pl-train) · [VIPER 증류](#pl-viper) · [평가](#pl-eval)
4. [핵심 설계](#design) — obs/action/reward · 마스킹 · 파일계약 · H_max floor · 특징 obs · 부하균형 규칙 · 재설계 v2 · VIPER
5. [환경변수 레퍼런스](#envvars)
6. [멀티리전 / 전국 RL](#multiregion)
7. [저장소 구조](#layout)
8. [Unity 디지털트윈](#unity)

---

<a id="status"></a>
## 1. 현재 상태 & 연구 여정

> **진행 현황 (2026-07-05, 재설계 v2 Phase 0~3 완주)**: 진단("RL이 자기 로그에서 나온 발송상한 규칙에게
> 지는" 원인 = obs 신호 결손·dest 랭킹 구조 부재·PPO 위생)을 L사다리로 교정 → **L3(essential+load obs
> 355 + 포인터 head + 위생 + pdrwog)가 최강 규칙 LB-T4를 시도17 16/17·holdout 신좌표 250점 212승으로
> 역전**. 그 위에서 **UAV 한계가치 곡선**(0→26대: 첫 5대가 총이득 75%, 10~15대 포화, 예방가능 사망
> 12.7%→9.2%)과 **해석 산출물 3종**(UAV 운용규칙 4종 / 해석 프로그램 = 격차 13% 회수 / T-메타 RL =
> 30% 회수 + 규칙 T=f(ρ,병원밀도)) 확보. **전 과정 서사는 `docs/연구흐름_종합_2026-07-06.md`**(로컬)
> 참조 — 단계별 보고서 6편을 하나의 이야기로 종합한 문서.

| 시기 | 단계 | 결과 |
|---|---|---|
| 2026-06 | **v1 재설계 3-Phase**(시나리오 I/O 통합 · H_max floor · 특징 obs · VIPER 증류) + 실규모 36런 | ✅ RL > 64룰(+1~2), 통신축(occ↔psent) 비용 ≈ 0 (구조적 null) |
| 2026-06 말 | 시뮬로그 분석 · 자원 트레이드오프 스윕 | ✅ ★**발송상한(LB) 규칙이 RL을 능가** / 격차 작은 원인 = 저스트레스 regime(용량이 부하의 16배) 규명 |
| 2026-07-02 | **방향성 재점검** + 기반 리셋(성남 헬기장 정정 → **47병원·헬기장 26** 전 좌표 재구축) | ✅ 진단 3건(obs p_sent 부재·flat dest 분류·PPO kl 0.28), 비교 기준선 → 적응T-LB로 이동 |
| 2026-07-03 | action **Green 차원 제거(192=2×48×2)** · 통신축 재정의 · sim 정합성 4수정 · 휴리 재베이스라인 | ✅ 신 기반 확정(구 산출물은 `results_archived_20260702_헬기장정정이전/` — 신구 혼용 금지) |
| 2026-07-04~05 | ★**재설계 v2 Phase 0~3** (플랜: "RL로 UAV 도입 효용가치 최적화", N=100 고정·occ·OSRM) | ✅ **L사다리 역전 · UAV 곡선 · 해석 3종** (위 blockquote) |
| 다음 | 논문 패키징 / 이월 검증(psent·Kakao·시드3) / D1 데이터 확장 / 농촌 가드 규칙 | ⏳ 착수 전 설계 합의 |

<details>
<summary><b>재설계 v2 핵심 수치 (시도17 paired 1000ep, PDR_woG↓)</b></summary>

```
최근접 휴리 0.2340 → L0(현행 재현) 0.1650 → L1(+PPO 위생) 0.1595 → [LB-T4 0.1199]
  → L2(+부하 obs) 0.1086 → L3(+포인터) 0.0923      ← RL이 최강 규칙 역전 (16/17)
사다리 기여: 위생 +0.006(미미) · 부하 obs +0.051(17/0 ★단일 최대) · 포인터 +0.016(17/0)
해석 스펙트럼(LB-T4→RL 격차 회수율): 고정 프로그램 −7% < 지역튜닝 프로그램 13% < T-메타 30% < full-RL 100%
UAV 곡선(RL): 0대 0.1268 → 5대 0.1008 → 10대 0.0971 → 15대 0.0948 → 26대 0.0923 (첫 5대 = 총이득 75%)
```
</details>

[↑ 목차](#toc)

---

<a id="setup"></a>
## 2. 환경 셋업

- Python 3.10 (conda env **`UAV`**, Windows 로컬·Linux 학습박스 공통). torch 2.8.0+cu128(RTX 50/A6000 공통 현행).
- SB3 + sb3-contrib(MaskablePPO), gymnasium <1, numpy <2, scikit-learn(VIPER/증류).

```bash
conda activate UAV
pip install -r requirements.txt
# torch 는 GPU 에 맞춰 별도 설치 (requirements.txt 주석 참고)
```

라우팅 백엔드(시나리오 생성용):

```bash
# (A) OSRM — 기본, 결정적, 교통 미반영. 대량 평가는 로컬 컨테이너 권장 (탐구 단계 표준)
tools/osrm_prepare_korea.sh
docker compose -f docker-compose.osrm.yml up -d
export MCI_OSRM_URL=http://localhost:5000
# docker 그룹 권한이 없는 세션에서는 `sudo -n docker compose -f docker-compose.osrm.yml up -d`

# 병원-병원 원본 도로거리 행렬 재생성(엑셀 결합 데이터.xlsx 기준, 페리 포함 OSRM 경로 사용)
tools/build_distance_matrix_osrm.py

# (B) Kakao Mobility — 출발시각 교통 반영(논문 단계 검증용). 키는 ENV 로만
export KAKAO_API_KEY=<your_key>
```

> 작업 분담: **RL/시뮬 학습은 SSH(Linux, `aigpu0617`)**, **Unity 디지털트윈은 로컬(Windows)**.
> 학습 박스(A6000)에 `UAV` env 가 셋업되어 PPO 학습·VIPER 증류까지 검증 가능.

[↑ 목차](#toc)

---

<a id="pipeline"></a>
## 3. 실험 파이프라인 (메인 경로)

데이터 흐름: **시나리오 YAML → gym env → 래퍼 → 학습/증류/평가.** 휴리스틱·RL·트리는 **동일 시나리오 파일**을
공유해 비교 일치성을 보장한다(모든 거리/시간은 생성 시 사전계산·동결 — sim-time API 호출 금지).

**고정 시나리오 표준 파라미터 (2026-07-02 성남 헬기장 정정 이후)**: 병원 `fixed_hos_num 47` ·
**헬기장 26 → `uav_num 26`** · `amb_count 30` · `incident_size 100` · 속도 50/200 · 핸드오버 5/10.
⚠️ uav_num 을 바꾸면 병원 집합이 달라질 수 있으니(superset 생성 규칙) 표준 세트는 26 고정 — 대수 실험은
런타임 노브 `MCI_UAV_NUM`으로. 세트 4부류 = {시군구 250, 시도 17} × {OSRM, Kakao}
(`scenarios/manifests/{sigungu_osrm,sigungu_kakao,sido_osrm,plan1}_manifest.json`).

<a id="pl-scenario"></a>
### 3.1 시나리오 생성

**17개 광역시도 일괄 (표준 파라미터):**

```bash
python src/sce_src/gen_regions.py \
  --incident_size 100 --amb_count 30 --uav_count 26 --uav_num 26 \
  --fixed_hos_num 47 --road_mode osrm --exp_prefix 시도osrm
# Kakao 짝: --road_mode kakao --departure_time 202607301400 (동일 좌표·병원선정, 도로행렬만 교체)
# → scenarios/exp_*/(lat,lon)/config_*.yaml + scenarios/manifests/<prefix>_manifest.json
```

- `--min_hos_mode auto`: **Pass1**(road API 0회)으로 자연 선정 병원수의 `H_max` 산출 → **Pass2** floor-up
  (차원 통일 장치, [자세히](#dsn-hmax)). 현행 표준 세트는 `--fixed_hos_num 47` 고정.
- `--amb_count`=AMB 런타임수, `--uav_count`=UAV 생성 superset 상한(헬기장 병원당 1대), `--uav_num`=UAV 런타임수(load 시 슬라이스).

**단일 좌표:**

```bash
python src/sce_src/make_csv_yaml_dynamic.py --base_path . \
  --experiment_id mix_seoul --latitude 37.5666 --longitude 126.9784 \
  --incident_size 100 --amb_count 30 --uav_count 26 --is_use_time True --kakao_api_key $KAKAO_API_KEY
```

**시군구 250 / hold-out 세트 (재구축 도구):**

```bash
# 시군구 Kakao 일괄(재개가능·키 로테이션): src/sce_src/gen_sigungu_kakao.py --keys_file <keys>
# 시군구 OSRM 좌표고정 재구축: src/sce_src/regen_sigungu_osrm.py
# hold-out 평가점(OSRM, 기존 좌표 재사용/신규): src/sce_src/gen_eval_holdout_osrm.py [--points_from <json>]
# (구식 Kakao hold-out 경로: sample_region_points.py → gen_eval_points.py --min_hos_num <H>)
```

<a id="pl-heur"></a>
### 3.2 휴리스틱 baseline

```bash
# 단일 시나리오 — 64룰 전수 시뮬
python src/sim_src/main.py --config_path "scenarios/exp_*/(lat,lon)/config_*.yaml"
# results/<EXP_ID>/results_*.txt, results_*_stat.txt  (64룰 × 5지표그룹 = 320행)

# 매니페스트 일괄(64룰×1000ep, occ/psent 게이트) + 집계 → results/<prefix>[_psent]_best.csv
python tools/exp_drivers/run_heur_batch.py <manifest.json> occ <병렬수>   # ★OMP/MKL/OPENBLAS=1 핀 필수
python tools/exp_drivers/aggregate_heur.py <manifest.json> "" <prefix>
```

**64룰** = START/ReSTART × RedOnly/YellowNearest × Red 수단 4 × Yellow 수단 4 (2×2×4×4).
**부하균형(발송상한) 규칙**(64룰보다 강한 기준선)은 `src/rl_src/loadbalance_heuristic.py` — [§4 설계](#dsn-lb).

<a id="pl-train"></a>
### 3.3 RL 학습

**현행 주경로 — 재설계 v2 최종 채택 구성(L3)**: essential+load obs + 포인터 head + PPO 위생 + pdrwog 보상,
시군구 250 매니페스트 전국 단일 10M.

```bash
MCI_OBS_VARIANT=essential+load MCI_CAP_GATE=occ \
python src/rl_src/train_ppo_feature.py \
  --config_path scenarios/manifests/sigungu_osrm_manifest.json \
  --extractor pointer --reward_mode pdrwog --norm_reward \
  --lr_anneal --target_kl 0.03 --batch_size 512 --n_epochs 5 \
  --n_envs 8 --vec subproc --total_timesteps 10000000 --seed 0 \
  --log_dir results/rl/redesign/L3_pointer_s0
# TensorBoard: tensorboard --logdir <log_dir>/tb
```

- `--extractor mlp | deepsets | pointer` — pointer 는 병원 랭킹 head(순열등변). ⚠️ **uav_num=0(action 96)은
  pointer 미지원 → deepsets** 사용.
- UAV 대수별 개별 모델(한계가치 곡선용): 앞에 `MCI_UAV_NUM=k` 만 바꿔 동일 명령 반복.
- T-메타(발송상한 T 만 학습): `train_ppo_tmeta.py` (시군구 250, 5M).
- 하이퍼파라미터 근거는 `docs/RL_재설계_설계노트_2026-07-04.md`(무지성 금지 원칙).

> 비교용 레거시 경로: `run_all_parallel.py`(DQN+PPO+REINFORCE 동시), `train_ppo.py`(인덱스 obs),
> `train_{dqn,reinforce}.py`.

<a id="pl-viper"></a>
### 3.4 VIPER 트리 증류

```bash
MCI_REDUCED_OBS=1 MCI_OBS_VARIANT=essential MCI_CAP_GATE=occ \
python src/rl_src/viper_distill.py \
  --manifest scenarios/manifests/sigungu_osrm_manifest.json \
  --model <model.zip> \
  --n_iter 5 --rollout_eps 10 --max_depth 8 --crit loggap \
  --heur_csv results/sigungu_heuristic_best.csv --out_dir results/viper
# → results/viper/viper_loggap_d8.pkl + _rules.txt + _region_eval.csv
```

반복 Q-DAGGER 로 오라클 PPO 를 결정트리로 증류 → PPO·휴리스틱 대비 **margin 유지율** 보고. [자세히](#dsn-viper)
학습이 VecNormalize 였으면 통계 동결 로드(`--vecnorm`, 미지정 시 model 디렉터리 자동탐색).
⚠️ env var 는 **학습 당시와 동일하게**(`MCI_OBS_VARIANT`/`MCI_CAP_GATE`; psent 면 `MCI_CARED_OBS=0` 짝).
구 `MCI_GREEN_MASK` 는 2026-07-03 폐기(Green 이 action 차원에서 제거됨).

<a id="pl-eval"></a>
### 3.5 평가

**재설계 v2 paired 드라이버(주경로)** — 전부 시드 11000·1000ep·per-episode PDR_woG, 사용례는 각 docstring:

```bash
python src/rl_src/paired_eval_ladder.py --n_eps 1000 --workers 17   # L사다리 vs 적응T-LB·LB-T4·휴리best
python src/rl_src/uav_curve_eval.py     --n_eps 1000 --workers 34   # UAV 대수별 한계가치 곡선
python src/rl_src/program_eval.py --combos "4:0.8:0:0" --with_rl --n_eps 1000  # 해석 프로그램
python src/rl_src/tmeta_eval.py         --n_eps 1000 --workers 17   # T-메타
# 부하균형 규칙 검증: lb_validate17.py / lb_validate_sigungu.py → results/viper/lb_paired_*.csv
```

**범용/레거시:**

```bash
# (a) 단일 좌표 — RL vs 휴리스틱
python src/rl_src/evaluate.py --config_path <yaml> --ppo_path <model.zip> --include_heuristic --n_episodes 100
# (b) 17개 광역 diagonal (지역별 모델 vs 자기 지역 휴리스틱 최선)
python src/rl_src/run_grid_eval.py --manifest scenarios/manifests/plan1_manifest.json \
  --model_root results/rl/plan1 --n_episodes 1000
# (c) 정보수준/변형 obs 평가: eval_obs_variant.py  (d) 하이브리드(AMB=휴리, UAV=RL): hybrid_eval.py
# (e) 자원·부하 트레이드오프: tradeoff_sweep.py / tradeoff_simlog.py (MCI_INCIDENT_SIZE 등 노브 스윕)
```

⚠️ 평가 코드에서 `MaskablePPO.load` 전에 `from pointer_policy import …`(pointer 모델) 또는
`from hospital_set_extractor import …`(deepsets 모델) import 필수 — 누락 시 로드 실패.

[↑ 목차](#toc)

---

<a id="design"></a>
## 4. 핵심 설계

<details>
<summary><b>관측(obs) / 행동(action) / 보상(reward) 인코딩</b></summary>

- **Obs (dict)** — `p_states (N,5)`=[class, rescued, move, moved, cared]; `h_states (H,3)`=[idle, queue, occupied];
  `p_sent (H,)`; `amb_states`/`uav_states (n,3)`=[dest, time_remaining, severity]; `p_at_site (4,)`=[R/Y/G/B 현장];
  `n_amb_at_site`, `n_uav_at_site`, `time`. → RL 래퍼가 1D float32 로 평탄화(주력 학습은 `MCI_REDUCED_OBS=1`
  + `MCI_OBS_VARIANT` 특징 obs 사용, 아래 블록).
- **Action `[class, dest, mode]`** — class 0=Red/1=Yellow (**2026-07-03 Green 을 action 차원에서 제거** —
  G/B 는 sim 코어가 일괄이송); dest 0=현장대기, 1..H=병원; mode 0=AMB/1=UAV.
  단일 차종일 때 mode 자동 고정(`amb_num=0`→UAV, `uav_num=0`→AMB). 래퍼가 `MultiDiscrete([2,H+1,2])` ↔
  `Discrete` 로 encode/decode — **H=47 표준에서 flat 192 = 2×48×2** (uav=0 은 96; 2026-07-02 이전
  구 시나리오는 282=3×47×2/H=46 — 모델 비호환).
- **Reward** — 환자 admit(처치 시작) 시점의 생존확률. Red/Yellow 는 시간 감쇠, Green=1, Black=0.
  `reward_redesign_wrapper.py` 가 `raw` | `woG`(Green 제외) | `rywt`(R/Y 가중) | **`pdrwog`**(=r_woG/
  preventable_woG, 규모 불변 0~1 — **현행 학습 표준**, `--norm_reward` 병용)로 재구성
  (`--reward_mode` 또는 `MCI_REWARD_MODE`).
- **평가 지표** — woG(절대) + **PDR_woG = 1 − woG/preventable_woG**(예방가능 사망률, 낮을수록 좋음,
  규모 불변 → 지역/규모 비교 표준). paired 체계: 시드 11000·1000ep·per-episode 비교.
</details>

<details>
<summary><b>행동 마스킹 (하드 제약) · 통신축 게이트</b></summary>

마스킹은 패널티가 아니라 **하드 제약**(`action_masks()`):
- Red → Tier3(상급종합) 병원만 (`MCI_TIER_MASK=0` 로 비활성).
- UAV(mode=1) → 헬기장 보유 병원만.
- dest 는 해당 class 환자 존재 ∧ 해당 mode 자원 존재 ∧ **용량 게이트 통과** 시만 허용. stay(dest=0)는 항상 허용.

**용량 게이트 = 통신축**(`MCI_CAP_GATE`): `occ`(기본, 통신 가용) = 입원 census + **이송중(in-flight)** <
max_send; `psent`(통신 단절) = 현장이 보낸 누적 `p_sent` < max_send + obs 도 병원 실시간 열 blackout
(`MCI_CARED_OBS=0` 짝 설정). **발송 게이트·obs·RL 마스크·휴리스틱 4곳이 같은 정의 공유**(쌍비교 불변식).
⚠️ 현행 시나리오는 총용량이 부하의 6~15배라 게이트가 거의 안 걸림 → 통신축 비용 ≈ 0 (확립된 null).

RL 경로는 **결합 마스크**(`action_masks_joint`)만 사용. 마스크 우회 방어로 `EventManager.proceed_action()`
에 `NO HELIPAD`/`NO PATIENT` 가드. 래퍼 체인(외→내): `Monitor → ActionMasker → [HeuristicAdvantage] →
HospitalFeatureWrapper(주력. 또는 FlattenAndDiscrete/Hybrid) → [RewardRedesign] → base env`.
**코어 파일은 무수정, 변형은 래핑으로.**
</details>

<details>
<summary><b id="dsn-contract">시나리오 파일 계약 (v1 Phase 1)</b></summary>

지역 폴더 `scenarios/exp_<prefix>_<region>_dep_<ts>/(lat,lon)/` 산출물:
- `hospital_info.csv` — 통합 병원 메타+현장거리: 요양기관명/종별코드/헬기장 여부/수술실수/병상수/`euc_dist`/`road_dist`/`road_duration`. **도로소요시간 오름차순 = 병원 인덱스**.
- `amb_station_info.csv` — 고유 안전센터/소방서당 1행(`보유대수`=count). load 시 보유대수만큼 `np.repeat` 전개 후 `amb_num` 만큼 슬라이스 → 같은 센터 동일거리 차량(Kakao road API 호출 절감).
- `uav_info.csv` — 헬기장 병원 superset(가까운 순, `hospital_idx` 보존). load 시 `uav_num` 슬라이스(**출발지만** — 착륙 가능 병원은 헬기장 전체 유지).
- `distance_Hos2Hos_{euc,road}.csv` — diversion 용 병원간 거리행렬(유지). `patient_info.csv`, `config_*.yaml`, `scene.json`(Unity 용).

**매핑**: `수술실수→hos_max_capa`, `병상수→hos_max_queue`, `종별코드 1→Tier3·그외→Tier2`, `헬기장 여부→helipad_idx`.
입원정원 `max_capa = 수술실수 + 병상수`, 발송상한 `max_send = max_capa`([1,1] 계수). 로더는 구포맷도 폴백 지원.
</details>

<details>
<summary><b id="dsn-hmax">데이터 기반 H_max floor (v1 Phase 2)</b></summary>

지역마다 자연 선정 병원수가 달라 Top-K obs 후보가 부족해지는 문제를, **2-pass floor** 로 해결:
- **Pass1**(road API 0회, Kakao 키 불필요): 각 대표점에 용량/tier 선정 로직만 적용해 자연 선정수 산출 → `H_max = min(max(natural), min(pool))`.
- **Pass2**: 모든 지역을 `min_hos_num=H_max` 로 floor-up(보장룰 자동 보존, cap-down 안 함).

병원 선정 공식: `eff = 수술실수 + 응급실병상수×(1−tier별 가동률)`, 누적 `eff ≥ 환자수×buffer_ratio(1.5)` 까지 가까운 순 +
tier/헬기장 보장룰. `fixed_hos_num`(cap)과 `min_hos_num`(floor)은 상호배타. 제주(섬)는 Kakao 페리 경로
지원으로 본토 병원 섞여도 정상. **현행 표준 세트는 fixed_hos_num=47**(2026-07-02 성남 헬기장 정정 —
"자원 추가 원칙"으로 46→47병원·25→26헬기장, 구 46 집합 전부 보존+성남 추가).
</details>

<details>
<summary><b id="dsn-feature">특징기반 병원 obs — local/comms/full → essential → essential+load (현행)</b></summary>

인덱스 기반 `h_states`/`p_sent` 대신 **병원당 특징 엔티티 행렬 (H, F)** (`HospitalFeatureWrapper`,
`MCI_OBS_VARIANT` 토글, train/eval 동일 필수):

- 초기 3종(정보수준 ablation): `local`(정적: tier3/헬기장/ETA) / `comms`(실시간: idle/queue/occ/cap) / `full`(8열).
- `essential`(v1 말기, dim 209): 병원당 4열 `[is_tier3, cap_remain, eta_amb, eta_uav]` + 글로벌 21.
- **`essential+load`(재설계 v2, 현행 표준, dim 355 = 47×7+26)**: 병원당 **7열** `[is_tier3, cap_remain_c,
  eta_amb, eta_uav, p_sent_c, in_flight, occ_ratio]`(전부 클립 유계 — 스케일 앵커 해소) + 글로벌 26
  (+**ρ**=잔여부하/용량·amb/uav_avail_frac·uav_frac·t_norm). 설계 의도: **"이기는 규칙(LB)이 쓰는 신호를
  RL에게"** — L사다리에서 단일 최대 기여(+0.051, 17/0)로 확증. ⚠️앞 4열 semantics 는 essential 과 동일
  → 분석 코드는 `reshape(H,4)`→`reshape(H,7)` 만 고치면 됨.
- ETA = `amb/uav_HtoS_t[0]`(lognormal 평균 = API duration), 시나리오 최근접≈1 정규화(지역 스케일 제거).
- extractor: `mlp` / `deepsets`(순열등변 집합 인코더) / **`pointer`(현행 — 병원별 score 랭킹 head,
  `pointer_policy.py`**; flat categorical 재매개변수화라 MaskablePPO 분포·마스킹 무수정).
</details>

<details>
<summary><b id="dsn-lb">부하균형(발송상한) 규칙 — 비교 기준선 (`loadbalance_heuristic.py`)</b></summary>

v1 RL 시뮬로그 분석에서 발견: 64룰 휴리는 최근접 1~2곳에 점유 ~500%까지 과집중(gini 0.94) → 목적지만
**"적격(마스크 통과) 중 누적발송 `p_sent < T(=4)` 인 최속, 차면 다음"**(병원당 정원제)으로 교체한 규칙이
64룰은 물론 **v1 RL 까지 능가**했다(시군구 240~245/250). 이 사건이 재설계 v2 진단의 출발점.

- `make_cap_policy(rule, T)` = LB-T4 등 고정 T. `make_adaptive_cap_policy()` = 적응T(T=f(스트레스)) —
  N=100 현행 부하에선 LB-T4 와 동일 동작(surge 에서만 갈라짐). **현행 비교 기준선 = 적응T-LB(≡LB-T4)**.
- 코덱은 `_codec_from_mask(len(mask), H)` — uav=0(action 96)/uav>0(192) 자동 분기(구 make_codec 하드코딩
  버그 회피). 검증 `lb_validate17.py`/`lb_validate_sigungu.py`.
- T-메타 RL(`t_meta_wrapper.py`)은 이 규칙의 T 를 상태의존으로 학습한 확장(action=Discrete{2,3,4,6,8,∞}).
</details>

<details>
<summary><b id="dsn-v2">재설계 v2 요약 (2026-07-04~05) — 진단→사다리→UAV 곡선→해석 3종</b></summary>

- **진단 3건**(`docs/연구방향_재점검_2026-07-02.md`): ① obs 에 LB 신호(p_sent·in_flight·ρ) 부재 +
  cap_remain 스케일 앵커 ② dest 는 랭킹 문제인데 flat 분류(dest acc 0.444) ③ PPO 위생(kl 0.28, clip 0.54).
- **L사다리**(진단과 1:1 누적 ablation, 시군구250 전국 단일 10M×4런): L0 재현 → L1 +위생 → L2 +부하 obs →
  L3 +포인터. **L3 이 LB-T4 를 16/17 역전**(0.1199→0.0923), holdout 신좌표 250점 212승 5무 33패.
  패배는 전부 농촌·산간·도서(RL 과분산) — 남은 표적.
- **UAV 한계가치 곡선**(레벨별 개별 모델, `MCI_UAV_NUM∈{0,5,10,15,26}`): 첫 5대가 총이득의 75%,
  10~15대 포화, AMB-only 대비 예방가능 사망 12.7%→9.2%(27%↓). RL 곡선이 전 구간 규칙 곡선 아래.
- **해석 추출 3부작**: 3-A 운용규칙 4종(결정로그 67만 — Red 우선·원거리 tier3 직행·농촌 2배·~20% 포화) /
  3-B 해석 프로그램(시간절감형 UAV mode — LB-T4 개선하나 RL 격차 13%만 회수) / 3-C **T-메타**(발송상한만
  RL 학습 — 30% 회수, 규칙 **T=f(ρ,병원밀도)**: "도시·고스트레스 조이고 농촌 완화", 수동 T4보다 T2~3).
- **전 과정 서사·재현 지도**: `docs/연구흐름_종합_2026-07-06.md` (로컬 docs/).
</details>

<details>
<summary><b id="dsn-viper">VIPER 트리 증류 (v1 Phase 3d)</b></summary>

**VIPER**(Bastani et al., NeurIPS 2018) = Q-DAGGER 가중 + 반복 DAGGER 루프로 PPO 정책을 해석 가능 결정트리로 증류:
- 루프: 현재 트리로 M 에피소드 롤아웃 → 방문 상태에 오라클 라벨 `a*=π*(s)` + criticality 가중치 → CART 재적합 → N회 후 CV 보상 기준 best 트리.
- criticality `ℓ̃(s) = max_a logπ − min_a logπ`(MaskablePPO 는 명시적 Q 없음 → 논문 §2 max-entropy `Q=logπ` 대용).
  `--crit loggap`(기본·논문 충실) / `probmargin` / `uniform`(plain DAGGER).
- 트리 롤아웃은 마스크 준수 masked-argmax. VecNormalize 동결 로드 지원(`--vecnorm`).
- v1 실측: 지역특화 d6 트리는 휴리 승, **전국 단일 트리는 실패** — 원인은 표현공간 부정합(랭킹+제약 규칙을
  축-정렬 분할로 못 담음). 재설계 v2 의 프로그램/T-메타 추출이 그 교훈의 후속(§dsn-v2).
</details>

[↑ 목차](#toc)

---

<a id="envvars"></a>
## 5. 환경변수 레퍼런스

| 변수 | 의미 | 비고 |
|---|---|---|
| `MCI_REDUCED_OBS=1` | obs 를 요약 통계로 축약(차원↓) | **train/eval 동일**해야 모델 로드됨 |
| `MCI_OBS_VARIANT` | `local`\|`comms`\|`full`\|`essential`\|**`essential+load`**(현행, dim 355) | 특징 obs 래퍼; train/eval 동일 |
| `MCI_REWARD_MODE` | `raw`\|`woG`\|`rywt`\|**`pdrwog`**(현행 학습 표준) | 모든 알고리즘 적용(`--reward_mode` 동등) |
| `MCI_CAP_GATE` | **`occ`**(기본: census+이송중)\|`psent`(통신단절: 누적발송) | 발송게이트·obs·마스크·휴리 4곳 공유 |
| `MCI_CARED_OBS` | `1`(기본)\|`0` — 병원 admit(cared) 관측 여부 | psent 와 짝(완전 현장정보 모델) |
| `MCI_TIER_MASK=0` | 등급-tier 마스킹 비활성 | 기본 활성(Red→Tier3) |
| `MCI_INCIDENT_SIZE`/`MCI_CAPA_SCALE`/`MCI_AMB_NUM`/`MCI_UAV_NUM` | 자원·부하 런타임 노브(시나리오 재생성 불요) | 트레이드오프/UAV 곡선용. ⚠️UAV_NUM 은 **출발지만** 슬라이스(착륙지=헬기장 전체) |
| `MCI_ADV_MODE`/`_SUBTRACT_AT`/`_CSV`/`_REGION` | advantage 보상 shaping(사전계산 CSV) | `advantage_wrapper.py` |
| `MCI_OSRM_URL` | OSRM 백엔드 URL | 기본 공개 라우터(대량은 로컬 docker) |
| `KAKAO_API_KEY` | Kakao Mobility 키 | Kakao 모드 필수(코드 하드코딩 금지) |
| `MCI_BUFFER_RATIO`/`MCI_MAX_SEND_COEFF`/`MCI_UTIL_BY_TIER` | 병원 선정/발송상한 knobs | CLI 미지정 시 fallback (기본 buffer 1.5) |

(구 `MCI_GREEN_MASK` 는 2026-07-03 폐기 — Green 이 action 차원 자체에서 제거됨.)

[↑ 목차](#toc)

---

<a id="multiregion"></a>
## 6. 멀티리전 / 전국 RL

매니페스트 JSON(`scenarios/manifests/`, `{region: config_path}`, **절대경로** — Linux 학습박스 경로)로 구동.
트레이너는 `config_path.endswith(".json")` 으로 분기 — `.json`=멀티지역(`MultiRegionEnv`/`FeatureMultiRegionEnv`,
reset 마다 지역 무작위 샘플), `.yaml`=단일. **한 매니페스트 내 모든 지역은 동일 H(=47)** 여야 obs/action 차원이
유지된다.

- **현행 주경로 — 시군구 250 전국 단일정책**: `sigungu_osrm_manifest.json` 으로 학습(§3.3), 시도17 paired +
  hold-out 신좌표로 평가(§3.5). 시군구 휴리 CSV 는 BOM+동명구 주의 → **sigcd 로 매칭**.
- **Plan 1 (지역별 정책, 레거시)** — `gen_regions.py` → `plan1_manifest.json` → `run_grid_parallel.py`(17지역×알고리즘,
  **CPU 강제** `CUDA_VISIBLE_DEVICES=""`) → `run_grid_eval.py` diagonal 평가.
- **hold-out 일반화** — `gen_eval_holdout_osrm.py`(현행) 또는 `sample_region_points.py`→`gen_eval_points.py`(구식,
  `ctprvn.shp` 내부 rejection 샘플).
- **sim 디버그 print**: 이벤트마다 stdout 출력 → 트레이너/워커는 **stdout→/dev/null**, stderr→`.err` 만
  캡처(TensorBoard 로 모니터). `sim_src` 수정 금지(설계 결정).
- 학습 박스는 공유 노드 — 병렬 규모는 **loadavg 로 게이트**(자세한 수칙은 `CLAUDE.md`).

[↑ 목차](#toc)

---

<a id="layout"></a>
## 7. 저장소 구조

<details>
<summary><b>폴더 트리</b></summary>

```
MCI_UAV/
├── scenarios/                  시나리오 (자동생성·gitignore) + seed 입력
│   ├── 엑셀 결합 데이터.xlsx     원본 병원 풀
│   ├── 안전센터와 소방서.csv     AMB 기지 데이터
│   ├── ctprvn.shp              통계청 시도 경계(hold-out 점 샘플용)
│   ├── manifests/              {sigungu,sido}×{osrm,kakao} / plan1 / holdout JSON
│   └── exp_*/(lat,lon)/        지역별 시나리오 산출물 (§4 파일계약)
│
├── src/
│   ├── sce_src/                시나리오 생성
│   │   ├── make_csv_yaml_dynamic.py   단일좌표 생성 (Kakao/OSRM)
│   │   ├── gen_regions.py             17 광역 일괄 (2-pass H_max)
│   │   ├── gen_sigungu_kakao.py  regen_sigungu_osrm.py  시군구 250 일괄/재구축
│   │   └── gen_eval_holdout_osrm.py  gen_eval_points.py  sample_region_points.py  hold-out
│   ├── sim_src/                시뮬레이터 코어 (event-driven, 무수정)
│   │   ├── main.py  ScenarioManager.py  EntityManager.py  EventManager.py
│   │   ├── RuleManager.py             휴리스틱 64룰
│   │   ├── MCIEnvironment_gymnasium.py  gym env (AMB+UAV)
│   │   └── config.yaml  event_info.json
│   └── rl_src/                 강화학습
│       ├── env_wrapper.py             dict→flat, MultiDiscrete→Discrete, joint 마스크, decode/encode
│       ├── hospital_feature_wrapper.py  특징 obs (essential/essential+load)
│       ├── pointer_policy.py            포인터 랭킹 head (재설계 v2)
│       ├── hospital_set_extractor.py    deepsets 인코더
│       ├── loadbalance_heuristic.py     ★부하균형 규칙(LB-T4/적응T) — 비교 기준선
│       ├── reward_redesign_wrapper.py   woG/rywt/pdrwog 보상
│       ├── multi_region_env.py          매니페스트 멀티지역 env
│       ├── train_ppo_feature.py         주력 트레이너 (위생 인자 포함)
│       ├── train_ppo_tmeta.py  t_meta_wrapper.py  T-메타 RL
│       ├── paired_eval_ladder.py  uav_curve_eval.py  program_{policy,eval}.py  tmeta_eval.py  재설계 v2 평가
│       ├── uav_decision_log.py  analyze_uav_rules.py  운용규칙 추출
│       ├── viper_distill.py             VIPER 증류
│       ├── sim_logger{,_sigungu}.py  tradeoff_{sweep,simlog}.py  시뮬로그·트레이드오프
│       ├── train_{ppo,dqn,reinforce}.py  run_all_parallel.py  레거시 학습
│       └── evaluate.py  hybrid_eval.py  run_grid_{parallel,eval}.py  eval_*.py  distill_policy.py …
│
├── tools/                      전국 GIS/OSM 파이프라인 + scene/trace export + exp_drivers/(휴리 배치)
├── external/ml-agents/         Unity ML-Agents (submodule; UAV_test/ 는 로컬 전용)
├── results/  experiment_logs/  학습/평가 산출 (gitignore) — 재설계 v2 는 results/rl/redesign/
├── docs/                       보고서·설계노트 (gitignore, 로컬) — 연구흐름_종합_2026-07-06.md 등
└── requirements.txt
```

각 `rl_src/*` 변형 스크립트는 모듈 docstring 에 재사용 의존성·목적이 적혀 있다.
</details>

[↑ 목차](#toc)

---

<a id="unity"></a>
## 8. Unity 디지털트윈

`external/ml-agents/UAV_test/` (C#) — 전국 255 시군구 3D 한국 지도에 사고 시뮬을 렌더링(건물/OSM 도로/교통/보행자).
**Unity 자산은 submodule 작업트리 안에 untracked 로 존재 → 로컬(Windows)에만 있고 `origin` 으로 push 불가**(의도된 구조).
런타임: `MapVersionSelector`(시나리오 선택) → 필요한 Region 씬 additive 로드 → `TracePlayer` 가 `scene.json`+`trace_flat.json` 재생.
자세한 빌드/임포트/MCP 가이드는 `CLAUDE.md` 의 "Unity 디지털트윈 아키텍처" 참조.

[↑ 목차](#toc)
