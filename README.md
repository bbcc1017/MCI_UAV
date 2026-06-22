# MCI_UAV

대규모 사상자 사고(**MCI**, Mass-Casualty Incident)에서 **구급차(AMB) + 드론(UAV)** 혼합 자원의
환자 triage·이송 의사결정을 **강화학습(RL)** 으로 학습하고, 학습된 정책을 **VIPER 결정트리**로
증류해 해석 가능한 규칙을 추출하는 연구용 코드.

- **시뮬레이터**: MCI_ADV 기반 이산사건 시뮬레이터(`src/sim_src/`) — 환자 구조·이송·병원 처치를
  event-driven 으로 모델링하고, 보상 = 병원 도착(처치 시작) 시점의 생존확률.
- **에이전트**: MaskablePPO(주력) + DQN/REINFORCE 비교, 32종 휴리스틱 룰 baseline, AMB=휴리스틱·UAV=RL 하이브리드.
- **일반화**: 단일 좌표 → 17개 광역시도 → 전국 단일정책(plan1nat) → hold-out 무작위 좌표 평가.
- **최종 목표**: RL·트리 성능을 휴리스틱 대비 유의미한 수준까지 끌어올린 뒤 **논문 라이팅**.

> Unity 디지털트윈 시각화(C#)는 같은 시나리오 데이터를 3D 한국 지도에 렌더링한다. 본 저장소(`origin`)는
> RL/시뮬 연구 코드(Python)만 버전관리하며, Unity 자산은 로컬(Windows)에만 존재한다 → [§ Unity 디지털트윈](#unity).

---

<a id="toc"></a>
## 목차

1. [현재 상태 & 로드맵](#status)
2. [환경 셋업](#setup)
3. [실험 파이프라인 (메인 경로)](#pipeline)
   - [시나리오 생성](#pl-scenario) · [휴리스틱 baseline](#pl-heur) · [RL 학습](#pl-train) · [VIPER 증류](#pl-viper) · [평가](#pl-eval)
4. [핵심 설계](#design) — obs/action/reward · 마스킹 · 파일계약 · H_max floor · 특징 obs · VIPER
5. [환경변수 레퍼런스](#envvars)
6. [멀티리전 / 전국 RL](#multiregion)
7. [저장소 구조](#layout)
8. [Unity 디지털트윈](#unity)

---

<a id="status"></a>
## 1. 현재 상태 & 로드맵

SSH 서버의 기존 시나리오가 소실되어 **재생성·재실험**이 필요한 상황에서, 2026-06 랩미팅 피드백 + VIPER 논문
적용을 계기로 **3-Phase 재설계**를 완료했다. 현재는 **재설계 완료 → 실규모 실험 직전** 단계.

| 단계 | 내용 | 상태 |
|---|---|---|
| **Phase 1** | 시나리오 파일 I/O 통합 (`hospital_info.csv`/`amb_station_info.csv`/`uav_info.csv`, amb 카운트화, load-time 슬라이스) | ✅ 완료 |
| **Phase 2** | 데이터 기반 **H_max floor** (2-pass, 임의의 46 폐기) | ✅ 완료 |
| **Phase 3a** | 특징기반 병원 obs 래퍼 (`HospitalFeatureWrapper`) | ✅ 완료 |
| **Phase 3b·c** | 특징 obs 학습 통합 (`train_ppo_feature.py`) + 순열등변 집합 인코더 (`hospital_set_extractor.py`) | ✅ 완료 |
| **Phase 3d** | **VIPER 트리 증류** (반복 Q-DAGGER, `viper_distill.py`) | ✅ 완료 |
| **실험** | 전국 Kakao 시나리오 생성 → feature-PPO 학습 → VIPER 증류 → 17지역·일반화·local/comms 평가 | 🔜 진행 예정 |
| **논문** | RL·트리가 휴리스틱 대비 유의미한 margin 달성 후 라이팅 | ⏳ |

<details>
<summary><b>다음 실험 순서 (요약)</b></summary>

1. **시나리오 생성** — `gen_regions.py`(17 광역) + `national_train.json`(전국 학습용) + hold-out 평가점. 전부 **Kakao 출발시각 교통** 기반, **H 고정**(min_hos_num=H_max)으로 obs/action 차원 통일.
2. **휴리스틱 baseline 기록** — 32룰 시뮬로 지역별 최선 룰 CSV 확보(RL·트리 비교 기준선).
3. **RL 학습** — `train_ppo_feature.py` 로 plan1nat(전국 단일정책) + plan1(17지역별). `--extractor {mlp,deepsets}` 비교.
4. **정보수준 ablation** — `MCI_OBS_VARIANT=local|comms|full` 3종 학습·평가(랩 피드백 #3).
5. **VIPER 증류** — `viper_distill.py --crit loggap` 로 해석트리 추출, 17지역 margin 유지율 보고.
6. **평가·집계** — 17지역 diagonal eval, hold-out 일반화, 하이브리드(AMB=heur/UAV=RL).

> ⚠️ 실험 전 점검: `gen_eval_points.py` 의 `--fixed_hos_num` 기본값(46)은 Phase 2 이전 잔재 →
> 학습 H(=H_max)와 **반드시 일치**시켜 재생성해야 모델 로드 시 obs 차원이 맞는다.
</details>

[↑ 목차](#toc)

---

<a id="setup"></a>
## 2. 환경 셋업

- Python 3.10 (conda env **`UAV`**). torch 2.8.0+cu128(RTX 50/Blackwell) 또는 2.5.1+cu121(RTX A6000).
- SB3 + sb3-contrib(MaskablePPO), gymnasium <1, numpy <2, scikit-learn(VIPER/증류).

```bash
conda activate UAV
pip install -r requirements.txt
# torch 는 GPU 에 맞춰 별도 설치 (requirements.txt 주석 참고)
```

라우팅 백엔드(시나리오 생성용):

```bash
# (A) OSRM — 기본, 결정적, 교통 미반영. 대량 평가는 로컬 컨테이너 권장
tools/osrm_prepare_korea.sh
docker compose -f docker-compose.osrm.yml up -d
export MCI_OSRM_URL=http://localhost:5000
# docker 그룹 권한이 없는 세션에서는 `sudo -n docker compose -f docker-compose.osrm.yml up -d`

# 병원-병원 원본 도로거리 행렬 재생성(엑셀 결합 데이터.xlsx 기준, 페리 포함 OSRM 경로 사용)
tools/build_distance_matrix_osrm.py

# (B) Kakao Mobility — 출발시각 교통 반영(실험 본편). 키는 ENV 로만
export KAKAO_API_KEY=<your_key>
```

> 작업 분담: **RL/시뮬 학습은 SSH(Linux)**, **Unity 디지털트윈은 로컬(Windows)**. 학습 박스(A6000)에
> `UAV` env 가 셋업되어 PPO 학습·VIPER 증류까지 검증 가능.

[↑ 목차](#toc)

---

<a id="pipeline"></a>
## 3. 실험 파이프라인 (메인 경로)

데이터 흐름: **시나리오 YAML → gym env → 래퍼 → 학습/증류/평가.** 휴리스틱·RL·트리는 **동일 시나리오 파일**을
공유해 비교 일치성을 보장한다(모든 거리/시간은 생성 시 사전계산·동결 — sim-time API 호출 금지).

<a id="pl-scenario"></a>
### 3.1 시나리오 생성

**17개 광역시도 일괄 (Plan 1, Kakao):**

```bash
export KAKAO_API_KEY=<key>
python src/sce_src/gen_regions.py \
  --departure_time 202605261530 --incident_size 100 \
  --amb_count 30 --uav_count 25 --uav_num 3 \
  --min_hos_mode auto --exp_prefix plan1
# → scenarios/exp_plan1_<region>_dep_<ts>/(lat,lon)/config_*.yaml
# → scenarios/manifests/plan1_manifest.json
```

- `--min_hos_mode auto`(기본): **Pass1**(road API 0회)으로 17지역 자연 선정 병원수를 구해 `H_max = min(max(natural), min(pool))` 산출 → **Pass2** 에서 모든 지역을 ≥H_max 로 생성(차원 통일). [자세히](#dsn-hmax)
- `--amb_count`=AMB 런타임수, `--uav_count`=UAV 생성 superset 상한(헬기장 병원당 1대), `--uav_num`=UAV 런타임수(load 시 슬라이스).

**단일 좌표:**

```bash
python src/sce_src/make_csv_yaml_dynamic.py --base_path . \
  --experiment_id mix_seoul --latitude 37.5666 --longitude 126.9784 \
  --incident_size 100 --amb_count 30 --uav_count 25 --is_use_time True --kakao_api_key $KAKAO_API_KEY
```

**전국 단일정책(plan1nat) 학습/평가 셋:**

```bash
# 학습용: scenarios/manifests/national_train.json 에 정의된 지역들
# 일반화 평가용 hold-out 무작위 좌표
python src/sce_src/sample_region_points.py --n 5            # ctprvn.shp 내부 점 샘플
python src/sce_src/gen_eval_points.py --min_hos_num <H_max> # ★ 학습 H 와 일치시킬 것
```

<a id="pl-heur"></a>
### 3.2 휴리스틱 baseline

```bash
python src/sim_src/main.py --config_path "scenarios/exp_plan1_서울_dep_<ts>/(lat,lon)/config_*.yaml"
# results/<EXP_ID>/results_*.txt, results_*_stat.txt  (보상/시간/PDR + woG)
```

32룰 조합(START/ReSTART × RedOnly/YellowNearest × Red mode 4 × Yellow mode 4)을 각각 시뮬해 CI 와 함께 기록.

<a id="pl-train"></a>
### 3.3 RL 학습 (특징 obs PPO)

```bash
# 전국 단일정책 (manifest .json → FeatureMultiRegionEnv, reset 마다 무작위 지역)
python src/rl_src/train_ppo_feature.py \
  --config_path scenarios/manifests/plan1nat_manifest.json \
  --total_timesteps 200000 --n_envs 4 --extractor mlp \
  --log_dir results/rl/ppo_feature
# 정보수준 ablation: 앞에 MCI_OBS_VARIANT=local (또는 comms) 를 붙여 동일 명령 반복
```

- `--extractor mlp`(기본, 평탄 obs) / `deepsets`(순열등변 집합 인코더 + 자기어텐션). [obs 설계](#dsn-feature)
- TensorBoard: `tensorboard --logdir results/rl/ppo_feature/tb`.

> 비교용 레거시 경로: `run_all_parallel.py`(DQN+PPO+REINFORCE 동시), `train_ppo.py`(인덱스 obs).

<a id="pl-viper"></a>
### 3.4 VIPER 트리 증류

```bash
MCI_REDUCED_OBS=1 python src/rl_src/viper_distill.py \
  --manifest scenarios/manifests/plan1nat_manifest.json \
  --model results/rl/ppo_feature/final_model.zip \
  --n_iter 5 --rollout_eps 10 --max_depth 8 --crit loggap \
  --heur_csv results/plan1nat_heur_eval.csv --out_dir results/viper
# → results/viper/viper_loggap_d8.pkl + _rules.txt + _region_eval.csv
```

반복 Q-DAGGER 로 오라클 PPO 를 결정트리로 증류 → 17지역에서 PPO·휴리스틱 대비 **margin 유지율** 보고. [자세히](#dsn-viper)

<a id="pl-eval"></a>
### 3.5 평가

```bash
# (a) 단일 좌표 — RL vs 휴리스틱
python src/rl_src/evaluate.py --config_path <yaml> \
  --ppo_path <model.zip> --include_heuristic --n_episodes 100

# (b) 17개 광역 diagonal (지역별 모델 vs 자기 지역 휴리스틱 최선)
python src/rl_src/run_grid_eval.py --manifest scenarios/manifests/plan1_manifest.json \
  --model_root results/rl/plan1 --n_episodes 1000

# (c) 정보수준 ablation 평가
python src/rl_src/eval_obs_variant.py --variant local --model <model.zip> \
  --manifest <manifest.json> --heur_csv <heur.csv> --tag local

# (d) 하이브리드 (AMB=휴리스틱, UAV=RL)
python src/rl_src/hybrid_eval.py --config_path <yaml> --ppo_path <model.zip> \
  --rule_priority START --rule_hos_select RedOnly \
  --rule_red_mode Both_AMBFirst --rule_yellow_mode Both_AMBFirst --mode_split strict --n_episodes 100
```

[↑ 목차](#toc)

---

<a id="design"></a>
## 4. 핵심 설계

<details>
<summary><b>관측(obs) / 행동(action) / 보상(reward) 인코딩</b></summary>

- **Obs (dict)** — `p_states (N,5)`=[class, rescued, move, moved, cared]; `h_states (H,3)`=[idle, queue, occupied];
  `p_sent (H,)`; `amb_states`/`uav_states (n,3)`=[dest, time_remaining, severity]; `p_at_site (4,)`=[R/Y/G/B 현장];
  `n_amb_at_site`, `n_uav_at_site`, `time`. → RL 래퍼가 1D float32 로 평탄화.
- **Action `[class, dest, mode]`** — class 0=Red/1=Yellow/2=Green; dest 0=현장대기, 1..H=병원; mode 0=AMB/1=UAV.
  단일 차종일 때 mode 자동 고정(`amb_num=0`→UAV, `uav_num=0`→AMB). 래퍼가 `MultiDiscrete([3,H+1,2])` ↔ `Discrete` 로 encode/decode.
- **Reward** — 환자 admit(처치 시작) 시점의 생존확률. Red/Yellow 는 시간 감쇠, Green=1, Black=0.
  `reward_redesign_wrapper.py` 가 `raw` | `woG`(Green 제외) | `rywt`(R/Y 가중)로 재구성(`MCI_REWARD_MODE`).
- **PDR**(Preventable Death Rate) = `1 − saved/preventable`, preventable 은 "구조 즉시 처치" 가정 best-case.
</details>

<details>
<summary><b>행동 마스킹 (하드 제약)</b></summary>

마스킹은 패널티가 아니라 **하드 제약**(`action_masks()`):
- Red → Tier3(상급종합) 병원만 (등급-tier 치료가능 마스크, `MCI_TIER_MASK=0` 로 비활성).
- UAV(mode=1) → 헬기장 보유 병원만.
- dest 는 해당 class 환자 존재 ∧ 해당 mode 자원 존재 ∧ 병원 capa 여유(`p_sent < max_send`)일 때만 허용. stay(dest=0)는 항상 허용.

RL 경로는 **결합 마스크**(`action_masks_joint`)만 사용한다(per-dim 마스크는 결합 제약을 표현 못 함). 마스크 우회
방어로 `EventManager.proceed_action()` 에 `NO HELIPAD`/`NO PATIENT` 가드를 둔다.

래퍼 체인(외→내): `Monitor → ActionMasker → [HeuristicAdvantage] → HospitalFeatureWrapper(또는 FlattenAndDiscrete/Hybrid) → [RewardRedesign] → base env`. **코어 파일은 무수정, 변형은 래핑으로.**
</details>

<details>
<summary><b id="dsn-contract">시나리오 파일 계약 (Phase 1)</b></summary>

지역 폴더 `scenarios/exp_<prefix>_<region>_dep_<ts>/(lat,lon)/` 산출물:
- `hospital_info.csv` — 통합 병원 메타+현장거리: 요양기관명/종별코드/헬기장 여부/수술실수/병상수/`euc_dist`/`road_dist`/`road_duration`. **도로소요시간 오름차순 = 병원 인덱스**.
- `amb_station_info.csv` — 고유 안전센터/소방서당 1행(`보유대수`=count). load 시 보유대수만큼 `np.repeat` 전개 후 `amb_num` 만큼 슬라이스 → 같은 센터 동일거리 차량(Kakao road API 호출 절감).
- `uav_info.csv` — 헬기장 병원 superset(가까운 순, `hospital_idx` 보존). load 시 `uav_num` 슬라이스.
- `distance_Hos2Hos_{euc,road}.csv` — diversion 용 병원간 거리행렬(유지). `patient_info.csv`, `config_*.yaml`, `scene.json`(Unity 용).

**매핑**: `수술실수→hos_max_capa`, `병상수→hos_max_queue`, `종별코드 1→Tier3·그외→Tier2`, `헬기장 여부→helipad_idx`.
입원정원 `max_capa = 수술실수 + 병상수`, 발송상한 `max_send = max_capa`([1,1] 계수). 로더는 구포맷도 폴백 지원.
</details>

<details>
<summary><b id="dsn-hmax">데이터 기반 H_max floor (Phase 2)</b></summary>

지역마다 자연 선정 병원수가 달라 Top-K obs 후보가 부족해지는 문제를, **2-pass floor** 로 해결:
- **Pass1**(road API 0회, Kakao 키 불필요): 각 대표점에 용량/tier 선정 로직만 적용해 자연 선정수 산출 → `H_max = min(max(natural), min(pool))`.
- **Pass2**: 모든 지역을 `min_hos_num=H_max` 로 floor-up(보장룰 자동 보존, cap-down 안 함).

병원 선정 공식: `eff = 수술실수 + 응급실병상수×(1−tier별 가동률)`, 누적 `eff ≥ 환자수×buffer_ratio(1.5)` 까지 가까운 순 +
tier/헬기장 보장룰. `fixed_hos_num`(cap, 구호환)와 `min_hos_num`(floor, 신규)은 상호배타. 제주(섬)는 Kakao 페리 경로
지원으로 본토 병원 섞여도 정상(별도 예외 불필요).
</details>

<details>
<summary><b id="dsn-feature">특징기반 병원 obs + 정보수준 ablation (Phase 3a·b·c)</b></summary>

인덱스 기반 `h_states`/`p_sent` 대신 **병원당 특징 엔티티 행렬 (H, F)** (`HospitalFeatureWrapper`):
- 병원당 특징 F (`MCI_OBS_VARIANT` 토글):
  - `local`(정적 사전지식): `[is_tier3, helipad, eta_amb, eta_uav]`
  - `comms`(실시간 통신): `[idle, queue, occ, cap_remain]`
  - `full`(기본, 8열) = local + comms
- ETA = `amb/uav_HtoS_t[0]`(lognormal 평균 = API duration, 랩 피드백 #1), 시나리오 최소값 정규화(지역간 스케일 제거).
- 글로벌 특징(37) = patient_agg(20) + vehicle_agg(10) + p_at_site(4) + n_amb(1) + n_uav(1) + time(1).
- action 은 `Discrete(H+1)` 유지(병원 위치 인덱스). `--extractor deepsets` 는 가중치공유 임베딩 + 자기어텐션의
  **순열 등변**(불변 아님 — 위치 인덱스 행동과 충돌 방지) 인코더.

랩 피드백 통합: #1 ETA=lognormal 평균, #2 tier 를 obs 특징으로(과거엔 마스킹 전용), #3 local↔comms 정보수준 축.
</details>

<details>
<summary><b id="dsn-viper">VIPER 트리 증류 (Phase 3d)</b></summary>

**VIPER**(Bastani et al., NeurIPS 2018) = Q-DAGGER 가중 + 반복 DAGGER 루프로 PPO 정책을 해석 가능 결정트리로 증류:
- 루프: 현재 트리로 M 에피소드 롤아웃 → 방문 상태에 오라클 라벨 `a*=π*(s)` + criticality 가중치 → CART 재적합 → N회 후 CV 보상 기준 best 트리.
- criticality `ℓ̃(s) = max_a logπ − min_a logπ`(MaskablePPO 는 명시적 Q 없음 → 논문 §2 max-entropy `Q=logπ` 대용).
  `--crit loggap`(기본·논문 충실) / `probmargin` / `uniform`(plain DAGGER).
- 트리 롤아웃은 마스크 준수 masked-argmax. 17지역에서 PPO·휴리스틱 대비 margin 유지율·트리 크기(leaves/depth) 보고.
</details>

[↑ 목차](#toc)

---

<a id="envvars"></a>
## 5. 환경변수 레퍼런스

| 변수 | 의미 | 비고 |
|---|---|---|
| `MCI_REDUCED_OBS=1` | obs 를 요약 통계로 축약(차원↓) | **train/eval 동일**해야 모델 로드됨 |
| `MCI_REWARD_MODE` | `raw`\|`woG`\|`rywt` 보상 재구성 | 모든 알고리즘 적용 |
| `MCI_OBS_VARIANT` | `local`\|`comms`\|`full` 정보수준 | 특징 obs 래퍼; train/eval 동일 |
| `MCI_TIER_MASK=0` | 등급-tier 마스킹 비활성 | 기본 활성(Red→Tier3) |
| `MCI_ADV_MODE` / `_SUBTRACT_AT` / `_CSV` / `_REGION` | advantage 보상 shaping(사전계산 CSV 기준) | `advantage_wrapper.py` |
| `MCI_OSRM_URL` | OSRM 백엔드 URL | 기본 공개 라우터 |
| `KAKAO_API_KEY` | Kakao Mobility 키 | Kakao 모드 필수(코드 하드코딩 금지) |
| `MCI_BUFFER_RATIO` / `MCI_MAX_SEND_COEFF` / `MCI_UTIL_BY_TIER` | 병원 선정/발송상한 knobs | CLI 미지정 시 fallback (기본 buffer 1.5) |

[↑ 목차](#toc)

---

<a id="multiregion"></a>
## 6. 멀티리전 / 전국 RL

매니페스트 JSON(`scenarios/manifests/`, `{region: config_path}`, **절대경로** — Linux 학습박스 경로)로 구동.
트레이너는 `config_path.endswith(".json")` 으로 분기 — `.json`=멀티지역(`MultiRegionEnv`/`FeatureMultiRegionEnv`, reset 마다 지역 샘플), `.yaml`=단일.
**한 매니페스트 내 모든 지역은 동일 H** 여야 obs/action 차원이 유지된다(Phase 2 H_max floor 로 보장).

- **Plan 1 (지역별 정책)** — `gen_regions.py` → `plan1_manifest.json` → `run_grid_parallel.py`(17지역×알고리즘, **CPU 강제** `CUDA_VISIBLE_DEVICES=""`) → `run_grid_eval.py` diagonal 평가.
- **plan1nat (전국 단일정책)** — `national_train.json` 으로 학습 → `plan1nat_manifest.json`. 일반화 평가는 hold-out 점(`sample_region_points.py`→`gen_eval_points.py`, `ctprvn.shp` 내부 rejection 샘플).
- **sim 디버그 print**: 이벤트마다 stdout 출력 → 트레이너/워커는 **stdout→/dev/null**, stderr→`.err` 만 캡처(TensorBoard 로 모니터). `sim_src` 수정 금지(설계 결정).

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
│   ├── manifests/              plan1 / plan1nat / national_train / eval_points JSON
│   └── exp_*/(lat,lon)/        지역별 시나리오 산출물 (§4 파일계약)
│
├── src/
│   ├── sce_src/                시나리오 생성
│   │   ├── make_csv_yaml_dynamic.py   단일좌표 생성 (Kakao/OSRM)
│   │   ├── gen_regions.py             17 광역 일괄 (2-pass H_max)
│   │   ├── gen_eval_points.py         hold-out 평가점 생성
│   │   └── sample_region_points.py    ctprvn.shp 내부 점 샘플
│   ├── sim_src/                시뮬레이터 코어 (event-driven, 무수정)
│   │   ├── main.py  ScenarioManager.py  EntityManager.py  EventManager.py
│   │   ├── RuleManager.py             휴리스틱 32룰
│   │   ├── MCIEnvironment_gymnasium.py  gym env (AMB+UAV)
│   │   └── config.yaml  event_info.json
│   └── rl_src/                 강화학습
│       ├── env_wrapper.py             dict→flat, MultiDiscrete→Discrete, joint 마스크, decode/encode
│       ├── hospital_feature_wrapper.py  특징 obs 래퍼 (Phase 3a)
│       ├── hospital_set_extractor.py    순열등변 집합 인코더 (Phase 3c)
│       ├── multi_region_env.py          매니페스트 멀티지역 env
│       ├── train_ppo_feature.py         특징 obs PPO (Phase 3b)
│       ├── train_{ppo,dqn,reinforce}.py 레거시 학습
│       ├── viper_distill.py             VIPER 증류 (Phase 3d)
│       ├── cross_location_eval.py  run_grid_{parallel,eval}.py  eval_*.py
│       ├── evaluate.py  hybrid_eval.py  distill_policy.py
│       └── advantage_wrapper.py  reward_redesign_wrapper.py  enriched_env_wrapper.py  aggregate_obs.py …
│
├── tools/                      전국 GIS/OSM 파이프라인 + scene/trace export
├── external/ml-agents/         Unity ML-Agents (submodule; UAV_test/ 는 로컬 전용)
├── results/  experiment_logs/  학습/평가 산출 (gitignore)
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
