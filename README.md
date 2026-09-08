# MCI_UAV

대규모 사상자 사고(**MCI**, Mass-Casualty Incident)에서 **구급차(AMB) + 무인기(UAV)** 혼합 자원의
환자 분류·이송 의사결정을 **강화학습(RL)** 으로 학습하고, 학습된 정책을 **현장 대원이 종이 한 장으로
쓸 수 있는 명시 규칙**으로 역설계하는 연구 코드다.

> **TL;DR (EN)** — Event-driven MCI simulator + MaskablePPO (pointer head over hospitals) for joint
> triage / hospital / vehicle-mode dispatch. The trained policy is reverse-engineered into a
> three-line field rule whose destination score is `travel time + λ · queue penalty`; on 750 held-out
> Korean districts the rule **matches the strongest learned teacher** and beats a strong
> capacity-capped heuristic baseline by 0.029 preventable-death rate. λ turns out to be a
> **queueing constant** (service time ÷ number of operating rooms), not a tuned magic number.

- **시뮬레이터** `src/sim_src/` — 환자 구조·이송·병원 처치를 이산사건으로 모델링. 보상 = **치료 개시
  시각의 생존확률**(수술실 배정 순간이지 병원 도착 순간이 아니다).
- **에이전트** — MaskablePPO(병원 랭킹 pointer head, 순열등변) 주력. 비교군 = 휴리스틱 64룰,
  문헌 규칙(Shin–Lee 16종), 발송상한 규칙(START-LB3), MILP 롤링호라이즌, 롤아웃 플래너(NCRP),
  의사결정나무·GBDT 증류.
- **일반화** — 단일 좌표 → 17 광역시도 → 시군구 250 전국 단일정책 → **미학습 좌표 750점 폐루프 판정**.
- **최종 산출** — 학습 모델이 아니라 **규칙집**. 목적지 점수 `도달시간(분) + λ·대기벌점`, λ ≈ 서비스시간/수술실수.

---

## 목차

1. [무엇을 푸는 문제인가](#problem)
2. [현재 결과](#results)
3. [저장소 구조](#layout)
4. [환경 셋업](#setup)
5. [실험 파이프라인](#pipeline)
6. [핵심 설계 — obs / action / reward / 마스킹](#design)
7. [판정 규약(재현성)](#protocol)
8. [환경변수 레퍼런스](#envvars)
9. [Unity 디지털트윈 시각화](#unity)
10. [문서와 기록](#docs)
11. [데이터 출처 · 라이선스](#license)

---

<a id="problem"></a>
## 1. 무엇을 푸는 문제인가

사고 현장에 환자 100명, 구급차 30대, 무인기(에어앰뷸런스) 26대, 후보 병원 47곳이 있다.
구조된 환자가 한 명 나올 때마다 지휘소는 **세 가지를 동시에** 정해야 한다.

| 축 | 선택지 | 제약 |
|---|---|---|
| 등급 | Red(긴급) / Yellow(응급) 중 누구를 먼저 | Green/Black 은 코어가 일괄 처리 |
| 목적지 | 현장 대기 또는 병원 1..47 | Red 는 Tier3(권역) 병원만, UAV 는 헬기장 보유 병원만 |
| 수단 | AMB / UAV | 그 수단이 현장에 대기 중일 때만 |

목표는 **예방가능 사망률(PDR_woG) 최소화**다. `PDR_woG = 1 − (달성 생존확률 합) / (예방가능 최대)`
로 사고 규모에 불변이며 **낮을수록 좋다**. 병목은 병원 수용량이 아니라 **수술실 대기**다 —
전원(diversion)은 실측 0건이고, 손실은 100% **치료 개시 지연**에서 나온다.

연구의 종착점은 "모델을 만들었다"가 아니라 **RL이 발견한 의사결정 원리를 현장 관측변수만으로
설명 가능한 규칙으로 제시하고, 그 규칙이 폐루프 재시뮬레이션에서 실제로 더 낫다는 것을 보이는 것**이다.

---

<a id="results"></a>
## 2. 현재 결과

판정셋 = **test750**(시군구 250곳 × 각 3좌표, 학습·튜닝에 쓰지 않은 좌표) × **시드 30개**,
정책당 22,500 에피소드, 공통 난수(CRN) paired. PDR_woG 는 낮을수록 좋다.

| 정책 | 정보수준 | PDR_woG | 비고 |
|---|---|---:|---|
| `START-LB3` 발송상한 휴리스틱 | 현장 | 0.167356 | 강한 설명가능 기준선 |
| `CARD_K12` 거리 + 선형 부하 | 병원 통신 | 0.144985 | 규칙집 1세대(v17) |
| `PPO_NATIONAL` 전국 단일 교사 | 현장 obs | 0.139738 | 10M steps |
| **`CARD_P18`** 도달시간 + 대기행렬 벌점 | **통신 불요** | **0.139207** | 지휘소 화이트보드만으로 실행 |
| **`CARD_Q18`** 같은 형태 + 병원 재고 | 병원 통신 | **0.138445** | 최종 규칙 |
| `PPO_SIDO` 광역시도 17벌 교사 | 현장 obs | 0.138839 | **Q18 과 동률** |

- **Q18 vs 현행 규칙 K12** = +0.006540 ± 0.000318 (626승 122무 2패) — 개선의 본체는 변수 선택이
  아니라 **함수형 교체**(이진 발송상한 → 연속 교환율 → 대기행렬 유도형).
- **Q18 vs 최강 교사 `PPO_SIDO`** = +0.000393 ± 0.000340 → 판정선(0.00053) 미만 **동률**.
  전국 단일 교사에는 유의하게 앞선다(+0.001293, 시드 27/30). *"교사를 추월했다"고 쓰지 않는다.*
- **임계값이 구조상수다** — λ 는 튜닝 상수가 아니라 **서비스시간 ÷ 수술실수**. 치료시간 6배 범위에서
  로그-로그 기울기 +1.06(이론 +1), 용량축 의존성은 형태 보정마다 한 단위씩 소멸(−1.30 → −1.03 → −0.20).
  즉 **다른 물리 조건으로 옮겨도 재튜닝이 거의 필요 없다**(27조건 전이 후회 평균 +0.00010).
- **UAV 도입 효과** = 0.201 → 0.139, **−31%**. 무인기의 실제 역할은 원거리 접근이 아니라
  **구급차 대기열 흡수**로 측정됐다.
- **통신보다 계산이 비싸다** — 병원 통신을 끊는 비용은 +0.00076(회복률 99.4%)인데, 식에서 나눗셈을
  빼면 +0.00283, 선형화하면 +0.00497. 단 **병원 재고만 보고 내가 보낸 환자를 세지 않으면 +0.0929로 붕괴**한다.

전체 이력(v1~v21의 채택·기각 근거와 수치)은 [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md) ·
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) 에 있다. **기각된 실험도 그대로 남긴다** — 이 저장소에서
음성 결과는 실패가 아니라 결론이다(반응형 정책의 룩어헤드 흡수 불가 5중 확증, 표현력 확장 4연속 음성,
"모방 정확도 ≠ 폐루프 성능" 6회 재현 등).

---

<a id="layout"></a>
## 3. 저장소 구조

```
MCI_UAV/
├── src/
│   ├── sim_src/                 시뮬레이터 코어 (이산사건, 무수정 유지)
│   │   ├── main.py  ScenarioManager.py  EntityManager.py  EventManager.py
│   │   ├── MCIEnvironment_gymnasium.py    gym env (AMB + UAV)
│   │   ├── RuleManager.py                 휴리스틱 64룰
│   │   ├── ShinHeuristics.py              문헌 규칙 16종(Shin–Lee 2020 적응)
│   │   └── ShinAlignedHeuristics.py       위의 병원선택 정합 변형 16종
│   ├── sim_src_upgrade/         동일 결과·고속 코어(원본 무수정 사본 + 등가성 검증 하네스)
│   ├── sce_src/                 시나리오 생성 (OSRM / Kakao 라우팅, 병원 선정)
│   ├── rl_src/                  RL·규칙·증류·평가 (아래 표)
│   └── vis_src/                 지도 시각화
├── scenarios/                   병원·소방서 원본, 시도 경계 shp, manifests/
├── tools/                       라우팅 배관 + 집계·리포트 스크립트 + exp_drivers/(실험 드라이버)
├── scoreboard/                  버전별 판정 프로토콜(JSON) — 방법 ID·제외 사유의 정본
├── external/ml-agents/          Unity ML-Agents (submodule) — §9
├── CLAUDE.md / AGENTS.md        에이전트용 프로젝트 지침 = 사실상 엔지니어링 로그(정본)
├── CLAUDE.unity.md              Unity·GIS 파이프라인 지침(로컬 전용 자산 설명)
└── RESEARCH_HISTORY.md / RESEARCH_LOG.md   실험 발전사·기각 목록
```

`src/rl_src/` 주요 모듈:

| 파일 | 역할 |
|---|---|
| `env_wrapper.py` | dict→flat obs, MultiDiscrete→Discrete, **행동 마스킹**, `encode/decode_action` |
| `hospital_feature_wrapper.py` | 병원별 특징 obs (`field` / `essential+load+valid` 등) · `MCI_H_PAD` 패딩 |
| `pointer_policy.py` | 병원 랭킹 pointer head(순열등변, MaskablePPO 호환) |
| `train_ppo_feature.py` | 주력 트레이너(멀티지역 매니페스트 학습, `meta.json` 자동 기록) |
| `v17_field_rules.py` | **현장 규칙집** — 정적 물리량 추출 · 임계값 채굴(`mine`) · CARD 정책 생성 |
| `v17_rule_eval.py` / `v17_ppo_eval.py` | 규칙·PPO 폐루프 평가(동일 rollout·seed·CSV 규약) |
| `lb3_policy.py` / `loadbalance_heuristic.py` | 발송상한 기준선(START-LB3, LB-T) |
| `milp_policy.py` / `planner_policy.py` | MILP 롤링호라이즌 · NCRP 롤아웃 플래너(성능 상한 참고) |
| `tree_distill_policy.py` / `v10_tree_distill.py` | 후보랭킹 CART/GBDT 증류(병원 번호 비의존) |
| `paired_eval_ladder.py` | paired 판정 하네스(지역별 에피소드 배열 + 95%CI) |
| `v20_mechanism_eval.py` | 손실 분해 계측기(미진입 손실 + 지연 손실 = PDR 항등식 검증) |

> **저장소에 없는 것**: 학습 산출물(`results/`), 생성된 시나리오(`scenarios/exp_*`), 보고서 본문(`docs/`),
> 대용량 GIS 데이터는 `.gitignore` 대상이다(수십~수백 GB). 종결된 실험 자산과 일회성 분석 코드는
> `archive/`(로컬 보관, 원장 README 만 추적)로 옮기고 복원 경로를 함께 적어둔다.

---

<a id="setup"></a>
## 4. 환경 셋업

- Python 3.10 (conda env **`UAV`**), torch 2.8.0+cu128, SB3 + sb3-contrib(MaskablePPO),
  gymnasium <1, numpy <2, scikit-learn.
- **GPU 는 병목이 아니다** — `env.step` 이 지배적이라 CPU 학습이 GPU와 동일 처리량(실측 222 vs 221 fps).

```bash
conda activate UAV
pip install -r requirements.txt          # torch 는 GPU 에 맞춰 별도 설치
```

라우팅 백엔드(시나리오 생성 시에만 필요 — 학습·평가는 사전계산된 거리행렬만 읽는다):

```bash
# (A) OSRM — 기본. 결정적, 교통 미반영. 대량 생성은 로컬 컨테이너 권장
tools/osrm_prepare_korea.sh
docker compose -f docker-compose.osrm.yml up -d
export MCI_OSRM_URL=http://localhost:5000
python tools/build_distance_matrix_osrm.py     # 병원↔병원 도로거리 행렬 재생성

# (B) Kakao Mobility — 출발시각 교통 반영. 키는 환경변수로만
export KAKAO_API_KEY=<your_key>
```

---

<a id="pipeline"></a>
## 5. 실험 파이프라인

데이터 흐름: **시나리오 YAML → gym env → 래퍼 → 학습/증류/평가.**
휴리스틱·RL·규칙·트리는 **같은 시나리오 파일과 같은 행동 마스크**를 공유한다. 모든 거리·시간은
생성 시점에 사전계산·동결되며 시뮬 중 외부 API 호출은 없다.

**표준 시나리오 파라미터**: 병원 `fixed_hos_num 47` · 헬기장 26(`uav_num 26`) · `amb_count 30` ·
`incident_size 100` · 속도 50/200 km/h · 인계 5/10분.

### 5.1 시나리오 생성

```bash
# 단일 좌표
python src/sce_src/make_csv_yaml_dynamic.py --latitude 37.4511 --longitude 126.6565 --fixed_hos_num 47
# 17 광역시도 일괄
python src/sce_src/gen_regions.py --road_mode osrm --fixed_hos_num 47
# 시군구 250 × 30좌표 풀(현행 정본, OSRM 스냅 500m 게이트)
python src/sce_src/gen_sigungu30_osrm.py
python src/sce_src/split_sigungu30.py --wave1 16        # 지역별 학습/예산/평가 분할 재생성
```

### 5.2 휴리스틱 기준선

```bash
# 단일 시나리오 64룰 전수
python src/sim_src/main.py --config_path scenarios/exp_*/…/config_*.yaml
# 매니페스트 일괄(64룰 × 1000ep) + 집계
OMP_NUM_THREADS=1 python tools/exp_drivers/run_heur_batch.py <manifest> occ 32
python tools/exp_drivers/aggregate_heur.py <manifest> "" <prefix>
# 문헌 규칙(Shin–Lee) 스모크
python tools/smoke_shin_heuristics.py
```

### 5.3 RL 학습 (현행 v19 레시피)

```bash
MCI_OBS_VARIANT=field MCI_H_PAD=47 MCI_CAP_GATE=occ \
python src/rl_src/train_ppo_feature.py \
  --config_path scenarios/manifests/sigungu30_train6000_manifest.json \
  --extractor pointer --n_attn_blocks 0 \
  --reward_mode pdrwog --norm_reward \
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03 --n_epochs 5 \
  --n_steps 512 --batch_size 512 --embed_dim 64 --ctx_dim 128 --head_hidden 128 \
  --n_envs 8 --vec subproc --total_timesteps 10000000 --seed 0 \
  --save_vecnormalize --log_dir results/rl/v19/national
```

- obs `field` = **현장 지휘소가 실제로 아는 값만**(차량 실시간 잔여시간·병원 실시간 점유·미구조 환자
  등급을 제거). `--n_attn_blocks 0`(attention 제거)이 성능·재현성 모두에서 채택된 구성이다.
- ⚠️ **병원 블록은 슬롯별 정규화를 하면 안 된다** — 병원 슬롯이 현장 거리순으로 정렬돼 있는데
  pointer head 는 순열등변이라, 같은 20분이 슬롯0 에서 +0.21σ, 슬롯46 에서 −6.07σ로 읽힌다.
  전역 상수로만 나눈다(`pad_vecnorm.py`).

### 5.4 규칙 합성 · 증류

```bash
# 교사 결정 로그에서 물리단위 임계값 채굴(λ, UAV 전환거리, 등급 임계)
python src/rl_src/v17_field_rules.py static   # 좌표별 병원 물리량
python src/rl_src/v17_field_rules.py table    # 결정 테이블
python src/rl_src/v17_field_rules.py mine lambda     # (redkm / yhold / stability 도 동일)

# 후보랭킹 트리 증류(병원 번호 비의존 → 병원 수·순서에 불변)
python src/rl_src/v10_tree_distill.py …
```

### 5.5 판정(폐루프 평가)

```bash
# 규칙 팔 — 정책 목록은 ';' 구분(규칙 이름에 쉼표가 들어간다)
python src/rl_src/v17_rule_eval.py \
  --manifest scenarios/manifests/sigungu30_test750_manifest.json \
  --policies "CARD_Q18=cardt:18,6.6,0,hingerate;CARD_P18=cardt:18,6.6,0,hingerate_psent;START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst" \
  --n_eps 30 --workers 44 --out results/scoreboard/v21/test750_rules.csv

# 학습 정책 팔 — 같은 rollout·seed 규약
python src/rl_src/v17_ppo_eval.py --model_dir results/rl/v19/national \
  --obs_variant field --policy_name PPO_NATIONAL --n_eps 30 --workers 44 \
  --manifest scenarios/manifests/sigungu30_test750_manifest.json \
  --out results/scoreboard/v21/test750_ppoN.csv

# 집계·판정·감사
python tools/v21_infoladder_report.py judge --base CARD_Q18
```

실험 드라이버(전수 스윕 · 사다리 · 재학습)는 `tools/exp_drivers/run_v{19,20,21}_*.sh` 에 있다.

---

<a id="design"></a>
## 6. 핵심 설계 — obs / action / reward / 마스킹

- **Action** `[class, dest, mode]` → Discrete **192 = 2 × 48 × 2**. `class` 0=Red/1=Yellow,
  `dest` 0=현장 대기·1..47=병원, `mode` 0=AMB/1=UAV. 한쪽 수단만 존재하면 자동 고정.
- **Observation** (dict → flat float32). 병원별 특징 F열 × H + 글로벌. 현행 `field` 는 차원 389.
  `MCI_H_PAD` 로 병원 수가 지역마다 달라도 고정 차원을 유지한다(패딩 병원은 마스크로 차단).
- **Reward** = 치료 개시 시각의 생존확률(Red/Yellow 시간감쇠, Green=1, Black=0).
  `--reward_mode pdrwog` 는 이를 예방가능 최대로 나눠 사고 규모에 불변인 0~1 값으로 만든다.
- **마스킹은 페널티가 아니라 하드 제약**이다(`action_masks()`): Red→Tier3 전용, UAV→헬기장 전용,
  발송 게이트. **규칙·MILP·트리도 전부 이 마스크에서만 후보를 고른다** — 비교 공정성의 핵심.
- 래퍼 체인(외→내): `Monitor → ActionMasker → [HeuristicAdvantage] → FlattenAndDiscrete
  (또는 HybridAMBHeur) → [RewardRedesign] → base env`. **코어는 수정하지 않고 감싼다.**

---

<a id="protocol"></a>
## 7. 판정 규약(재현성)

이 저장소가 지키는 규칙이고, 논문 수치가 여기서 나온다.

1. **좌표 분리** — 학습(train6000) / 튜닝(budget750) / 판정(test750) 좌표는 교집합 0.
   판정 좌표에서 파라미터를 고르면 누수다(v20 에서 실제로 한 번 발생 → budget750 재도출로 정정).
2. **CRN paired** — 모든 팔이 같은 시드·같은 시나리오를 본다. 판정선은 **두 팔 차이의 95%CI ≈ 0.00053**.
   학습 시드 잡음(0.00114)은 *규칙 실험에는 없는 잡음원*이라 규칙 대 규칙 판정에 쓰지 않는다.
3. **규칙 대 학습정책 비교는 30시드 이상 + 시드 부호 일관성**을 병기한다(시드축 CI 가 지역축과 같은 크기).
4. **최강 기준선과 비교한다** — 계열이 여러 개면 그중 최선과 비교한다(단일 시드/단일 계열 비교는 과대추정).
5. **부분집계로 판정하지 않는다** — 에피소드 비용이 난이도에 비례해 완료 순서가 난이도와 역상관이다.
6. **아키텍처 판정은 3시드 평균 대 시드 잡음 바닥**, 배수 ≥2. 훈련곡선 중간 우세는 채택 근거가 아니다.
7. **모방 정확도로 정책을 고르지 않는다** — 교사 행동 재현율과 폐루프 성능이 역전한 사례가 6번 나왔다.
8. 게이트 실패를 step 증가·threshold 완화로 우회하지 않는다.

---

<a id="envvars"></a>
## 8. 환경변수 레퍼런스

| 변수 | 값 | 뜻 |
|---|---|---|
| `MCI_OBS_VARIANT` | `field` / `essential+load+valid` / … | obs 구성. **학습↔평가 일치 필수**(호출자 책임) |
| `MCI_H_PAD` | 예 `47` | 병원 슬롯 패딩 상한. 미설정 = 구 동작 비트동일 |
| `MCI_REWARD_MODE` | `raw`/`woG`/`pdrwog`/`rywt` | 보상 재설계 |
| `MCI_CAP_GATE` | `occ`(기본) / `psent` | 발송 게이트 정의 = 통신축. obs·마스크·휴리스틱 4곳이 같은 정의 공유 |
| `MCI_CARED_OBS` | `1`(기본) / `0` | 병원 처치 완료 관측 여부. `psent` 와 짝지어 완전 통신단절 모델 |
| `MCI_INCIDENT_SIZE` `MCI_CAPA_SCALE` `MCI_AMB_NUM` `MCI_UAV_NUM` | 정수/실수 | 자원·부하 런타임 노브(시나리오 재생성 불요) |
| `MCI_AMB_VELOCITY` `MCI_UAV_VELOCITY` `MCI_AMB_HANDOVER` `MCI_UAV_HANDOVER` | 실수 | 물리축 노브. 미설정 = 구 동작 비트동일 |
| `MCI_OSRM_URL` `KAKAO_API_KEY` | URL / 키 | 시나리오 생성 라우팅 |

새 노브를 넣을 때의 규칙: **미설정이면 기존 경로와 비트동일**이어야 하고, 회귀 스모크로 그것을 증명한다.

---

<a id="unity"></a>
## 9. Unity 디지털트윈 시각화

시뮬레이션 결과를 **전국 3D 한국 지도** 위에 재생하는 Unity 프로젝트가 함께 있다.
RL/시뮬 코드와는 시나리오 데이터(`scene.json` + `trace_flat.json`)로만 연결된다.

### 9.1 재난 대응 트윈 — `UAV_test`

![MCI 디지털트윈 — 구급차·무인기 이송 재생](docs/assets/KoreaDigitalTwin_UAV.gif)

- **255 시군구 씬**을 필요한 지역만 additive 로드한다. 각 씬에는 정사영상·건물·OSM 도로·교통시설·
  공원/수계·POI(병원/학교/소방)가 메시로 구워져 있다(vWorld·OSM·DEM 수집 → 에디터 임포터 베이크).
- 좌표계는 시군구별 EPSG:5186 프레임(`RegionRegistry`)으로 WGS84 → Unity 월드(미터) 변환.
- `MapVersionSelector` 로 시나리오를 고르면 `TracePlayer` 가 AMB/UAV 배차·병원 처치·카메라를 재생하고,
  NPC 차량(응급차량 양보·신호 준수)과 보행자가 함께 돌아간다.
- eVTOL 에어앰뷸런스는 **무인기 설정**이라 조종사 없이 의사·환자·구급대원 3인 캐빈만 있고,
  계기는 아날로그 EFIS 가 아니라 **자율주행차식 인지 화면**(LiDAR 자유공간 조감도 · 자율성 상태 카드 ·
  깊이 카메라)이다.

### 9.2 자율주행 씬 — `CAR_test`

![강남 자율주행 씬 — 정밀도로지도 기반](docs/assets/KoreaDigitalTwin_ADS.gif)

- MCI 시뮬과는 **무관한 별도 씬**이지만 같은 GIS 파이프라인 산출물(정밀도로지도 차선그래프 LGV2,
  보행망, 표준링크)을 소비한다.
- 강남 110타일(EPSG:5186 11×10 km) k-ring 스트리밍. 지면 높이의 단일 출처는 **정밀도로지도 차선 z**다.
- 센서 리그: LiDAR16 · 77GHz 레이더 · GNSS/INS 융합 · IMU · 열화상 · 초음파 12구 · V2X SPaT ·
  전방 RGB · 스테레오 깊이. 차량 물리는 WheelCollider + ABS/TCS/ESC + 엔진 토크곡선 파워트레인.

### 9.3 이 저장소와의 관계 (중요)

`external/ml-agents` 는 **upstream Unity-Technologies/ml-agents 를 가리키는 서브모듈**이고,
Unity 프로젝트 `UAV_test/` · `CAR_test/` 는 그 서브모듈 작업트리 안에 **untracked 로** 존재한다.
따라서 **Unity C#·Assets 는 원격에 올라가지 않는다(의도된 구조)** — 이 저장소가 버전관리하는 것은
RL/시뮬 Python 코드다. Unity 쪽 아키텍처·임포터·함정 기록은 [`CLAUDE.unity.md`](CLAUDE.unity.md) 에 있다.

<!-- TODO(local): 아래는 로컬에서 채울 것 -->
- Unity 버전 / 렌더 파이프라인: *(TODO)*
- 프로젝트 열기·씬 실행 절차: *(TODO)*
- 위 영상 촬영 씬·구성: *(TODO)*
- 3D 자산·GIS 데이터 배포 가능 여부: *(TODO)*

---

<a id="docs"></a>
## 10. 문서와 기록

| 파일 | 내용 |
|---|---|
| [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md) | 연구 발전사 요약(무엇을 왜 바꿨는가) |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | 실험 이력·수치·**기각 근거** |
| [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) | 코딩 에이전트용 지침 겸 엔지니어링 로그 정본. 배관 계약·함정·판정 규약이 여기 누적된다 |
| [`CLAUDE.unity.md`](CLAUDE.unity.md) | Unity·GIS 파이프라인(로컬 전용 자산) |
| `scoreboard/v*_protocol.json` | 버전별 판정 프로토콜 — 방법 ID·평가셋 역할·제외 사유 |
| `archive/README.md` | 종결 자산·일회성 코드 보관 원장(복원 경로 포함) |

보고서 본문(`docs/*.md`)과 산출 CSV(`results/`)는 용량·저작물 사유로 로컬 전용이다.
수치가 필요하면 위 두 이력 문서와 `CLAUDE.md` 에 요약돼 있다.

---

<a id="license"></a>
## 11. 데이터 출처 · 라이선스

- 병원 풀·안전센터/소방서·시도 경계는 국내 공공데이터, 도로 경로는 OSM/OSRM 또는 Kakao Mobility API,
  Unity 트윈의 정사영상·건물·DEM·정밀도로지도는 vWorld·국토지리정보원 계열 자료를 수집해 사용한다.
  **원자료의 재배포 조건은 각 제공기관 약관을 따른다** — 이 저장소는 스크립트만 버전관리하고
  대용량 원자료·가공 산출물은 포함하지 않는다.
- 코드 라이선스: *(TODO — 논문·특허 진행 상태에 맞춰 결정)*
- 인용: *(TODO — 논문 게재 후 BibTeX 추가)*

<!--
TODO(local): LICENSE 파일, CITATION.cff, 논문 링크
-->
