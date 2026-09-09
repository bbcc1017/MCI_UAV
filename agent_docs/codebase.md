# 코드 지도와 변경 계약

확인: 2026-09-09. 파일의 버전 접두사보다 현재 호출 관계가 우선이다.
`v17_*`는 v21의 현행 진입점이고, 오래된 클래스도 저장 모델 역직렬화 때문에 살아 있다.

## 데이터 흐름과 수정 위치

`좌표/원장 → 시나리오 YAML·거리행렬 → base env → 특징/보상/마스크 wrapper → 정책 → 폐루프 평가 → 집계`

| 범위 | 먼저 볼 코드 | 맡는 일 |
|---|---|---|
| 시나리오 | `src/sce_src/make_csv_yaml_dynamic.py`, `gen_sigungu30_osrm.py`, `split_sigungu30.py` | 병원/자원 선택, 거리 생성, 좌표 분할 |
| 원본 시뮬 | `src/sim_src/{ScenarioManager,EntityManager,EventManager,MCIEnvironment_gymnasium}.py` | 설정·상태·이벤트·gym 경계 |
| 환경 생성 | `src/rl_src/env_factory.py`, `viper_distill.py::make_feature_env` | 설정 로드, 평가 env 캐시·정규화 |
| 행동/특징 | `env_wrapper.py`, `hospital_feature_wrapper.py`, `aggregate_obs.py` (`src/rl_src/`) | flatten/codec/mask, 병원 토큰·글로벌 관측 |
| 학습 | `src/rl_src/train_ppo_feature.py` | `FeatureMultiRegionEnv`, worker shard, 모델·vecnorm·메타 저장 |
| 신경망 | `src/rl_src/pointer_policy.py`, `pad_vecnorm.py` | 병원 랭킹 head, 정규화 면제 |
| 현장 규칙 | `src/rl_src/v17_field_rules.py`, `score_features.py` | 물리 특징·CARD·채굴 |
| 기준선 | `lb3_policy.py`, `loadbalance_heuristic.py`, `v10_full_baselines.py`, `v16_baseline_alignment.py`, `shin_full_baselines.py` (`src/rl_src/`) | LB·Full64·Shin 비교군 |
| 계획/증류 | `planner_policy.py`, `milp_policy.py`, `tree_distill_policy.py`, `v10_tree_distill.py` (`src/rl_src/`) | NCRP·OR·후보랭킹 학생; 기존 연구 재현 |
| 평가/리포트 | `src/rl_src/v17_{rule,ppo}_eval.py`, `paired_eval_ladder.py`; `tools/v{20_threshold,21_infoladder}_report.py` | 에피소드 CSV, paired 판정 |
| 기전 | `src/rl_src/v20_mechanism_eval.py`, `tools/v20_mechanism_report.py` | 치료개시 지연·손실 항등식 |
| 가속 | `src/sim_src_upgrade/` | 원본 결과와 동치인 별도 코어·런처·검증 |

`tools/exp_drivers/`는 배치 명령의 근거다. 과거 예제에는 이동된 스크립트나 다른 버전 기본값이
있으므로 파일 존재·argparse·실제 메타를 함께 본다. flat import 구조를 유지한다.
정리 시 import뿐 아니라 subprocess·셸 경로 호출·pickle 클래스 참조도 조사한다.
`score_cma`와 `exit_distill`은 현행 planner/라벨 모듈에서 참조되므로 제거하지 않는다.

## 행동·마스크

- 논리 행동 `[class,dest,mode]`: Red=0/Yellow=1, dest0=현장대기·1..H=병원, AMB=0/UAV=1.
  G/B는 RL 행동에서 제외되고 시뮬 코어가 처리한다.
- 두 수단·H47이면 action192. 한 수단만 있으면 auto-pin되어 action96이며 pointer는 이 경로를
  지원하지 않는다. 모델 head 지원을 먼저 확인한다. deepsets도 `valid`와 무조건 호환되지 않는다.
- `env_wrapper.encode_action/decode_action` 또는
  `loadbalance_heuristic._codec_from_mask(mask_len,H_layout)`을 재사용한다.
  실병원수 `H_real`과 패딩 후 `H_layout`을 구별한다. `a//96` 같은 수식을 새 코드에 복사하지 않는다.
- 적격성은 `action_masks()`가 결정한다(Red tier3·UAV helipad·환자/차량 가용성·용량).
  같은 제약을 다른 정책에서 별도 근사하거나 페널티로 바꾸지 않는다.
- `MCI_H_PAD`는 wrapper의 obs/action/mask 레이아웃만 늘린다. sim은 실제 H로 실행,
  패딩 목적지는 마스크+step 가드로 차단, 실H>H_pad는 오류. 자연-H는 유효열/마스크드 pooling까지 필요하다.

## 관측·모델 호환성

H47에서의 레이아웃. 주석의 “7열” 같은 설명보다 `_FIELD_COLS`, env shape, 모델 메타가 우선이다.

| variant | 병원 F | 글로벌 | 전체 dim | 용도 |
|---|---:|---:|---:|---|
| `essential` | 4 | 21 | 209 | 초기 재현 |
| `essential+load` | 7 | 26 | 355 | v2~v5 |
| `essential+load+valid` | 8 | 26 | 402 | v6~v18 및 공통 규칙 평가 wrapper |
| **`field`** | **8** | **13** | **389** | **v19/v20 교사 실제 설정** |
| `field+valid` | 9 | 13 | 436 | 별도 레이아웃, 기존 field 모델과 비호환 |

`field` 병원 열 순서 = `[is_tier3,n_or,n_bed,eta_amb,eta_uav,in_flight_amb,in_flight_uav,delivered]`.
ETA는 인계시간을 포함하며 /60, 병상수 /31, 나머지 /1의 공통 상수를 쓴다.
글로벌은 구조·분류 완료 R/Y 현장 대기, 이송중 등급×수단, 차량 국면, 시계(13열)다.
병원 실시간 census, 이미 샘플된 차량 잔여시간, 미구조 환자 등급은 field obs에서 제외한다.

**병원 블록 전체를 슬롯별 VecNormalize에서 면제**한다(`train_ppo_feature`의 `field_variant` 분기).
v19 field는 앞 376열 면제·글로벌13 정규화다. 같은 ETA가 병원 슬롯마다 다른 z-score가 되면
공유 pointer scorer의 의미가 깨진다. valid 사용 시 유효열도 원값0/1을 보존한다.

모델 로드 전후 확인:

1. `meta.json`의 obs_variant/H/F/dim/head/seed와 실제 zip의 policy kwargs를 확인한다.
   v19 기준 `results/rl/v19/national`·`sido_*`: field389/action192/attention0.
2. `pointer_policy`, `hospital_set_extractor`, `pad_vecnorm` 등 해당 저장 클래스가 import 가능해야 한다.
3. **같은 checkpoint 시점의** 모델과 vecnorm을 짝지어 동결 로드한다.
   `viper_distill.load_vecnorm`은 `(mean,std,clip,exempt)`를 반환하고 `_NormObs`가 면제열을 복원한다.
4. 언피클 VecNormalize의 없는 속성에 `getattr`을 쓰면 SB3 재귀에 빠질 수 있다.
   해당 상태 조회는 `obj.__dict__.get(...)` 계약을 따른다.
5. resume은 lr anneal이 거의0으로 복원될 수 있다. fine-tune 시 실제 lr/KL을 확인한다.
   `--init_from`과 `--resume_from`은 optimizer·step 이력이 다르다.

`train_ppo_feature`는 `MCI_OBS_VARIANT`를 자동 export하지 않는다. 호출자가 명시한다.
`v17_ppo_eval` 기본값은 여전히 v10 모델·대표점250·essential+load+valid이므로 v19 실행에서는
`--manifest`, `--model_dir`, `--obs_variant field`, `--policy_name`을 명시한다.
규칙 평가는 모델 obs를 읽지 않아 `v17_rule_eval`의 essential+load+valid wrapper를 유지한다.

## 보상·용량·정보

- `PDR_woG = 1 − reward_woG / preventable_woG`. 분모는 reset 시 정해진다.
  크기가 다른 사고에서는 raw reward/woG 합을 직접 비교하지 않는다.
- 보상의 `p_admit`은 **수술실 배정으로 치료를 시작한 시각**이다. 병원 도착/인계 후 병상에서
  대기할 수 있다. `EventManager.ev_p_care_ready`·`ev_p_def_care`의 기록을 확인한다.
- 용량 세 층: 발송/마스크의 `max_send`, 물리 수용의 수술실+병상,
  치료개시의 `n_idle>0`(수술실). 큰 수용량이 대기 병목 부재를 뜻하지 않는다.
  `score_features.compute_static()['max_capa']`는 `hos_max_capa`(수술실수)다.
- `MCI_CAP_GATE=occ`는 census+in-flight, `psent`는 누적발송을 쓴다.
  완전 현장정보 실험은 `MCI_CARED_OBS=0`와 마스크까지 함께 감사한다.
  CARD_P의 부하신호 교체와 전체 gate를 psent로 바꾸는 실험은 서로 다르다.
- `raw/woG/pdrwog/rywt`와 기각 실험 `pdrwog_da`의 의미는 `reward_redesign_wrapper.py`가 정본.
  변형 추가는 wrapper/명시 옵션으로, 기본 결과 불변을 확인한다.

## 런타임 노브

`ScenarioManager.py`에서 읽으며 **환경 생성/캐시 전** 설정한다.
명시한 인자뿐 아니라 상속된 `MCI_*`도 확인·기록한다. 비밀 API 키는 출력/메타에 쓰지 않는다.

| 변수 | 의미/주의 |
|---|---|
| `MCI_INCIDENT_SIZE` | 고정 인프라에서 환자 부하 변주 |
| `MCI_AMB_NUM`, `MCI_UAV_NUM` | 차량/출발지 풀 슬라이스; UAV 수를 줄여도 착륙 가능한 병원 집합은 그대로 |
| `MCI_CAPA_SCALE` | 수술실·병상 스케일, 최소1 floor; 발송만 조이는 max_send_coeff와 다름 |
| `MCI_AMB_VELOCITY`, `MCI_UAV_VELOCITY` | 속도; 거리 기반 OSRM 시나리오에서 런타임 적용 |
| `MCI_AMB_HANDOVER`, `MCI_UAV_HANDOVER` | 인계시간; Kakao `is_use_time=True`의 부분 적용 여부 별도 확인 |
| `MCI_CAP_GATE`, `MCI_CARED_OBS` | gate·관측 정보축 |
| `MCI_H_PAD`, `MCI_OBS_VARIANT`, `MCI_REWARD_MODE` | 학습↔평가 호환 계약 |

시나리오 생성은 OSRM/Kakao를 쓰지만 학습·평가는 저장 거리행렬을 읽는다.
API 호출이 필요 없는 작업에서 시나리오를 재생성하지 않는다.

## 시뮬 변경·가속·trace

`src/sim_src/`가 정본이다. 성능/로그 편의를 위해 원본을 임의 재설계하지 않는다.
요청된 버그 수정·물리 노브·계측 변경은 영향 범위를 좁혀 수행하고 가속 사본도 대조한다.
시뮬을 바꾸지 말라는 원칙을 필요한 버그 수정의 영구 금지로 해석하지 않는다.

가속 작업 전 [고속 경로 README](../src/sim_src_upgrade/README.md)의 해당 절과 실제 런처를 읽는다.
`origin_sync.py --check` 실패를 `--write`나 `--skip_preflight`로 숨기지 않는다.
먼저 사본을 동기화·검증한 뒤 의도한 기준 hash를 갱신한다.
2026-09-09 점검에서는 EventManager·ScenarioManager의 G0 불일치가 있었다.
현재 상태는 `--check`로 다시 확인한다([개편 검토 기록](2026-09-09-review.md)).
`--mask_only`는 **실제 obs를 읽지 않는 허용목록 드라이버만** 사용한다.
신경망·flat obs 트리·데이터수집에는 사용할 수 없다. 이름이 규칙/트리라고 안전한 것은 아니다.

검증은 변경에 맞게 선택한다: 궤적/지표=`verify_equivalence`, obs=`verify_obs_patch`/`verify_rl_obs`,
복제/planner=`verify_deepcopy`, 학습 경로=`verify_train_equiv`, 정책·gate 조합=`coverage_matrix`.
BLAS 스레드·device·dtype까지 통제한다. float64 동치와 float32 저장 허용오차를 혼동하지 않는다.

trace 계측은 reset **전** 활성화한다. 초기 rescue가 reset 내부에서 시작된다.
병상 대기 후 치료개시(`ev_p_def_care`, `from_queue`)를 포함해야 한다.
`PDR = 미진입손실 + 지연손실` 항등식과 trace on/off 결과 불변을 검증한다.
2026-09-07 보완 전 trace로 기다린 환자 통계를 추정하지 않는다.
기존 v20 조건에서 미진입손실0이라는 사실을 모든 미래 조건에 일반화하지 않는다.
