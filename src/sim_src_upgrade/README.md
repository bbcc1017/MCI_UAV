# sim_src_upgrade — 결과 동일, 연산시간만 단축

`src/sim_src/` 는 **정본이고 절대 수정하지 않는다.** 이 폴더는 같은 로직의 별도 실행경로다.
목표는 하나뿐이다: **같은 조건에서 비트 단위로 같은 결과, 더 짧은 시간.**

## 왜 만들었나

대규모 실험이 CPU에 막혀 있었다. v16/v17급 전수평가(1.01억 에피소드)가 약 2,800 core-h,
PPO 10M 학습 1런이 7.5h. 공유 노드라 코어 병렬은 이미 포화(loadavg 120+) → **코어당 속도**가
남은 유일한 레버였다.

프로파일 결과 원인은 알고리즘이 아니라 구현 오버헤드였다. 작은 배열(H=47, AMB 30, UAV 26,
환자 100)에 numpy를 이벤트마다 재호출하고, 매번 집계를 처음부터 다시 계산하고 있었다.

## 실측 (인터리브 min-of-N, 종로구, 지표 서명 전 변형 동일)

| 경로 | 배속 | 비고 |
|---|---|---|
| 규칙·트리 정책 스윕 | **3.65×** | `new_mask` (특징 obs 생성 생략) |
| RL 관측 경로 (env.step) | **2.23×** | `new_obs` (관측 전부 생성) |
| **v16 드라이버 전체(wall)** | **3.34×** | 274s → 82s, 산출 NPZ 비트동일 |
| **v10 드라이버 전체(wall)** | **2.86×** | 109s → 38s, 산출 NPZ 비트동일 |
| **PPO 학습 (프로덕션 레시피·subproc)** | **1.31×** | 225.5s → 172.3s (50k steps). 고정오버헤드 제외 시 1.37× |
| PPO 학습 (프로덕션 레시피·dummy) | 1.49× | 82.5s → 55.2s (20k steps) |
| NCRP 플래너 | 1.37× | K4·h5·m2 에서 170→124 ms/dec. 롤아웃 중 **모델 forward** 는 안 빨라져 배속이 낮다 |

환산: v16급 1.01억 에피소드 약 50h → **약 15h**.

### 학습 배속이 낮은 이유 — 구간 분해 (프로덕션 레시피, `--vec dummy`, 20k steps)

| 구간 | 구 코어 | 신 코어 | 배속 |
|---|---|---|---|
| **wall** | **82.5s** | **55.2s** | **1.49×** |
| ├ `collect_rollouts` | 71.6s (86.8%) | 44.8s (81.0%) | 1.60× |
| │ ├ **env.step 합** | **42.9s (52.0%)** | **19.1s (34.5%)** | **2.25×** |
| │ └ 정책 forward+버퍼 | 28.7s (34.8%) | 25.7s (46.5%) | 1.12× |
| ├ `train` (경사갱신) | 4.4s (5.3%) | 4.5s (8.1%) | 1.00× |
| └ 그밖(셋업·저장) | 6.5s | 6.0s | — |

env.step 자체는 **2.25×**(독립 벤치 2.23× 와 일치)인데 학습 전체는 1.3~1.5× 다.
env.step 이 학습 wall 의 52% 뿐이고 정책 forward·버퍼·경사갱신은 안 빨라지기 때문이다.
`--vec subproc` 은 8 env 가 병렬이라 env.step 의 wall 기여가 더 줄고, 대신 IPC·스케줄링이
자리를 차지해 배속이 1.31× 로 더 내려간다(부하 높은 공유 노드에서 특히).

⚠️ **측정 조건이 결론을 바꾼다**: 같은 학습을 비프로덕션 하이퍼(`batch 128`)로 재면 1.02× 가
나온다 — batch 가 작으면 경사 스텝이 4배 많아져 sim 비중이 눌린다. 배속을 인용할 때는
**하이퍼파라미터·vec 방식·노드 부하를 함께 적어야 한다**.
(부수 관찰: 정상상태 환산으로 신 코어+dummy 2.46 ms/step 이 구 코어+subproc 2.7 ms/step 보다
빠를 여지가 있다 — 공유 노드에서 dummy 가 유리할 수 있으나 미검증.)

## 동등비교 — "기존 휴리스틱 결과와 같은가, 시간만 줄었나"

`verify/head_to_head_heur.py` 로 **같은 좌표·같은 시드·같은 규칙 64개**를 구/신 코어에서
각각 별도 프로세스로 통째로 돌려 비교했다(v10 드라이버의 휴리스틱 경로 그대로).

### ① 서울시청 좌표 신규 동등비교 — Full64 × 30에피 (1,920 에피소드)

| 실행 | wall | 평균 PDR_woG | Best-of-64 |
|---|---|---|---|
| 구 코어(현행) | 253.0s | 0.2060837746 | 0.1100968122 |
| 신 코어(obs 전부 생성) | **126.7s (2.00×)** | 0.2060837746 | 0.1100968122 |
| 신 코어(`--mask_only`) | **67.6s (3.74×)** | 0.2060837746 | 0.1100968122 |

지표 배열 `(64규칙 × 30에피 × 5지표)` = **9,600개 원소 전부 비트동일, 최대차이 0**.

### ② 이미 나와 있는 결과와의 대조 (가장 중요)

지금 돌고 있는 **v17 HEUR64 전수평가가 원본 코드로 디스크에 써 둔 체크포인트**와 직접 대조했다.
무작위로 고른 완료 좌표 5곳, 각 64규칙 × 앞 30시드:

| 좌표 | 판정 | 최대차이 |
|---|---|---|
| 강남구_11680_p0 | PASS | 0 |
| 서구_26140_p0 | PASS | 0 |
| 중구_28110_p1 | PASS | 0 |
| 송파구_11710_p3 | PASS | 0 |
| 중구_27110_p0 | PASS | 0 |

같은 좌표(강남구_p0)의 시간 짝: 구 코어 **262.4s** → 신 코어 **59.3s (4.42×)**.
그리고 **구 코어로 돌린 쪽도 동결값과 정확히 일치**했다 — 이 하네스가 프로덕션과 같은 경로를
쓰고 있다는 뜻이라, 위 비교가 사과 대 사과임을 보증한다.

```bash
python src/sim_src_upgrade/verify/head_to_head_heur.py --core old  --region 서울 --n_eps 30
python src/sim_src_upgrade/verify/head_to_head_heur.py --core fast --mask_only --region 서울 --n_eps 30
python src/sim_src_upgrade/verify/head_to_head_heur.py --compare <old.npz> <fast.npz>
# 동결 결과 대조
python src/sim_src_upgrade/verify/head_to_head_heur.py --core fast --mask_only \
    --manifest train1000 --region 강남구_11680_p0 --n_eps 30 \
    --frozen results/scoreboard/v17/heur64_eta_aligned_full1000/work/heur/train1000/강남구_11680_p0.npz
```

## 쓰는 법

### 기존 드라이버를 그대로 가속 (권장)

드라이버는 한 줄도 안 고친다. 런처가 `make_feature_env`·`make_base_env`·규칙 클래스만
고속판으로 재바인딩하고 `main()` 을 부른다. 체크포인트·CSV·집계 로직은 원본 그대로다.

```bash
# 규칙/트리 정책 전수평가 (obs 생성 생략 가능 → 가장 빠름)
python src/sim_src_upgrade/drivers/run_fast.py --target v16_baseline_alignment --mask_only -- \
    --out_dir results/scoreboard/v17/... --n_eps 1000 --workers 56 --phase run

python src/sim_src_upgrade/drivers/run_fast.py --target v10_full_baselines --mask_only -- \
    --workers 104 --n_eps 1000

# 모델(신경망)이 obs 를 쓰는 드라이버는 --mask_only 를 빼야 한다
python src/sim_src_upgrade/drivers/run_fast.py --target paired_eval_ladder -- <인자>
```

* 작업 시작 전 **사전점검**이 자동으로 돈다(한 좌표 × 2규칙 × 3시드 지표 완전일치).
  실패하면 그 자리에서 멈춘다. `--skip_preflight` 로만 생략된다.
* `src/sim_src` 소스 해시가 그대로라 **진행 중이던 원본 실행의 체크포인트를 이어받아** 계속
  돌릴 수 있다(결과가 비트동일이므로 섞여도 무방).
* ⚠️ **`--mask_only` 는 규칙·트리 정책 전용**이다. 신경망 정책에 쓰면 obs 가 0 벡터다.
  규칙정책은 `env.unwrapped.en_manager.get_full_obs()` 를 직접 읽으므로 402차원 벡터가
  아예 필요 없다 — 그래서 규칙 전수평가에서는 항상 켜는 게 맞다.

### 어느 경로가 지원되나

| 경로 | 시작방식 | 지원 | 배속 |
|---|---|---|---|
| 규칙·휴리스틱 전수평가 (`v10_full_baselines`·`v16_baseline_alignment`·`shin_full_baselines`) | fork Pool | ✅ `--mask_only` | 3.7~4.4× |
| **PPO 평가·추론** (`paired_eval_ladder`·`planner_eval`·`v10_student_suite`·`sim_logger`·`score_eval`…) | fork Pool | ✅ (mask_only 금지) | ~2.2× |
| **PPO 학습** (`train_ppo_feature`) | SB3 `SubprocVecEnv` = **forkserver** | ✅ `fastcore_boot` 경유 | ~2.2× |

fork 자식은 부모의 패치를 메모리째 물려받는다. forkserver/spawn 자식은 인터프리터를 새로
띄우므로 `fastcore_boot/sitecustomize.py` 가 부팅 시 패치를 다시 건다 — 런처가
`PYTHONPATH` 와 `MCI_FASTCORE=1` 을 심고, 그 환경변수는 자식에게 상속된다.
자식에서 패치가 실패하면 **원본 코어로 조용히 되돌리고 경고만** 남긴다(결과는 동일, 속도만 손해).

### 이미 돌고 있는 전수평가에 갈아타기 (검증됨)

`src/sim_src` 를 안 고쳤으므로 **체크포인트 소스 해시가 그대로**라 진행 중 실행을 이어받을 수 있다.
실증(스크래치 디렉터리에 완료 체크포인트를 배치하고 고속 런처 실행):

```
v10_full_baselines      [heur] completed_reused=2 · 실행 job=0 · 16초
v16_baseline_alignment  [v16] 모든 체크포인트 완료 — 실행 job=0 · 13초
```

* v16 은 체크포인트에 `source_bundle_sha256` 을 박아 검증한다 → 저장값과 현재 계산값 **MATCH** 확인.
* v10 은 해시를 안 쓰고 `shape·seeds·rule_names·done·유한성` 만 본다(`v10_full_baselines.py:221-236`)
  → 조건 없이 재개된다.
* `t4` phase 도 실전 규모(1,000ep)로 구/신 대조 완료: 118s → 45s, NPZ **maxΔ=0**.

**절차**: ①`origin_sync.py --check` ②남은 job 수를 먼저 센다 ③out_dir 을 포함한 정밀 `pkill`
(⚠️스크립트명만 매칭하면 다른 실험까지 죽는다) ④런처로 재개 ⑤**로그의 `jobs=` 가 ②에서 센 값과
근사한지 확인** — 전체 수(1250·2500)로 찍히면 체크포인트를 못 읽은 것이므로 즉시 롤백한다.
기존 NPZ 는 읽기만 하므로 손상되지 않는다.

⚠️ **프로베넌스**: `protocol_meta.json` 은 어느 코어로 계산했는지 남기지 않는다. 전환 시
out_dir 에 `CORE_SWITCH_NOTE.md`(전환 시각·전환 시점 진행분·등가성 근거)를 남길 것.

### 직접 쓰기

```python
from sim_src_upgrade.env_factory_fast import make_feature_env_fast
from sim_src_upgrade import fast_obs_patch
fast_obs_patch.apply()                       # rl_src 관측 핫스팟 등가 교체
factory = make_feature_env_fast(cfg_path, mask_only=True)
```

## 무엇을 바꿨나 (전부 "로직 동일")

| 항목 | 원본 | 바꾼 것 |
|---|---|---|
| S1-1 이송중 카운트 | 차량 56대 파이썬 루프 | `bincount` (판정 순서 보존) |
| S1-2 GB 일괄이송 후보 | 호출마다 `argsort(47)` + `set(numpy)` 2개 | 시나리오당 1회 사전계산 |
| S1-3 디버그 출력 | 이벤트마다 `print(c_event)` | `TRACE_PRINT=False` 게이트 |
| S1-4 `_patient_agg` | `np.sum` 20회 | `bincount` 1회 (`fast_obs_patch`) |
| S1-5 규칙정책 obs | 402차원 벡터 매 스텝 생성 | `MaskOnlyFeatureWrapper` 로 생략 |
| S2-6 종료·구조 판정 | 이벤트마다 `np.all` 전수 스캔 | 증분 카운터 (`_n_cared`/`_n_rescued`) |
| S2-7 이송중 | 호출마다 전수 재계산 | 증분 카운터 `_in_flight_cnt` (배차/도착 시 ±1) |
| S2-8 결합 mask | `2×(H+1)×2` 파이썬 3중 루프 | numpy 브로드캐스트 |
| S2-9 `patient_info` | 이벤트마다 DataFrame 라벨 조회 | `ScenarioManager` 에서 순수 파이썬 사전 추출 |
| S2-11 차량 잔여시간 | dict 4단 조회 + 슬라이스 재생성 | reset 때 열 뷰 캐시 |
| S2-13 `_fleet_agg` | 불필요한 복사·정수 카운트 | 축소 (⚠️ `min`/`mean` 은 **원본 호출 그대로**) |

### 일부러 **안** 건드린 것

부동소수 연산 순서나 tie-break 가 바뀌면 비교가 뒤집혀 궤적이 갈린다. 그래서 금지:

* 차량 잔여시간을 절대시각(`t_abs - now`)으로 전환 — 반복 감산의 누적오차가 달라진다
* `room = (max_capa + max_queue) - occ - in_flight` 연산 순서 변경
* `np.argsort` kind, `p_wait[...].pop()` 순서, 정렬 안정성 가정
* RNG 드로우 추가/제거/재배치
* 이벤트 디스패치 `getattr(self, "ev_"+name)` → dict (측정해보니 전체의 0.1% 라 무의미)

### 기각한 최적화 (측정 근거)

**`en_properties` 공유형 deepcopy** — 플래너가 결정마다 env 를 복제하니 정적 대형 배열
(`d_HtoH` 47×47 등)을 공유하면 이득이 클 줄 알았다. 실측하니 **deepcopy 2.83ms 중
`en_properties` 는 0.28ms(10%)** 뿐이고, 나머지는 공유가 불가능한 가변 상태
(`en_status`·이벤트 큐·rng)였다. NCRP 결정 하나에서 롤아웃 스텝이 비용의 약 91%
(2560스텝×1.5ms vs 128복제×2.83ms)라 복제 최적화의 상한 자체가 낮다. 위험 대비 이득이
없어 **기각**하고, 플래너는 스텝 가속(2.2×)으로만 이득을 본다.

## 이전보다 얻을 수 **없는** 정보 (정직한 목록)

계산 결과·지표는 하나도 잃지 않는다. 잃는 것은 **부수 산출물** 두 개이고, 둘 다 되돌릴 수 있다.

### 1) 이벤트별 raw print 스트림 — 기본 OFF (`TRACE_PRINT=False`)

원본은 이벤트마다 `print(c_event)`(=`(시각, 이벤트ID, 이벤트명, entity_idx)`)와
`print("Action:", action)` 을 뿜었다. 대부분의 드라이버는 이 stdout 을 `/dev/null` 로 버리므로
실질적으로 아무도 안 읽는다. 다만 `src/sim_src/main.py` 로 직접 돌릴 때는 `_Tee` 가
`experiment_logs/sim_*.log` 에 저장하므로 **거기서는 정보였다**.

* 되살리기: `core/EventManager.py` 의 `TRACE_PRINT = True`
* 원본 CLI(`src/sim_src/main.py`)는 **손대지 않았으므로 그대로 전부 남는다** (사본에는 main.py 없음)
* **진단용 print 는 끄지 않았다** — `NO UAV`/`NO AMB`/`NO HELIPAD`/`NO PATIENT`/`EventQueueEmpty`
  는 원본과 똑같이 출력된다(희귀 경로라 비용이 없다)
* 구조화 트레이스(`enable_trace` → `get_trace()`)는 **무수정**이며 이쪽이 정본이다.
  커버: `onset`·`rescue`·`transport_start`·`hospital_arrival`·`diversion`·`care_start`·`care_complete`.
  ⚠️ raw print 에는 있었지만 구조화 트레이스에 **없는** 것: 차량의 **현장 복귀 도착**
  (`amb_arrival_site`/`uav_arrival_site`)과 이벤트ID·전역 팝 순서. 이게 필요하면 `TRACE_PRINT=True`.

### 2) 특징 obs 402벡터 — `--mask_only` 를 **줬을 때만**

옵션이며 기본 OFF다. 켜면 그 실행에서 402차원 벡터가 안 만들어진다.
sim 상태·action mask·`env.unwrapped.en_manager.get_full_obs()` 는 전부 그대로다.

증류 관점에서 중요한 사실:

| 증류 자산 | 특징 원천 | `--mask_only` 영향 |
|---|---|---|
| v10 Track D/E 후보랭킹 트리 (`tree_distill_policy`) | `env.unwrapped` + `en_properties` | **없음** — 같이 써도 된다 |
| `score_features.py` (φ12) | 같음 (평탄 obs 슬라이스 금지가 원칙) | **없음** |
| PPO 교사 라벨 (`bc_dataset`·`exit_labels`·`ncrp_labels`·`collect_decisions`·`distill_zoo`) | 평탄 obs 402 | **치명적** — 쓰면 안 됨 |
| 구 VIPER 슬롯트리 (`viper_distill.make_tree_policy`) | 평탄 obs 402 | **치명적** |
| 신경망 정책 평가 (`paired_eval_ladder`·`planner_eval`) | 평탄 obs 402 | **치명적** |

그래서 세 겹으로 막았다:

1. **런처 허용목록** — `--mask_only` 는 `MASK_ONLY_SAFE`(규칙 전용 드라이버 6종)에서만 통한다.
   그 밖이면 실행을 거부하고 이유를 찍는다. `--force_mask_only` 로만 우회.
2. **obs 자리에 NaN 센티넬** — 0 벡터는 "그럴듯한 관측"처럼 보여 조용히 잘못된 결과를 만든다.
   NaN 은 신경망에서 즉시 예외(`MaskableCategorical` logits 검증에서 터짐).
   ⚠️ 단 **sklearn 트리는 NaN 을 결측치로 받아들여 예외를 내지 않는다** — 그래서 1번 허용목록이 본 방어선이다.
3. **`mask_only` + vecnorm 동시 사용 금지** — 정규화는 관측을 쓴다는 뜻이라 생성 시 예외.

### 3) 그밖에 바뀐 것 (정보 손실 아님, 참고)

* `check_termination` 이 `np.bool_` → 파이썬 `bool` 반환. 값 동일.
* `default_transportation_GB` 가 `np.int64` → 파이썬 `int` 반환. 값 동일(`_record_trace` 는 어차피 `int()` 캐스팅).
* `en_properties['patient']['patient_info_fast']` 키 **추가**(원본 `patient_info` DataFrame 은 그대로 남음).
* `EventManager`/`EntityManager` 에 내부 캐시 속성 추가(`_gb_order`·`_n_cared`·`_in_flight_cnt`·`_amb_t`·`_uav_t`).
* ⚠️ 고속 코어로 **pickle 한 env 는 원본 코어에서 못 읽는다**(모듈 경로가 다르다).
  env 를 파일로 저장하는 경로는 없으므로 실사용 영향은 없다.

## 학습 device·스레드 실측 (2026-08-11, loadavg 180 · 신 코어)

sim 가속과 별개로 "학습을 CPU/GPU 중 무엇으로, 스레드 몇 개로 돌려야 하나"를 재봤다.

**구간 분해** (`profile_train.py`, dummy vec, 20k steps, 프로덕션 레시피)

| 조건 | wall | env.step | 정책 fwd+버퍼 | **train(경사갱신)** |
|---|---|---|---|---|
| GPU · t=1 | 55.2s | 19.1s | 25.7s | **4.5s** |
| CPU · t=1 | 104.4s | 18.4s | 15.8s | **66.0s** |
| CPU · t=8 | **53.1s** | 15.7s | 17.1s | **16.4s** |

**전체 wall** (`subproc`, n_envs 8, 50k steps): CPU t=1 292s · CPU t=8 **202s** ·
CPU t=16 282s · GPU t=1 **131s**

읽는 법:

1. **`env.step` 은 device 무관**(18~19s 고정) — 시뮬레이터는 CPU 에서 돈다. 교수님의
   "RL 은 CPU 기반" 진단이 이 구간을 정확히 가리킨다. sim 최적화가 유효한 이유도 이것.
2. **경사갱신만 병렬화에 극적으로 반응**: 66.0s(1스레드) → 16.4s(8스레드) → 4.5s(GPU).
   ⚠️ **CPU + 1스레드가 최악**이다. PPO 는 배치 512 행렬연산을 하므로 GPU 든 멀티스레드 든
   하나는 반드시 줘야 한다.
3. **롤아웃 forward 는 CPU 가 더 빠르다**(15.8~17.1s vs GPU 25.7s) — 배치가 `n_envs`=8 뿐이라
   GPU 는 커널 실행 오버헤드에 묶인다. 한 학습 안에 GPU 유리 구간과 CPU 유리 구간이 공존한다.
4. `t=16` 이 `t=8` 보다 느리다(282 vs 202s) — 공유 노드에서 스레드 과다는 역효과.
5. `dummy` 에서는 CPU t=8 ≈ GPU 인데 `subproc` 에서는 GPU 가 1.5× 앞선다 — 메인의 BLAS 8스레드가
   env 워커 8개와 CPU 를 다투기 때문. ⚠️ loadavg 180 에서 잰 값이라 절대치는 ±30% 잡음.

**권고**

| 상황 | 권고 |
|---|---|
| 단일 학습 런 | **GPU + `OMP_NUM_THREADS=1`** — `tools/exp_drivers/run_v12_*.sh` 가 이미 이 조합이다(바꿀 것 없음) |
| GPU 못 쓸 때 | **`OMP_NUM_THREADS=8`** (1도 16도 아님) |
| 다수 런 동시(51-way 그리드) | CPU + t=1 유지 — GPU 1장을 51개가 다투는 것보다 낫다. 단 **개별 런의 경사갱신이 약 14배 느린 상태**임을 인지할 것 (`run_grid_parallel.py:102,106-108`) |

### ⚠️ 재현성 — device 만의 문제가 아니다 (실측)

같은 시드·같은 설정·같은 코어로 50k 학습한 모델 4개를 서로 대조했다.

| 비교 | 다른 텐서 | 절대 최대차이 |
|---|---|---|
| GPU vs CPU(t=1) | 46/46 | 0.165 |
| **CPU(t=1) vs CPU(t=8)** — device 동일 | **46/46** | **0.112** |
| **CPU(t=8) vs CPU(t=16)** — device 동일 | **46/46** | **0.075** |
| *(대조군)* GPU 구코어 vs GPU 신코어 | **0/46** | **0** |

**device 를 안 바꾸고 BLAS 스레드 수만 바꿔도 같은 크기로 갈린다.** 원인은 "CPU vs GPU" 가 아니라
**부동소수 누산 순서를 바꾸는 모든 변경**이며 device 는 그중 하나다. 학습이 카오스적이라
첫 경사에서 1e-7 로 갈린 것이 5만 스텝 뒤 0.1 대로 증폭된다 — 성질상 **다른 시드로 학습한 것과 같다**.
반대로 **조건을 고정하면 구/신 코어는 0 차이**이므로 이 폴더의 비트동일 검증은 영향받지 않는다.

**성능 차이는 확인되지 않았다**(위 4모델, 좌표 10 × 30ep paired):
`cpu(t=1)` Δ=−0.0118±0.0352 · `cpu_t8` Δ=−0.0203±0.0369 · `cpu_t16` Δ=+0.0118±0.0256 —
**셋 다 95% CI 안**이라 유의하지 않다. 단 이 모델들은 50k 스텝(PDR_woG 0.36~0.39)으로
**수렴 전 고분산 구간**이고 좌표도 10곳뿐이다. 완전학습 10M(챔피언 0.1484)에서 어떤지는
**측정되지 않았다** — 알려면 device 별 10M 학습 후 정본 paired 프로토콜(대표점250 seed 0–29)로
시드 잡음 바닥(sd 0.00114)과 비교해야 한다(약 14h).

**실무 규칙**: 어떤 학습 산출을 비트 단위로 재현하려면 device **와 BLAS 스레드 수**가 같아야 한다.
⚠️`meta.json` 은 `device` 만 남기고 **스레드 수는 어디에도 기록하지 않는다** — 실제 구멍이다.

## ★ BLAS 스레드 수가 부동소수 결과를 바꾼다 (학습 비교 시 물린 함정)

`drivers/run_fast.py` 는 import 시 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` 을 핀한다.
그런데 원본 학습을 그냥 실행하면 핀이 안 걸려 BLAS 가 전 코어(128 스레드)를 쓴다.
**스레드 수가 다르면 torch 병렬 축소의 누산 순서가 달라져 마지막 비트가 변한다.**

첫 G8 실행이 여기 걸렸다 — 학습 가중치 46개 텐서 전부 maxΔ 1.17e-4, 소요 32.5× 라는
비현실적 배속. sim 코어 가속은 2배 남짓이므로 32배는 원인이 다른 데 있다는 신호였고,
실제로 "코어 차이"가 아니라 "스레드 차이 + loadavg 120 에서의 128스레드 thrashing" 이었다.

**규칙: 가중치·손실 등 신경망 수준을 비교할 때는 양쪽 스레드 수를 명시적으로 같게 고정한다.**
`verify_train_equiv.py` 는 이제 양쪽 모두에 `*_NUM_THREADS=1` 을 넣는다.
(부수 확인: 학습은 같은 시드에서 **완전히 결정적**이다 — 구 코어 2회 46개 텐서 maxΔ=0.
이 재현성이 없으면 가중치 대조로는 코어 동치를 판정할 수 없다.)

## ★ deepcopy 가 numpy 뷰를 끊는다 (실제로 물린 함정)

`copy.deepcopy` 는 numpy **뷰**를 독립 배열로 복제한다 — 뷰↔원배열 연결이 끊긴다.

```python
a = np.zeros((3,3), np.float32); v = a[:,1]      # v.base is a → True
d = copy.deepcopy({"arr": a, "view": v})          # d["view"].base is d["arr"] → False
d["view"] -= 5                                    # d["arr"] 는 안 바뀐다
```

S2-11(차량 잔여시간 열 뷰 캐시)이 여기 걸렸다. **비복제 경로에서는 G1/G2/G4/G5 를 전부
통과**하면서, env 를 복제하는 NCRP 플래너에서만 잔여시간 감산이 차량 상태 배열에 반영되지
않아 조용히 다른 결정을 냈다(`pdr_base` 는 같은데 `pdr_planner` 만 달랐다).
`EventManager.__deepcopy__`/`__getstate__`/`__setstate__` 로 복제·역직렬화 후 뷰를 다시
잡아 고쳤고, `verify/verify_deepcopy.py` 가 이 부류를 상시 감시한다.

교훈: **복제 경로는 별도 게이트가 필요하다.** 일반 경로의 비트동일만으로는 못 잡는다.

## 검증 게이트

```bash
python src/sim_src_upgrade/origin_sync.py --check                    # G0 드리프트
python src/sim_src_upgrade/verify/verify_equivalence.py --audit      # G1 궤적 + G2 지표
python src/sim_src_upgrade/verify/verify_obs_patch.py                # 관측 패치 단위검사
python src/sim_src_upgrade/verify/verify_rl_obs.py --model results/rl/redesign/v10_random4_1000_pointer_s0
python src/sim_src_upgrade/verify/coverage_matrix.py                 # G4 조합 전수
python src/sim_src_upgrade/verify/verify_deepcopy.py                 # G7-a 복제 동치
python src/sim_src_upgrade/verify/verify_train_equiv.py              # G8 학습 가중치 동치
python src/sim_src_upgrade/verify/compare_driver_outputs.py <orig> <fast>
python src/sim_src_upgrade/verify/head_to_head_heur.py --core old|fast --region 서울
```

| 게이트 | 내용 | 소규모 결과(2026-08-11) |
|---|---|---|
| **G0** | 사본이 파생된 `src/sim_src` 해시 불변 | PASS |
| **G1** | 실행 이벤트열 `(time, ev_name, entity_idx)` + 액션열 SHA256 | PASS (165쌍) |
| **G2** | per-episode `(reward, r_woG, pdr, pdr_woG, time, n_steps)` float64 `==` | PASS (165쌍) |
| **G4** | occ/psent × cared 1/0 × 고정47/자연-H × AMB0/UAV0 × capa0.3 × surge300 | PASS (9셀 498쌍) |
| **G5** | 동결 PPO 모델로 obs 402 **비트동일** + 액션열 동일 | PASS (15쌍) |
| **G7-a** | deepcopy·pickle 복제본 전진 동치 + 뷰 무결성 + 원본 무접촉 | PASS (18케이스) |
| **G7-b** | `planner_eval` CSV (`pdr_planner`·`pdr_base`·`n_dec`·`n_switch`) | PASS (완전일치) |
| **G8** | 학습 산출 모델 파라미터 46개 텐서 | PASS (2k CPU / **20k dummy GPU** / **50k subproc GPU** 전부 maxΔ=0) |
| **드라이버** | v16·v10 체크포인트 NPZ 비트동일 | PASS (maxΔ=0.0) |

⚠️ **벤치는 반드시 결과도 대조한다.** 처음 `bench_train.py` 는 시간만 재고 가중치를 안 봤다 —
속도만 보는 벤치는 스스로를 속이는 지름길이다. 지금은 `compare_weights()` 가 붙박이로 돌아
다르면 FAIL 을 찍는다. GPU(cuda) 에서도 20k·50k 학습이 비트동일함을 확인했다.

### 게이트가 진짜 잡는지 (자가진단)

항상 PASS 만 내는 게이트는 무가치하다. 그래서 결함 주입을 지원한다 — **FAIL 이 정상**이다.

```bash
# GB 배차 tier 우선순위 뒤집기 → 궤적이 갈림 (G1·G2 둘 다 FAIL)
python src/sim_src_upgrade/verify/verify_equivalence.py --selftest traj
# 생존확률에 1e-12 더하기 → 궤적은 같고 지표만 갈림 (G1 PASS / G2 FAIL)
python src/sim_src_upgrade/verify/verify_equivalence.py --selftest metric
```
실측으로 두 경우 모두 의도대로 잡혔다(1e-12 차이까지 G2 가 검출).
G7-a 도 `EventManager.__deepcopy__` 를 지운 채 돌리면 즉시 FAIL 한다(뷰 무결성·obs 불일치 양쪽).

### 증분 카운터 상시 대조

`--audit` 을 주면 `_n_cared`/`_n_rescued`/`_in_flight_cnt` 를 **매 호출** 전수 스캔과 대조한다.
카운터 계열 최적화를 건드렸다면 반드시 한 번은 켜고 돌릴 것.

## 구조

```
core/                     sim_src 사본 (상대 import). _origin_hashes.json 이 파생 시점 고정
env_factory_fast.py       make_base_env_fast / make_feature_env_fast / make_feature_env_old
mask_only_wrapper.py      HospitalFeatureWrapper 서브클래스 — mask 유지, obs 생성만 생략
fast_obs_patch.py         rl_src 관측 핫스팟 등가 교체 (opt-in, 원본 파일 무수정)
origin_sync.py            G0 드리프트 게이트 (--write / --check / --diff)
verify/                   trace_hook · verify_equivalence · verify_obs_patch · verify_rl_obs
                          · coverage_matrix · compare_driver_outputs · policies
bench/                    bench_core (인터리브 min-of-N) · profile_core
drivers/run_fast.py       기존 드라이버 무수정 가속 런처
```

**왜 패키지 + 상대 import 인가**: 구 코어(flat `EventManager`)와 신 코어
(`sim_src_upgrade.core.EventManager`)를 한 프로세스에 동시에 올려야 in-process 동치검증이
가능하다. `sys.path` 앞에 끼워넣는 방식으로는 둘 중 하나만 존재한다.

## 사본 드리프트

`src/sim_src` 가 바뀌면 이 사본은 조용히 낡는다. 모든 검증 진입점이 먼저 G0 를 확인하고,
불일치하면 즉시 멈춘다. 원본이 바뀌었다면 **사본에 그 변경을 반영한 뒤**
`origin_sync.py --write` 로 기준을 갱신한다. 그냥 `--write` 만 하면 드리프트를 덮는 것이다.

## 통합은 아직 안 한다

전 게이트를 통과해도 `sim_src` 를 대체하지 않는다. 병합하면 `src/sim_src/*` 해시가 바뀌어
**v10/v16 체크포인트 재개가 전부 무효화**된다. 진행 중인 전수평가가 끝난 뒤 별도로 정한다.
그때까지 `sim_src` 가 정본, 여기는 "결과 동일·속도만 다른 실행경로"로 병존한다.

## 남은 일

* **SB3 `SubprocVecEnv` 학습 경로(spawn/forkserver) 지원** — 런처의 몽키패치가 자식에
  전파되지 않는다. `sitecustomize.py` 를 `PYTHONPATH` 로 주입하는 방식이 유력.
  (그 전까지 학습은 원본 경로 그대로)
* **v17 종료 후 전수 게이트** — `coverage_matrix.py --full` + 좌표·시드 확대 +
  `results/scoreboard/v10/full1000/work/heur/*.npz` 동결값 대조
* 통합(‑ sim_src 대체) 시점 결정
