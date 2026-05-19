# MCI_UAV

대규모 사상자 사고(MCI) 시뮬레이션에서 **AMB(구급차) + UAV** 혼합 자원으로 환자 triage·이송 의사결정을 학습하는 연구용 코드.

MCI_ADV 의 시뮬·시나리오 베이스에 강화학습(PPO/DQN/REINFORCE) 비교 + 휴리스틱 baseline + 다지점 일반화 평가 + AMB=휴리스틱·UAV=RL 하이브리드 평가를 결합.

> 이전 단계에서 잠시 UAV 단독 운용으로 좁혔던 자산은 `*/archived/` 로 보존 이동했음.

---

## 폴더 구조

```
MCI_UAV/
├── scenarios/                  사고지점별 시나리오 (자동 생성, gitignore)
│   ├── 엑셀 결합 데이터.xlsx    원본 병원 풀
│   ├── 안전센터와 소방서.csv    AMB 기지 데이터 (MCI_ADV 호환)
│   ├── exp_*_osrm/             OSRM 모드 시나리오 출력
│   ├── exp_*_dep_<ts>/         Kakao 모드 (출발시각별) 시나리오 출력
│   └── archived/               UAV-only 헬기장 풀 등 옛 자산
│
├── src/
│   ├── sce_src/
│   │   ├── make_csv_yaml_dynamic.py   AMB+UAV 동적 시나리오 생성 (Kakao/OSRM)
│   │   └── archived/make_uav_scenario.py
│   ├── sim_src/                시뮬레이터 코어 (event-driven, 무수정)
│   │   ├── main.py                 휴리스틱 룰 시뮬
│   │   ├── ScenarioManager.py      yaml → entity 셋업
│   │   ├── EntityManager.py
│   │   ├── EventManager.py
│   │   ├── RuleManager.py          휴리스틱 룰 (START/ReSTART × RedOnly/YellowNearest × mode)
│   │   ├── MCIEnvironment_gymnasium.py  gym env (AMB+UAV 자동 처리)
│   │   └── event_info.json
│   ├── rl_src/                 강화학습
│   │   ├── env_factory.py
│   │   ├── env_wrapper.py          dict→flat, MultiDiscrete→Discrete, action_mask, decode/encode
│   │   ├── train_dqn.py
│   │   ├── train_ppo.py            MaskablePPO
│   │   ├── train_reinforce.py
│   │   ├── reinforce_agent.py
│   │   ├── evaluate.py             1안 — RL/heur 비교
│   │   ├── hybrid_eval.py          2안 — AMB=heur, UAV=RL 하이브리드
│   │   ├── log_heuristic_baseline.py
│   │   ├── run_all_parallel.py
│   │   └── cross_location_eval.py  17개 광역 좌표 일괄 평가
│   └── vis_src/
│       └── archived/map_helipad_center.py  (헬기장 시각화)
│
├── external/ml-agents/         Unity ML-Agents (submodule)
├── experiment_logs/, results/  학습/평가 산출 (gitignore)
└── requirements.txt
```

---

## 환경

- Python 3.10 (conda env `UAV`)
- torch 2.8.0+cu128 (RTX 50 호환)
- 자세한 의존성: `requirements.txt`

```
pip install -r requirements.txt
```

OSRM 도로 거리 모드(기본)는 외부 서버 호출이라 인터넷 연결 필요. 대량 평가는 로컬 OSRM 컨테이너 권장:
```
docker run -t -i -p 5000:5000 osrm/osrm-backend osrm-routed --algorithm mld /data/<korea>.osrm
python ... --osrm_url http://localhost:5000
```
Kakao Mobility API 모드는 `--is_use_time True --kakao_api_key <KEY>` 로 활성.

---

## 사용 흐름

### 1) 시나리오 생성 (AMB+UAV 혼합)

`scenarios/엑셀 결합 데이터.xlsx` 의 일반 병원 풀 + `안전센터와 소방서.csv` 의 AMB 기지에서 사고 좌표 기준 거리·용량 셋업. 기본은 OSRM (`is_use_time=False`), Kakao 쓰려면 키 지정.

```
python src/sce_src/make_csv_yaml_dynamic.py --base_path . --experiment_id mix_seoul --latitude 37.5666 --longitude 126.9784
```

기본값: `--incident_size 100 --amb_count 30 --uav_count 25 --total_samples 1000 --is_use_time False`. 출력 폴더는 OSRM 모드면 `scenarios/exp_mix_seoul_osrm/(37.5666,126.9784)/`.

### 2) 휴리스틱 시뮬

```
python src/sim_src/main.py --config_path "scenarios/exp_mix_seoul_osrm/(37.5666,126.9784)/config_(37.5666,126.9784).yaml"
```

`results_*.txt` / `results_*_stat.txt` 저장. AMB·UAV 32 룰 조합(START/ReSTART × RedOnly/YellowNearest × Red mode 4 × Yellow mode 4) 각각 시뮬.

### 3) RL 동시 학습 (DQN + PPO + REINFORCE)

```
python src/rl_src/run_all_parallel.py --config_path <yaml> --total_timesteps 200000 --seed 0 --suffix mix_seoul
```

각 학습 결과: `results/rl/{dqn,ppo,reinforce}_mix_seoul/final_model.{zip,pt}`. AMB+UAV 둘 다 활성이면 wrapper 가 자동으로 `Discrete(3*(H+1)*2)` action space 로 학습.

### 4) TensorBoard

```
tensorboard --logdir results/rl --port 6006
```

휴리스틱 baseline 평탄선:
```
python src/rl_src/log_heuristic_baseline.py --config_path <yaml> --n_episodes 1000 --tb_dir results/rl --max_step 200000
```

### 5-A) 1안 평가 — RL vs 휴리스틱 (단일 좌표)

```
python src/rl_src/evaluate.py --config_path <yaml> --ppo_path results/rl/ppo_mix_seoul/final_model.zip --include_heuristic --n_episodes 100
```

### 5-B) 1안 평가 — 17개 광역 일괄

```
python src/rl_src/cross_location_eval.py --ppo_path <...> --dqn_path <...> --reinforce_path <...> --n_episodes 1000 --exp_prefix crosseval_mix
```

### 5-C) 2안 평가 — 하이브리드 (AMB=휴리스틱, UAV=RL)

```
python src/rl_src/hybrid_eval.py --config_path <yaml> --ppo_path results/rl/ppo_mix_seoul/final_model.zip --rule_priority START --rule_hos_select RedOnly --rule_red_mode Both_AMBFirst --rule_yellow_mode Both_AMBFirst --mode_split strict --n_episodes 100
```

`--mode_split strict` 는 RL 이 AMB 모드를 고른 시점에 휴리스틱이 class/dest/mode 전부 결정. `loose` 는 RL 의 class/dest 유지, mode 만 휴리스틱 매핑.

---

## 주요 설계 결정

- **시나리오 생성**: MCI_ADV 의 `make_csv_yaml_dynamic.py` (1296줄) 이식. argparse 기본값만 MCI_UAV 파라미터(환자 100 / UAV 25 / total_samples 1000 / is_use_time=False) 로 변경.
- **AMB+UAV 혼합**: sim_src 의 `ScenarioManager`/`RuleManager`/`EventManager` 는 amb_num>0 자동 활성. `MCIEnvironment_gymnasium.action_space` = `MultiDiscrete([3, H+1, 2])`.
- **wrapper 자동 차원 조정**: `FlattenAndDiscreteWrapper` 는 amb_num/uav_num 검사 후 mode 차원 자동 유지·제거.
- **Reward**: 환자 admit 시점의 생존확률 (Red/Yellow 시간 감쇠, Green=1, Black=0).
- **하이브리드 정책 (2안)**: `hybrid_eval.py` 가 wrapper 의 `decode_action`/`encode_action` 으로 RL 출력을 가공해 AMB 결정만 룰로 덮어씀.

자세한 알고리즘 / 코드 세부는 각 모듈 docstring 참고.
