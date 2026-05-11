# MCI_UAV

대규모 사상자 사고(MCI) 시뮬레이션에서 **UAV 단독 자원**으로 환자 triage·이송 의사결정을 학습하는 연구용 코드.

이전 MCI_ADV(AMB+UAV) 베이스에서 UAV 전용으로 축소·재구성하고 PPO/DQN/REINFORCE 비교 + 휴리스틱 baseline + 다지점 일반화 평가까지 포함.

---

## 폴더 구조

```
MCI_UAV/
├── scenarios/                  사고지점별 시나리오 (자동 생성, gitignore)
│   ├── 엑셀 결합 데이터.xlsx     원본 병원 데이터 (헬기장 여부 컬럼 포함)
│   ├── 안전센터와 소방서.csv    (현재 미사용)
│   └── exp_*_uav/              개별 사고지점 시나리오 출력
│
├── src/
│   ├── sce_src/                시나리오 생성
│   │   └── make_uav_scenario.py
│   ├── sim_src/                시뮬레이터 코어 (event-driven)
│   │   ├── main.py                 휴리스틱 룰 1000ep 시뮬
│   │   ├── ScenarioManager.py      yaml → entity 셋업
│   │   ├── EntityManager.py        entity 상태 보관
│   │   ├── EventManager.py         이벤트 큐 + 핸들러
│   │   ├── RuleManager.py          휴리스틱 룰 (START/ReSTART × RedOnly/YellowNearest × mode)
│   │   ├── MCIEnvironment_gymnasium.py  gym env wrapper
│   │   └── event_info.json
│   ├── rl_src/                 강화학습
│   │   ├── env_factory.py          config → MCIEnvironment_gym
│   │   ├── env_wrapper.py          dict obs → flat, MultiDiscrete → Discrete, action mask
│   │   ├── train_dqn.py
│   │   ├── train_ppo.py            MaskablePPO + ActionMasker
│   │   ├── train_reinforce.py      Monte Carlo policy gradient
│   │   ├── reinforce_agent.py      REINFORCE 정책망/I-O
│   │   ├── evaluate.py             PPO/DQN/REINFORCE/Random/heuristic 표 비교
│   │   ├── log_heuristic_baseline.py   휴리스틱 best 를 TB scalar로 기록
│   │   ├── run_all_parallel.py     3개 알고리즘 동시 학습 launcher
│   │   └── cross_location_eval.py  17개 광역 좌표 일괄 평가 + plot
│   └── vis_src/                지도 시각화
│       └── map_helipad_center.py   folium 기반 HTML (헬기장 병원/중점/광역청/V-world)
│
├── external/
│   └── ml-agents/              submodule (Unity ML-Agents)
│
├── experiment_logs/            시뮬/시나리오 생성 로그 (gitignore)
├── results/                    학습/평가 출력 (gitignore)
├── helipad_location.py         V-world API에서 헬기장 좌표 수집
├── helipad_location.csv        수집된 헬기장 데이터
├── requirements.txt
└── README.md
```

---

## 환경

- Python 3.10 (conda env `UAV`)
- torch 2.8.0+cu128 (RTX 50 호환)
- 자세한 의존성: `requirements.txt`

설치:
```
pip install -r requirements.txt
```

---

## 사용 흐름

### 1) 시나리오 생성

원본 엑셀(`scenarios/엑셀 결합 데이터.xlsx`)에서 헬기장 25개 풀 추출 → 사고 좌표에서의 거리·용량 셋업 + UAV·환자·yaml 산출.

```
python src/sce_src/make_uav_scenario.py --base_path . --experiment_id helipad_center --latitude 36.245107096 --longitude 127.462534992 --incident_size 100 --uav_count 25 --rule_test
```

### 2) 휴리스틱 시뮬

생성된 yaml로 1000 episode × 4룰 시뮬. `results_*.txt` / `results_*_stat.txt` 저장.

```
python src/sim_src/main.py --config_path scenarios/exp_helipad_center_uav/(36.245107096,127.462534992)/config_(36.245107096,127.462534992).yaml
```

### 3) RL 동시 학습 (DQN + PPO + REINFORCE)

```
python src/rl_src/run_all_parallel.py --config_path <yaml 경로> --total_timesteps 200000 --seed 0 --suffix helipad_center
```

각 학습 결과: `results/rl/{dqn,ppo,reinforce}_helipad_center/final_model.{zip,pt}`

### 4) TensorBoard 비교

```
tensorboard --logdir results/rl --port 6006
```

휴리스틱 baseline을 같은 차트에 평탄선으로 띄우려면 학습 전/후에 한 번:
```
python src/rl_src/log_heuristic_baseline.py --config_path <yaml> --n_episodes 1000 --tb_dir results/rl --max_step 200000
```

### 5) 다지점 일반화 평가

17개 광역 좌표(서울/부산/...)에서 학습 모델 + 휴리스틱 일괄 평가, csv + png 출력.

```
python src/rl_src/cross_location_eval.py --ppo_path <...> --dqn_path <...> --reinforce_path <...> --incident_size 100 --uav_count 25 --n_episodes 1000 --seed 0
```

### 6) 지도 시각화 (HTML)

헬기장 병원 25 / helipad_center / 광역시도청 17 / 전국 헬기장 (V-world) 을 layer 별로 toggle 가능한 인터랙티브 지도.

```
python src/vis_src/map_helipad_center.py
```

출력: `results/map_helipad_center.html` (CartoDB Positron 타일 + LayerControl)

---

## 주요 설계 결정

- **AMB 비활성화**: `amb_info_road.csv` 0바이트 + `setup_ambulance` early-return.
- **road CSV 미사용**: UAV 직선거리 기반이라 `*_road.csv` 4종 모두 0바이트, 시뮬레이터는 `_euc` 로 자동 폴백.
- **Action space**: `MultiDiscrete([3, H+1, 2])`. AMB 0대 시 wrapper에서 mode 차원 제거 → `Discrete(3*(H+1))`.
- **Reward**: 환자 admit 시점의 생존확률 (Red/Yellow 시간 감쇠, Green=1, Black=0).
- **Hospital pool**: 엑셀 `헬기장 여부 == 1` 인 25개만 사용. `uav_count` 만큼 가까운 순으로 UAV 배치.

자세한 알고리즘 / 코드 세부는 각 모듈 docstring 참고.
