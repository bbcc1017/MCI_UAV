# 실행·검증·복구

## 환경과 추적 범위

Linux 기준 cwd=`/home/ryu/MCI_UAV`, Python=`/home/ryu/anaconda3/envs/UAV/bin/python`.
먼저 실제 cwd·git status·파일 존재를 확인한다. 패키지 버전은 실행 환경에서 확인하며 설치를 추측하지 않는다.
공유 학습 노드의 현재 여유는 `uptime`, `free -h`, `nvidia-smi`, 프로세스 상태로 판단한다.
과거 worker 수·GPU 점유·처리량은 현재 예약 용량이 아니다.

**GPU 는 device 0 만 쓴다(2026-09-14).** 노드에 2장이 있다 — `0` RTX A6000 · `1` RTX 6000 Ada
Generation(각 49 GiB). **`1` 은 타 사용자 몫**이므로 GPU 를 쓰는 모든 실행에
`CUDA_VISIBLE_DEVICES=0` 을 붙인다. 그러면 프로세스 안의 `cuda:0` 이 물리 0번이 되므로
코드에 물리 인덱스를 박지 않는다(`cuda:1`·`set_device(1)`·`device_count()` 기반 분배 금지).
`nvidia-smi` 의 device 1 점유율은 **우리 가용 용량 계산에 넣지 않는다.**

추적 대상은 Python 코드·공용 도구·루트 문서·상위 manifests/프로토콜 등이다.
생성 시나리오 `scenarios/exp_*`, `results/`, 대부분 `docs/`, `archive/`, `.agents/`, `.codex/`는 로컬이다.
예외는 `.gitignore`와 `git ls-files`로 확인한다(`docs/assets/`, `archive/README.md` 등).
이 지침의 `agent_docs/`는 공개 가능한 운영 메모만 담는 추적 경로다.
논문 초고·특허·원문 PDF·개인 로그를 여기에 옮기지 않는다.

`external/ml-agents`는 upstream submodule(`ignore=dirty`)이며 `UAV_test/`·`CAR_test/`는 그 안의
미추적 Windows 자산이다. parent repo에서 Unity 변경을 stage하거나 submodule 포인터를 바꾸지 않는다.
Windows/Unity/GIS 작업 전 `CLAUDE.unity.md`와 존재하는 `.claude.local.md`를 직접 읽는다.
Linux에 해당 파일·산출물이 없다고 삭제되었다고 단정하지 않는다.

Windows는 env Python 직접 호출+`PYTHONIOENCODING=utf-8`, raster/GDAL은 `qgis_batch` 환경을 확인한다.
Windows Python 출력의 CRLF를 Bash 반복문에서 처리하고, Python↔Git-Bash 간 경로는
프로젝트 상대경로를 쓴다. 생성 시나리오/trace의 `Y:/scenarios/...`와 Linux 경로를 혼동하지 않는다.

## 최소 검증 선택

| 변경 | 적절한 검증 |
|---|---|
| 문서/지침 | diff·링크/파일 존재·명령 인자·현재 근거 확인; 학습 불필요 |
| 규칙/평가 배선 | 개발 좌표 소수·고정 seed 폐루프, 마스크 위반/유한값/행 수·두 팔 키 검사 |
| obs/codec/정규화 | 실제 reset obs로 H/F/dim·면제열·decode·padding·해당 모델 로드 확인 |
| sim/가속 | codebase.md의 해당 등가성 게이트 + 작은 원본/가속 paired rollout |
| import/파일 정리 | Python·셸·subprocess·역직렬화 참조 조사 + 주요 실제 import |
| 연구 결론 변경 | research.md의 split·완전성·직접 paired·선택 이력 검증 |

통합 pytest suite를 전제로 하지 않는다. 대신 `src/sim_src_upgrade/verify/`,
`src/rl_src/pad_smoke.py`, `tools/smoke_shin_heuristics.py` 등의 전문 검증기가 있다.
문법 검사만으로 동작 검증을 대신하지 않는다. 반대로 문서 수정 때문에 대규모 학습/전수평가를 시작하지 않는다.
all-zero obs는 valid도0이 되어 전 병원 padding/attention NaN을 만들 수 있으므로 실제 유효 병원을 포함한다.

## 짧은 규칙 평가 예제

저장소 루트에서 실행한다. 기본 대표점250을 피하도록 manifest를 명시한다.
budget 좌표 1개·2seed·3정책의 **배선 확인용**이며 성능 판정이 아니다.
새 임시 출력 디렉터리를 쓰므로 기존 scoreboard 재개 상태와 섞이지 않는다.

```bash
MCI_PY=/home/ryu/anaconda3/envs/UAV/bin/python
MCI_MAN=scenarios/manifests/sigungu30_budget750_manifest.json
MCI_KEY=$("$MCI_PY" -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1], encoding="utf-8")))))' "$MCI_MAN")
MCI_SMOKE_DIR=$(mktemp -d /tmp/mci-rule-smoke.XXXXXX)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$MCI_PY" src/rl_src/v17_rule_eval.py \
  --manifest "$MCI_MAN" --regions "$MCI_KEY" --n_eps 2 --seed0 0 --workers 1 \
  --policies 'CARD_Q18=cardt:18,6.6,0,hingerate;CARD_P18=cardt:18,6.6,0,hingerate_psent;START_LB3=cap3:START, YellowNearest, Red OnlyUAV, Yellow Both_AMBFirst' \
  --out "$MCI_SMOKE_DIR/rules.csv"
```

실행 전 의도치 않은 상속 물리 노브가 없는지 확인한다. `MCI_CAP_GATE` 등 일부는 evaluator가
설정하지만 자원/속도/인계 노브까지 초기화하지는 않는다. `.meta.json`의 `scenario_knobs`도 확인한다.

기존 v21 수치 재집계(읽기 전용):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ryu/anaconda3/envs/UAV/bin/python tools/v21_infoladder_report.py judge
```

필요할 때 `audit`로 seed0..9와 seed0..29의 정합을 확인한다.
`ladder`는 튜닝 최적 JSON을 쓰므로 단순 조회와 구별한다.

## 학습 설정을 찾는 순서

먼저 목표 모델의 `meta.json`·실제 zip·vecnorm·드라이버를 확인한다.
v19 전국 모델은 `results/rl/v19/national/`: train6000, `MCI_OBS_VARIANT=field`,
H_pad47, occ, pointer, attention0, pdrwog+reward norm, fresh10M/seed0,
embed64/ctx128/head128, lr3e-4 anneal, KL.03, epochs5, n_steps512, batch512, 8subproc.
이것은 재현 레시피이지 모든 신규 실험의 고정 예산은 아니다.
현재 물리축 재학습 예는 `tools/exp_drivers/run_v20_retrain.sh`다.

짧은 배선 확인은 train/budget에서 한다. test750으로 반복 디버깅·체크포인트 선택하지 않는다.
학습 데이터가 달라진 실험에 기존 대표점 학습 모델을 초기화하고 미학습 일반화라고 주장하지 않는다.
checkpoint 저장 시 `--save_vecnormalize`를 포함한다.

## 장시간 작업

사용자가 요청한 실험은 필요한 작은 검증을 마친 뒤 실행·완료 확인까지 이어간다.
분석/문서 요청만으로 새 대규모 학습을 시작하지 않는다. 반복 승인 대신 현재 요청의 범위를 따른다.

1. 기존 작업·현재 자원·예상 출력 경로를 확인한다. 중첩 Pool의 worker 합과 메모리를 계산하고,
   numpy/torch import 전 BLAS/OpenMP 스레드를 명시적으로1에 고정한다.
2. 실험마다 출력 디렉터리를 분리하고 command, cwd, git SHA/dirty diff, manifest hash,
   모델/vecnorm, 물리 노브, seed 범위, 예상 행 수를 메타에 기록한다.
3. foreground는 현재 Codex 도구가 돌려준 session/cell ID로 이어서 확인한다.
   장기 detach가 필요하면 PID/PGID·로그·종료코드·진행 파일을 기록한다.
   구 Claude 전용 `run_in_background`·`SendMessage`·워처 자동통지를 가정하지 않는다.
4. stdout 억제는 sim 이벤트 출력에 국한해 사용한다. `_suppress_stdout()` 안의 내 print도 사라진다.
   진행 출력은 블록 밖에서 flush, 장기 로그는 파일로 보낸다.
   실행 중 Python을 `| head`로 잘라 SIGPIPE를 만들지 않는다.
5. 정지는 내 run의 PID/PGID와 전체 명령/출력 경로를 검증한 뒤 수행한다.
   스크립트 이름만으로 `pkill -f`하지 않는다. 재기동 전 driver와 자식 모두 종료됐는지 확인한다.
   잠든 옛 driver가 깨어 같은 CSV를 쓰는 중복 실행을 피한다.
6. 재개 전 기존 CSV의 정책 스펙·모델·seed 범위·knobs·hash를 비교한다.
   실행 중 코드를 수정해도 이미 import된 worker에는 소급 적용되지 않는다.
7. 완료는 exit code+실패 수+기대 행/키+메타로 확인한다. 일부 driver는 실패 후에도 DONE을 쓰고,
   일부 evaluator는 실패 지역이 있어도 meta를 남긴다. 필요 데이터가 빠지면 판정을 보류하고 복구한다.

짧은 대기마다 같은 상태를 반복 보고하지 않는다. 의미 있는 진행/장애/완료를 전달한다.
다음 작업으로 이어질 때 `results/<run>/` 안에 목적·실행명령·PID·완료/남은 범위·다음 행동을
짧게 남긴다. 일시적 PID나 현재 GPU 점유를 루트 AGENTS에 누적하지 않는다.

## Git과 복구

여러 머신이 같은 origin을 사용한다. 기존 사용자 수정은 보존하며 파일을 명시해 stage한다.
`git add -A`와 force-push는 쓰지 않는다. push가 요청되고 fetch-first로 거절되면 fetch 후
현재 브랜치/원격/작업트리를 확인하고 비파괴 통합·검증 후 재시도한다.
모든 작업에 무조건 `rebase origin/main`을 실행하는 규칙은 두지 않는다.
새 브랜치가 필요하면 `codex/` 접두사, 커밋/코드 주석은 한국어, Co-Authored-By 표기는 English.

- 2026-09-08 종결 코드 이동 원장: [archive/README.md](../archive/README.md).
  원본은 `git show 3bd3dea6:<당시 경로>`, 로컬 사본은 `archive/code_20260908/`.
- 2026-09-06 Unity/GIS 도구의 복원 기준: `9f4d4fc`, 상세 `CLAUDE.unity.md`.
- 이전 Windows HEAD에서 tracked였던 파일은 추적 해제 커밋 pull 때 제거될 수 있다.
  필요한 로컬 자산 백업 후 pull, 필요 경로만 옛 커밋에서 **worktree에만** 복원한다.
- `sigungu30/` 분할 manifests는 `split_sigungu30.py --wave1 16`,
  `sigungu250/`는 `split_sigungu_manifests.py --holdout p3 --wave1 16`으로 재생성한다.
  후자의 `_index.json`은 로컬 `results/scoreboard/v17/fieldrules/static_train1000.npz`도 필요하다.
- 구 AGENTS 전문은 `git show 0b749f07e:AGENTS.md`로 조회한다. 과거 명령을 복원할 때만 읽는다.
