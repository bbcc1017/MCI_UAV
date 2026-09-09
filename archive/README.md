# archive — 종결 자산 보관소

2026-09-06 레포 정돈 시 신설. **`archive/` 전체가 gitignore 대상이며 이 README 만 추적된다.**
여기 있는 것은 "지웠지만 아까운 것"이 아니라 **판정이 끝나 더는 경로가 참조되지 않는 것**이다.
현행 아크(v17~v19)와 논문 자산은 원래 자리에 그대로 있다.

## 구조

| 경로 | 크기 | 내용 |
|---|---:|---|
| `docs/reports/` | 29개 | v3~v15 종결 아크 보고서. 판정·수치는 `RESEARCH_LOG.md`·`RESEARCH_HISTORY.md`; 과거 지침 전문은 `git show 0b749f07e:AGENTS.md` |
| `docs/presentations/` | 5개 | 일회용 랩미팅·노션 보고자료 (07-27, 07-30, 08-03, 08-21 및 TRE 방향설정) |
| `docs/papers/` | 2개 | VIPER(NeurIPS 2018) 원문, 연속수치지형도 설명서(Unity GIS) |
| `tools/labmeeting_20260903/` | 8개 py | 09-03 랩미팅 그림 생성기. 산출 PNG 는 `archive/docs/presentations/260903랩미팅/` 에 있다 |
| `logs/` | 36개 | 2026-06~08 배치 실행 로그 |
| `results_20260702_헬기장정정이전/` | 37G | 성남 헬기장 정정(2026-07-02) 이전 산출물. **신구 수치 혼용 금지** |
| `scenarios_pre20260702/` | 17G | 같은 사유의 구 시나리오 3종 (아래 표) |

### `scenarios_pre20260702/` 상세 (구 `scenarios/archived/`, 2026-09-01 이동분)

이동 조건 = `scenarios/manifests/**` 어느 파일에서도 경로가 참조되지 않을 것.

| 디렉터리 | 크기 | 사유 |
|---|---:|---|
| `exp_eval_holdout_pre20260702` | 17G | 헬기장 정정 이전 산출물 |
| `exp_home_dep_202606221800` | 4.3M | 일회성 Kakao departure 실험 |
| `exp_inha_dep_202606181500` | 9.2M | 일회성 Kakao departure 실험 |

## ⚠️ 옮기지 않은 시나리오와 그 이유

| 디렉터리 | 참조 매니페스트 | 왜 살아 있어야 하나 |
|---|---:|---|
| `exp_sigungu30` | v19 정본 | 시군구 250×30점 = 현행 학습·평가 풀 (109G) |
| `exp_eval_holdout` | 519 | train1000 random4 학습 정본 |
| `exp_시군구` | 40 | 대표점250 평가 정본 |
| `exp_distill_external` | 2 | 외부250 평가 정본 |
| `exp_v15_blind` | 2 | v15 블라인드 250 — **미개봉** |
| `exp_시도` / `exp_시도natural` | 3 / 1 | 시도17 판정셋 |
| `exp_시군구natural` / `exp_holdoutAnatural` | 3 / 1 | v6 자연-H 매니페스트 |
| `exp_train_pool` | 2 | legacy_center250_plus_random750 (문서화 사유 보존) |

시나리오 경로는 매니페스트에 **절대경로**로 박혀 있어 디렉터리를 옮기면 그 매니페스트가
전부 깨진다. 아카이브 전에 반드시 `grep -rl <dirname> scenarios/manifests/` 로 확인할 것.

## 같은 날 삭제한 것 (아카이브 아님)

**학습 중간 체크포인트 155.7 GB / 17,137개 `*_steps.zip`** 을 지웠다. 런당 5.3G 였고
최종 산출(`final_model.zip` 11M · `vecnormalize.pkl` · `meta.json` · `tb/`)은 전부 남아 있다.

보존한 체크포인트 2종:
- `results/rl/v19/{national, sido_*}` — 현행 최신 학습 모델. 학습곡선을 `v17_ppo_eval --checkpoint`
  로 그 시점 통계에서 평가하는 데 쓰인다 (180개 · 1.84G).
- `results/rl/zoo/{reinforce_s0..2, probe_reinf_*}` — v5 알고리즘 비교 기준선. 이 5개 런만
  `final_model.zip` 이 없어 체크포인트가 유일 산출물이다.

**되돌릴 수 없다.** 잃은 것은 "중간 스텝에서의 학습 재개·재평가" 뿐이고, v3~v15 는 판정이
동결돼 재개 계획이 없다.

---

## 2026-09-08 — 공개 저장소 전환 정돈

`origin` 을 공개 레포로 전환하기 위해 **원격에서 보이는 것을 현행 아크(v17~v21) + 파이프라인 +
논문 기준선**으로 줄였다. 지운 것은 없다 — 파일은 여기 남고 커밋 이력에도 그대로 있다.
**복원 기준 커밋 = `3bd3dea6` (정돈 직전)**, 개별 파일은
`git show 3bd3dea6:<원경로> > <원경로>`.

### `code_20260908/` — 일회성 코드 71개

기준: **다른 추적 파일이 import 하지도, 경로로 호출하지도 않고**(도달성 0),
아크가 종결됐고(v6~v15), 정책·래퍼·알고리즘 클래스가 아닌 **드라이버·집계·플롯·스모크**만 옮겼다.
이동 후 import 폐쇄성(남은 코드가 아카이브 모듈을 참조하지 않음)을 기계 검사로 확인했고,
주요 모듈 30개 실제 import 스모크도 통과했다.

| 경로 | 원 위치 | 개수 | 내용 |
|---|---|---:|---|
| `code_20260908/rl_src/` | `src/rl_src/` | 23 | v3 score 추출 드라이버(`collect_score_dataset`·`fit_score`·`score_eval`)·ExIt 라벨러(`exit_labels`)·VIPER 후속(`viper_{interpret,resim_wog,comms_versions}`)·v10/v13/v14/v15 일회성 suite·구 트레이너(`train_{ppo_reward,sb3_extra}`)·스모크 3종·v17 이전 card 시험(`card_rule_suite`·`card_survival_sensitivity`·`v17_hospital_rule_repair`) |
| `code_20260908/tools/` | `tools/` | 30 | v6~v15 스코어보드·리포트·플롯 일회성(`v{6,10,11,12,13,14,15}_*`)·Shin 모드별 플롯 4종·`verify_all`·`verify_v15_freeze`·`create_rl_pipeline_slide`·`subset_distill_npz` |
| `code_20260908/tools_exp_drivers/` | `tools/exp_drivers/` | 14 | v11·v12·v13·v15 실험 드라이버와 구 psent 기준선 배치 |
| `code_20260908/sce_src_archived/` | `src/sce_src/archived/` | 2 | 이미 archived 로 표시돼 있던 초기 시나리오 생성기 |
| `code_20260908/vis_src_archived/` | `src/vis_src/archived/` | 2 | 초기 헬기장 지도 스크립트 |

**남긴 것**: 정책·래퍼·알고리즘 클래스는 도달성 0 이어도 전부 남겼다
(`score_policy`·`compact_rule_policy`·`guideline_rule_policy`·`portfolio_policy`·`card_rule_policy`·
`gopt_policy`·`leaf_value`·`masked_{dqn,qrdqn,sac_discrete}`·`value_guided_ppo` 등) — 논문 비교표의
기준선이거나 기각 사유의 증거물이다. v1 멀티알고 레거시(`train_{ppo,dqn,reinforce}`·`run_all_parallel`)도
README 문서화·보존 결정(2026-07-11)을 그대로 따랐다.
`score_cma.py`·`exit_distill.py` 는 일단 옮겼다가 **현행 코드가 실제로 import 해서 되돌렸다**
(`planner_eval`·`ncrp_labels` → `score_cma.select_tune_regions`, `ncrp_label_probe` → `exit_distill`).

### 추적 해제 — 지역별 분할 매니페스트 2,002개

`scenarios/manifests/{sigungu30,sigungu250}/` 는 부모 매니페스트에서 스크립트로 재생성되는
파생물인데 추적 파일 수의 95% 를 먹고 있었다. gitignore 로 내리고 재생성 명령을 같은 자리에 적었다.

```bash
python src/sce_src/split_sigungu30.py --wave1 16              # sigungu30/  (1,501)
python src/sce_src/split_sigungu_manifests.py --holdout p3 --wave1 16   # sigungu250/ (501)
```

`sigungu250/_index.json` 은 `results/scoreboard/v17/fieldrules/static_train1000.npz` 를 읽으므로
그 산출물이 있는 박스에서만 완전 재생성된다. 부모 매니페스트·좌표 원장·시도 단위 매니페스트
(`sido/`·`sido_kakao/`·`eval_holdout_sido/`·`sigungu30_sido/`, 70개)는 계속 추적한다.

### `docs/` 추가 이동분

| 이동 | 사유 |
|---|---|
| `260903랩미팅/` → `docs/presentations/` | 09-08 랩미팅으로 대체 |
| `MCI_UAV_paper_{blueprint,audit}_2026-08-09.*` → `docs/reports/` | manuscript 2026-08-11 로 흡수 |
| `1-s2.0-S1366554526003200-main.pdf` → `docs/papers/` | 참고논문(Yan et al. 2026) 원문 — 저작물 |

`docs/` 는 계속 로컬 전용이며(`.gitignore` 를 `docs/` → `docs/*` 로 바꿨다), **예외는
`docs/assets/` 하나**다 — README 에 임베드하는 Unity 데모 GIF 2종(52MB)이 여기 있다.
디렉터리 자체가 ignore 되면 하위 negation 이 먹지 않아 패턴을 바꿔야 했다.
