# archive — 종결 자산 보관소

2026-09-06 레포 정돈 시 신설. **`archive/` 전체가 gitignore 대상이며 이 README 만 추적된다.**
여기 있는 것은 "지웠지만 아까운 것"이 아니라 **판정이 끝나 더는 경로가 참조되지 않는 것**이다.
현행 아크(v17~v19)와 논문 자산은 원래 자리에 그대로 있다.

## 구조

| 경로 | 크기 | 내용 |
|---|---:|---|
| `docs/reports/` | 29개 | v3~v15 종결 아크 보고서. 판정·수치는 `RESEARCH_LOG.md` 와 `CLAUDE.md` 에 요약돼 있다 |
| `docs/presentations/` | 5개 | 일회용 랩미팅·노션 보고자료 (07-27, 07-30, 08-03, 08-21 및 TRE 방향설정) |
| `docs/papers/` | 2개 | VIPER(NeurIPS 2018) 원문, 연속수치지형도 설명서(Unity GIS) |
| `tools/labmeeting_20260903/` | 8개 py | 09-03 랩미팅 그림 생성기. 산출 PNG 는 `docs/260903랩미팅/` 에 있다 |
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
