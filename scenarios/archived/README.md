# archived — 참조 매니페스트 0개인 구 시나리오

2026-09-01 이동. 이동 조건 = `scenarios/manifests/**` 어느 파일에서도 경로가 참조되지 않을 것.

| 디렉터리 | 크기 | 사유 |
|---|---|---|
| `exp_eval_holdout_pre20260702` | 17G | 성남 헬기장 정정(2026-07-02) 이전 산출물. 신구 수치 혼용 금지 대상 |
| `exp_home_dep_202606221800` | 4.3M | 일회성 Kakao departure 실험 |
| `exp_inha_dep_202606181500` | 9.2M | 일회성 Kakao departure 실험 |

## ⚠️ 옮기지 않은 것과 그 이유

| 디렉터리 | 참조 매니페스트 | 왜 살아 있어야 하나 |
|---|---:|---|
| `exp_eval_holdout` | **519** | train1000 random4 학습 정본. **v18 학습 잡 19개가 실행 중 사용** |
| `exp_시군구` | 40 | 대표점250 평가 정본 |
| `exp_distill_external` | 2 | 외부250 평가 정본 |
| `exp_v15_blind` | 2 | v15 블라인드 250 — **미개봉** |
| `exp_시도` / `exp_시도natural` | 3 / 1 | 시도17 판정셋 |
| `exp_시군구natural` / `exp_holdoutAnatural` | 3 / 1 | v6 자연-H 매니페스트 |
| `exp_train_pool` | 2 | legacy_center250_plus_random750 (문서화 사유 보존) |

시나리오 경로는 매니페스트에 **절대경로**로 박혀 있어 디렉터리를 옮기면 그 매니페스트가
전부 깨진다. 아카이브 전에 반드시 `grep -rl <dirname> scenarios/manifests/` 로 확인할 것.
