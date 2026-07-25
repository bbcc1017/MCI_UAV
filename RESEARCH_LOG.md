# RESEARCH_LOG.md — MCI RL 재설계 v3~v5 상세 기록

`CLAUDE.md`/`AGENTS.md` 에서 분리한 **연구 이력·측정 수치·산출물 경로**. 매 세션 필요한 결론(챔피언 모델·평가 불변식·기각 목록)은 CLAUDE.md 의 **"재설계 v3~v5 — 현행 정답과 기각 목록"** 에 남기고, 재현·검증에 필요한 전체는 여기 둔다.

**읽어야 할 때**: ① 새 실험을 설계하기 전(→ "이미 해봤고 왜 기각했나" 확인) ② v3~v5 수치·CSV·보고서 경로를 인용할 때 ③ 스코어 추출·동물원·플래너 도구를 실제로 돌릴 때(함정이 여기 있다).

세 단계 공통 실험 설계: **학습=시군구250 중심점 / 판정=시도17 대표점 / 일반화=홀드아웃 250 새 좌표**(셋 다 좌표 무중복), seed **11000**, 시뮬 코어 무수정.

⚠️본문이 가리키는 `docs/*.md`·`results/rl/**` 산출물은 **Linux 학습박스(`aigpu0617:/home/ryu/MCI_UAV`) 전용** — Windows 체크아웃엔 없다.

## 재설계 v3 자산 (2026-07) — 성능 극대화 + 추출 2.0

전국 단일 정책 성능 최대화(S0~S5) + 해석가능 규칙 추출(B0~B7). 학습=시군구250 중심점, 판정=시도17 대표점, 일반화=홀드아웃 250 새 좌표(셋 다 좌표 무중복).

- **train_ppo_feature.py 신규 플래그**: `--gamma/--gae_lambda/--embed_dim/--ctx_dim/--head_hidden`(폭 스윕), `--region_weights`(지역 가중샘플). ⚠️`--resume_from`은 저장된 lr 스케줄을 복원하고 SB3가 진행률을 `num_timesteps/(num_timesteps+추가)`로 재계산 → **+3M 정도면 ~7e-5부터 재개**(파인튠 유효). ⚠️단 **소량 resume(+수십만~300k)은 progress≈0.98→lr 5e-6→1.8e-8=정책 동결**(approx_kl 1e-9·greedy 무변) — 소량 파인튠은 상수 `finetune_lr` 리셋 필수(`train_vgppo` 선례, v7 lr≈0 함정으로 규명). `FeatureMultiRegionEnv`는 매니페스트 지역수>500일 때만 워커별 shard(그 이하는 워커마다 전 지역 빌드 → 1000점 매니페스트는 shard 필수, RSS).
- **성능 트랙**: `rollout_oracle.py`(롤아웃 룩어헤드=도달상한 headroom, deepcopy 결정론 검증), `gen_train_pool_osrm.py`(시군구 폴리곤당 3점, holdout 1km 이격), `region_weights.py`(regret/headroom 가중 — ⚠️softmax가 이상치 1곳에 붕괴 → winsorize(p90)+β상한 유계화 필수), `exit_labels/exit_distill.py`(ExIt: 오라클 라벨→BC→PPO 파인튠, ⚠️DAgger switch율 높음=분포효과지 버그 아님). **결론**: 반응형 정책은 오라클 상한의 ~56% 도달, 나머지는 온라인 룩어헤드 전용(ExIt 증류 실패=구조적 천장).
- **추출 2.0**: `score_features.py`(φ12 지역불변, dict obs·en_properties·get_static_eta 원천 — 평탄 obs 슬라이스 금지), `score_policy.py`(dest=argmax w·φ 선형 스코어, mode timesave/joint — ⚠️정원제 끄려면 `T_hard=9999`, `score_cma.py` CLI 기본이 `--T_hard 4.0`), `fit_score.py`(조건부로짓 MLE), `score_cma.py`(자작 CEM), `score_eval.py`(paired ablation, `--models 이름=디렉터리=variant`, `--dump_pe`). 최종 규칙 회수율 65.5%(prog11→T메타30→스코어65.5→RL100).
- **paired 평가 관례(불변식)**: seed **11000**, **시도17=판정 전용(튜닝 절대 금지)**, 튜닝풀=시군구 **40점 CRN**, holdout=`eval_holdout_A` `_p0` 250점(match sigcd). 회수율=(PDR_LB−PDR)/(PDR_LB−PDR_RL). **승/무/패=`_paired`(지역별 에피 배열 95%CI 유의성: 평균개선>CI=승, <−CI=패, 그사이=무), 지역평균 임계값 아님** — CSV서 W/T/L 재현 시 이 기준 필수(region-mean 1e-9로 세면 tie 수 불일치, v5 NCRP holdout이 대표 예). 챔피언(구, v4에서 `v4_plr2`로 대체) `results/rl/redesign/s3_plr_s{0,1,2}`(wide+농촌재가중), 보고서 `docs/{성능극대화_사다리,알고리즘_검토,스코어추출}_*.md`.

## 재설계 v4 (2026-07-11) — 상한 재도전 결과

- **챔피언 = `results/rl/redesign/v4_plr2_s{0,1,2}`**(s3_plr+반복재가중 2R resume +3M, obs essential+load 355 그대로) — 전 시드 시도17 LB-T4 17/0, holdout250 LB패배 s0 15/시드평균 20, **오라클 갱신: 도달률 57.1%(0.0684)**. s3_plr는 구 챔피언.
- 신규 obs `essential+load+ctx`(F10·글로벌32, dim 502)와 `--n_attn_blocks≥2`는 **기각**(시도17 전부 열세 — 반응형 관측·구조 포화의 이중 확증). 배선은 보존, 기본값=구모델 호환. ⚠️훈련곡선 중간 우세는 채택 근거 불가(v4_full 곡선 1위 → 판정 꼴찌).
- **반복재가중 재현**: 학습풀 regret 스크린(`paired_eval_ladder --manifest sigungu_osrm --n_eps 200`) → **w∝1+3·min(regret,p90⁺)/p90⁺**(p90⁺=양수 regret만의 p90 — ⚠️`region_weights.py`의 softmax·floor식과 다른 별도 공식, v3 bounded CSV 역산 maxΔ=0 확정) → `--resume_from`+`--region_weights`. 수확 체감(holdout 패배 28→21→15 s0).
- 보고서: `docs/{v4_상한재도전_보고서,연구여정_v2-v4_종합}_2026-07-11.md`, 총괄 입구 `docs/README.md`.

## 재설계 v5 (2026-07-14) — 다알고리즘 공정비교 + 재난특화 플래너(NCRP)

두 목표: ① masked 알고리즘 동물원(논문 baseline 비교), ② 반응형 천장(v3·v4 이중증명 도달률 57%)을 **결정시점 계획**으로 돌파. 시뮬 코어 무수정, 시도17=판정 전용 불변식 유지.

- **공정비교 하네스(신규, 커밋 57bbdab·61f488e)**: 전 알고리즘이 **학습·수집·타깃·평가 전부 하드 마스킹**(v1 DQN의 no-op 페널티 의존과 결정적 차이). `mask_info_wrapper.py`(step info에 next-state `action_mask`·`dt` 주입 — flat obs355로는 helipad 등 마스크 재계산 불가) → `masked_replay_buffer.py`(masks/next_masks/dts 저장, `_pending_cur_masks` 계약: `_sample_action`이 행동선택 직후 세팅→같은 스텝 add() 소비) → `masked_dqn.py`(DoubleDQN+γ^Δt 훅)·`masked_qrdqn.py`(n_quantiles 50)·`masked_sac_discrete.py`(twin-Q·masked Categorical·**per-state 목표엔트로피 c·log n_valid**)·`reinforce_vec.py`(vec vanilla PG+배치표준화 베이스라인, 클리핑·재사용 없음=PPO와 차별 보존). 통합 CLI `train_zoo.py`(**obs/게이트 env var 내부 설정** — train_ppo_feature와 달리), 저장물에 `meta.json`{algo,hypers,git_sha}. 판정 `paired_eval_ladder.py --models 이름=디렉터리=variant[=algo]`(4토큰, 3토큰=ppo 후방호환 바이트동일 회귀 확인), `evaluate.masked_model_policy`(predict_masked 공통계약), `learning_curve_zoo.py`(⚠️PPO tb는 norm_reward 단위라 zoo raw와 절대비교 금지).
- **비교 결과(시도17 1000ep PDR_woG, 낮을수록 좋음)**: 64룰 0.234 / SAC-D 0.274±0.048(엔트로피 민감, 붕괴) / QRDQN 0.245(분포형 역효과) / DQN+SMDP 0.243 / REINFORCE 0.215±0.021(2M에 이미 64룰 동률이나 시드분산 큼) / **DQN 0.190±0.016(동물원 최선)** / LB-T4 0.120 / PPO(v3_wide) 0.095 / **v4_plr2 0.093(챔피언)**. **가치기반·PG는 현행체계(64룰)는 넘어도 LB-T4엔 전부 0/17 완패** → "PPO만 규칙 압도"의 실증(v3 알고리즘_검토 분석판 확정). holdout250도 동일 계층.
- **★P1 NCRP(비천리안 제한 롤아웃 플래너, `planner_policy.py`·`planner_eval.py`·`leaf_value.py`, 커밋 9587d67·6e5234b) = 채택**: 오라클(`rollout_oracle.py`)의 deepcopy가 rng까지 복제=**천리안**. 배포가능판 = 복제 후 `ev_manager.set_seed(default_rng(...))` **재시드(미래 무지)** + **h-결정 절단** + 후보 K8·엄격개선 스위치 + **j번째 상상미래 후보 간 CRN 공유**(⚠️후보별 독립 재시드=그리드 v1 전멸 원인, 실현노이즈가 랭킹 오염) + m회 평균. 최종 **K8·h10·m8·leaf none**. 챔피언(동결) 위 시도17 **0.0905→0.0862(+0.0043, 16승1무0패)**·holdout250 30ep **0.1481→0.1450(유의 손실 0곳: 149승 101무; 평균기준 236/250 개선)**, **오라클 도달률 57.1%→65.4%**(W/T/L=지역별 에피 95%CI 유의성, `_paired`). 손실분해: greedy 57.1%→NC 65.4%→h10천리안 75.7%→완전천리안 100%(다음 레버=m·h 증액). 결정당 ~0.2s(천리안)~2.3s(m8, 노드경합) — 분단위 결정간격 대비 무시가능.
- **음성 결과(정직 기록)**: **SMDP γ^Δt 기각**(DQN 0.190→0.243 악화, γ스윕 정합), **리프 가치망 기각**(val MAE 0.065가 후보 간 미세 Q차이 압도 + 오프폴리시 애프터스테이트 편향 — h절단+무리프 우세, `leaf_dataset.npz`·`leaf_value.pt` 보존하되 미사용), 스위치 마진 ε은 m 증가가 대체.
- **P2 DAR(`RewardRedesignWrapper` mode `pdrwog_da`, 커밋 50debc9)**: 입원창 계상 보상을 **합 보존**하며 결정 스텝으로 재배치(`r'=r̂(a)+[r_woG−성숙예정 r̂합]`, r̂=getSurvProb(now+E[transport]+handover)/prev). return-equivalence 4.4e-16, 크레딧질량 97.5% dispatch로 이동. **기각**: 시도17 0.1016 vs v3_wide 0.0916(+0.0101 악화, LB 17/0→15/2) — PPO의 GAE가 이미 신용할당 처리 + r̂의 ETA 근사가 편향 주입(결정귀속은 off-policy TD서만 잠재가치). ⚠️**train_ppo_feature는 `MCI_OBS_VARIANT`를 스스로 설정 안 함(호출자 책임)** — 1차 런이 essential(209)로 학습돼 무효(`v5_dar209_s0` 보존)→재학습.
- 산출물: 모델 `results/rl/zoo/{dqn,qrdqn,sacd,reinforce}_s{0,1,2}`·`dqn_smdp_s0`·`redesign/v5_dar_s0`, CSV `results/rl/zoo/v5_*`·`redesign/planner_*`, 정본 `docs/v5_알고리즘비교_보고서_2026-07-14.md`. P4(ExIt-online, 플래너=라벨러) 조건부 개방(미착수).
