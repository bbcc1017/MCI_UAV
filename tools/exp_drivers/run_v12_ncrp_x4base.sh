#!/usr/bin/env bash
# v12 wave 2 항목3 — base 개선이 NCRP 플래너 상단으로 전이되는가?
#
# wave 1 에서 X4_attn0(attention 제거)이 greedy 로 v10 대비 +0.00305(대표점250) 개선했다.
# 플래너는 base 정책 위에서 롤아웃하므로 base 가 좋아지면 최종 스택도 밀려 올라갈 수 있다.
# 이를 v11 과 **직접 비교 가능한 형태**로 재는 것이 목적이다.
#
# 프로토콜: v11 dev40 사다리와 **완전 동일**(튜닝 전용 dev40 40좌표 × seed 8000..8019,
#   occ/essential+load+valid/H_PAD 47, chunk 2). 다른 점은 --model_dir 하나뿐 →
#   기존 results/scoreboard/v11/dev40/{K8h20m16,K8h20m16_milpinj}.csv (base=v10)와 팔 대 팔 비교.
#   ⚠️ 대표점250 은 쓰지 않는다(v11 관례: 플래너 조건은 dev40 에서만 스크리닝).
#
# ★검정: v11 교훈대로 **팔 대 팔 paired** 로 본다. 각 팔의 base 대비 Δ만 비교하면 CI 가 커져
#   실제 차이를 놓친다(v11 에서 'h 는 죽은 축' 결론이 이 때문에 뒤집혔다).
#   단 base 가 서로 다르므로(v10 vs X4) pdr_base 일치 게이트는 적용 불가 — pdr_planner 를
#   같은 (region, ep) 로 맞춰 직접 paired 한다.
set -u
cd /home/ryu/MCI_UAV
P=/home/ryu/anaconda3/envs/UAV/bin/python
MODEL=results/rl/redesign/v12_x4_attn0_s0
MANIFEST=scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json
REGIONS=scoreboard/v11_ncrp_dev40_regions.txt
OUTDIR=results/scoreboard/v12/ncrp_x4base
WORKERS=${WORKERS:-60}
NEPS=${NEPS:-20}
SEED0=8000
mkdir -p "$OUTDIR"

[ -f "$MODEL/final_model.zip" ] || { echo "X4 모델 없음: $MODEL"; exit 1; }

run() {  # run <tag> <extra args...>
  tag=$1; shift
  out=$OUTDIR/$tag.csv
  want=$((40 * NEPS))
  have=0
  if [ -f "$out" ]; then have=$(( $(wc -l < "$out") - 1 )); fi
  if [ "$have" -ge "$want" ]; then echo "[skip] $tag ($have/$want)"; return 0; fi
  echo "=== [$(date +%H:%M:%S)] $tag  (기존 $have/$want) ==="
  MCI_CAP_GATE=occ MCI_REWARD_MODE=woG PYTHONIOENCODING=utf-8 \
  $P src/rl_src/planner_eval.py \
    --model_dir "$MODEL" --manifest "$MANIFEST" --regions_file "$REGIONS" \
    --n_eps "$NEPS" --seed0 "$SEED0" --leaf none \
    --obs_variant essential+load+valid --h_pad 47 \
    --workers "$WORKERS" --chunk 2 --tag "$tag" --out "$out" "$@" \
    2>&1 | grep -E "^\[planner\]|FAIL"
  echo "--- [$(date +%H:%M:%S)] $tag 완료"
}

run x4_K8h20m16         --K 8 --h 20 --m 16
run x4_K8h20m16_milpinj --K 8 --h 20 --m 16 --cand_source ppo+milp
echo "[v12] X4-base NCRP dev40 완료 $(date)"
