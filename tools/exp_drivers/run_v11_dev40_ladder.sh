#!/usr/bin/env bash
# v11 NCRP 조건탐색 + MILP(OR) 사다리 — 튜닝 전용 dev40(train1000 p2 40좌표) × seed 8000..8019.
#
# 불변식(v10 프로토콜 승계): MCI_CAP_GATE=occ, obs=essential+load+valid, MCI_H_PAD=47,
# base=v10_random4_1000_pointer_s0 greedy(같은 시드 재주행 paired). 대표점250은 미사용.
# 각 팔은 planner_eval.py 재개기능으로 중단·재실행 안전(기존 (region,ep) 스킵).
#
# 사용: nohup bash tools/exp_drivers/run_v11_dev40_ladder.sh > <log> 2>&1 &
set -u
cd /home/ryu/MCI_UAV
P=/home/ryu/anaconda3/envs/UAV/bin/python
MODEL=results/rl/redesign/v10_random4_1000_pointer_s0
MANIFEST=scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json
REGIONS=scoreboard/v11_ncrp_dev40_regions.txt
OUTDIR=results/scoreboard/v11/dev40
WORKERS=${WORKERS:-90}
NEPS=${NEPS:-20}
SEED0=8000
mkdir -p "$OUTDIR"

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

# --- 1) 저비용 팔: OR 기준선·천리안 상한(빠른 파이프라인 검증 겸용) ---
run milp                --policy milp
run milp_future         --policy milp --milp_future
run clair_h20m1         --K 8 --h 20 --m 1 --clairvoyant
run clair_hinfm1        --K 8 --h -1 --m 1 --clairvoyant
# --- 2) 핵심 (h, m) 격자 ---
run ref_K8h10m16        --K 8 --h 10 --m 16
run K8h20m16            --K 8 --h 20 --m 16
run K8h10m32            --K 8 --h 10 --m 32
run K8h40m16            --K 8 --h 40 --m 16
run K8h20m32            --K 8 --h 20 --m 32
# --- 3) 같은 예산 내 개선(할당·판정규칙)·OR 결합 ---
run K8h20m16_sh         --K 8 --h 20 --m 16 --alloc sh
run K8h20m16_z1         --K 8 --h 20 --m 16 --switch_z 1.0
run K8h20m16_milpinj    --K 8 --h 20 --m 16 --cand_source ppo+milp
echo "[v11] dev40 사다리 전체 완료 $(date)"
