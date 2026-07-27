#!/usr/bin/env bash
# v11 dev40 사다리 2차 — 1차 결과(h 축이 레버, m 축 h10서 포화)로 추가된 팔.
#   K8h20m8  : 예산 1.0 단위에서 h20 이 h10m16 을 넘는지(배포비용 동률 비교)
#   K8hinfm16: 종단까지 롤아웃(천리안 h∞ 상한이 h20 보다 크게 높았음)
set -u
cd /home/ryu/MCI_UAV
P=/home/ryu/anaconda3/envs/UAV/bin/python
MODEL=results/rl/redesign/v10_random4_1000_pointer_s0
MANIFEST=scenarios/manifests/sigungu_osrm_train1000_random4_manifest.json
REGIONS=scoreboard/v11_ncrp_dev40_regions.txt
OUTDIR=results/scoreboard/v11/dev40
WORKERS=${WORKERS:-30}
NEPS=${NEPS:-20}
run() {
  tag=$1; shift; out=$OUTDIR/$tag.csv; want=$((40 * NEPS)); have=0
  if [ -f "$out" ]; then have=$(( $(wc -l < "$out") - 1 )); fi
  if [ "$have" -ge "$want" ]; then echo "[skip] $tag ($have/$want)"; return 0; fi
  echo "=== [$(date +%H:%M:%S)] $tag (기존 $have/$want) ==="
  MCI_CAP_GATE=occ MCI_REWARD_MODE=woG PYTHONIOENCODING=utf-8 \
  $P src/rl_src/planner_eval.py --model_dir "$MODEL" --manifest "$MANIFEST" \
    --regions_file "$REGIONS" --n_eps "$NEPS" --seed0 8000 --leaf none \
    --obs_variant essential+load+valid --h_pad 47 --workers "$WORKERS" --chunk 2 \
    --tag "$tag" --out "$out" "$@" 2>&1 | grep -E "^\[planner\]|FAIL"
  echo "--- [$(date +%H:%M:%S)] $tag 완료"
}
run K8h20m8   --K 8 --h 20 --m 8
run K8hinfm16 --K 8 --h -1 --m 16
echo "[v11] dev40 사다리2 완료 $(date)"
