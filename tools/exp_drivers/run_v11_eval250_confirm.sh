#!/usr/bin/env bash
# v11 대표점250 확인 실험 — dev40에서 사전확정한 팔만 실행(선택은 이미 끝났고 여기선 확인만).
#   1) K8h20m16          : (h,m) 내부 최적점 = 1순위
#   2) K8h20m16_milpinj  : NCRP+OR(MILP 후보주입) 하이브리드 = 2순위(사전확정)
#   3) clair_h20m1       : 같은 h20 천리안 상한(격차분해·도달률용)
# seed 0..29 = 기존 v10 4행 cube 와 동일 → tools/v11_scoreboard.py 로 행 추가.
set -u
cd /home/ryu/MCI_UAV
P=/home/ryu/anaconda3/envs/UAV/bin/python
MODEL=results/rl/redesign/v10_random4_1000_pointer_s0
MANIFEST=scenarios/manifests/sigungu_osrm_eval250_representative_manifest.json
OUTDIR=results/scoreboard/v11/eval250
WORKERS=${WORKERS:-108}
mkdir -p "$OUTDIR"
run() {
  tag=$1; shift; out=$OUTDIR/$tag.csv; have=0
  if [ -f "$out" ]; then have=$(( $(wc -l < "$out") - 1 )); fi
  if [ "$have" -ge 7500 ]; then echo "[skip] $tag ($have/7500)"; return 0; fi
  echo "=== [$(date +%m-%d\ %H:%M)] $tag (기존 $have/7500) ==="
  MCI_CAP_GATE=occ MCI_REWARD_MODE=woG PYTHONIOENCODING=utf-8 \
  $P src/rl_src/planner_eval.py --model_dir "$MODEL" --manifest "$MANIFEST" \
    --n_eps 30 --seed0 0 --leaf none --obs_variant essential+load+valid --h_pad 47 \
    --workers "$WORKERS" --chunk 3 --tag "$tag" --out "$out" "$@" \
    2>&1 | grep -E "^\[planner\] (regions|완료|Δ 전체)|FAIL"
  echo "--- [$(date +%m-%d\ %H:%M)] $tag 완료"
}
run K8h20m16         --K 8 --h 20 --m 16
run K8h20m16_milpinj --K 8 --h 20 --m 16 --cand_source ppo+milp
run clair_h20m1      --K 8 --h 20 --m 1 --clairvoyant
echo "[v11] eval250 확인 전체 완료 $(date)"
