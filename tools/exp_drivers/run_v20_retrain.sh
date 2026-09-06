#!/usr/bin/env bash
# v20 — 시나리오 물리 파라미터별 교사 재학습.
# v19 national 과 레시피·데이터(train6000)·시드 전부 동일하고 런타임 물리 노브만 다르다.
# 목적: (1) 자원/부하/용량축 곡선에서 OOD 교란 제거 (2) 교사가 스스로 채굴하는 임계값이
#       파라미터에 따라 어떻게 움직이는지 = 임계값 일반화 법칙의 확증 팔.
set -u
P=/home/ryu/anaconda3/envs/UAV/bin/python
REPO=/home/ryu/MCI_UAV
MAN=$REPO/scenarios/manifests/sigungu30_train6000_manifest.json
cd $REPO
mkdir -p results/rl/v20/logs

run_one () {  # tag  "ENV=V ENV2=V2"
  tag=$1; envs=$2
  [ -f results/rl/v20/$tag/final_model.zip ] && { echo "skip $tag"; return; }
  ( export MCI_OBS_VARIANT=field MCI_H_PAD=47 MCI_CAP_GATE=occ \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
    for kv in $envs; do export "$kv"; done
    nohup $P src/rl_src/train_ppo_feature.py \
      --config_path $MAN --total_timesteps 10000000 --n_envs 8 --vec subproc --seed 0 \
      --log_dir results/rl/v20/$tag --extractor pointer --reward_mode pdrwog --norm_reward \
      --learning_rate 3e-4 --lr_anneal --target_kl 0.03 --n_epochs 5 --n_steps 512 \
      --batch_size 512 --embed_dim 64 --ctx_dim 128 --head_hidden 128 --n_attn_blocks 0 \
      --checkpoint_freq 2000000 --save_vecnormalize \
      > results/rl/v20/logs/$tag.out 2> results/rl/v20/logs/$tag.err &
    echo "launched $tag pid=$! envs=[$envs]" )
}

run_one amb10  "MCI_AMB_NUM=10"
run_one amb20  "MCI_AMB_NUM=20"
run_one uav6   "MCI_UAV_NUM=6"
run_one n200   "MCI_INCIDENT_SIZE=200"
run_one n50    "MCI_INCIDENT_SIZE=50"
run_one capa05 "MCI_CAPA_SCALE=0.5"
