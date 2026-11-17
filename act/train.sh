#!/bin/bash
# 训练 ACT。用法（仓库根目录）：bash act/train.sh <数据集文件夹名> <输出名> <步数> [保存间隔]
#   bash act/train.sh lerobot_full act_full 60000 10000
DS=$1; OUT=$2; STEPS=$3; SAVE=${4:-10000}
cd ~/Reset_station
rm -rf outputs/$OUT
lerobot-train \
  --dataset.repo_id=local/$DS \
  --dataset.root=$HOME/Reset_station/data/$DS \
  --policy.type=act \
  --policy.chunk_size=20 \
  --policy.n_action_steps=5 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --num_workers=8 \
  --steps=$STEPS \
  --save_freq=$SAVE \
  --log_freq=200 \
  --output_dir=$HOME/Reset_station/outputs/$OUT
