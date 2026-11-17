#!/bin/bash
# DAgger 第二步：把示教 + DAgger 数据合并成一个数据集，从已有检查点接着训练。
# 用法（仓库根目录）：bash act/dagger_train.sh <起点检查点目录> <数据集名> <输出名> <步数> <原始数据文件夹...>
INIT=$1; DS=$2; OUT=$3; STEPS=$4; shift 4
rm -rf data/$DS outputs/$OUT
python act/convert_to_lerobot.py "$@" data/$DS 2>&1 | tail -2
lerobot-train \
  --dataset.repo_id=local/$DS --dataset.root=$HOME/Reset_station/data/$DS \
  --policy.path=$INIT --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --num_workers=8 --steps=$STEPS --save_freq=10000 --log_freq=200 \
  --output_dir=$HOME/Reset_station/outputs/$OUT
