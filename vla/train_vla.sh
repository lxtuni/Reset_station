#!/bin/bash
# 服务器上：从 SmolVLA 基础模型出发，在复位台示教上微调。
# 用法：bash train_vla.sh <输出名> <步数> <多少步存一次> [批大小，默认 64]
# 默认设置下只训练“动作专家”部分（约 1 亿参数），视觉语言模型的主体冻结不动。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl
OUT=$1; STEPS=$2; SAVE=$3; BS=${4:-64}
INIT=${5:-lerobot/smolvla_base}                        # 第 5 个参数：从哪个模型出发（默认基础模型；也可以是自己的检查点目录）
DS=${6:-vla_all}                                       # 第 6 个参数：数据集名
cd ~/Reset_station
rm -rf outputs/$OUT
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
lerobot-train \
  --policy.path=$INIT \
  --dataset.repo_id=local/$DS --dataset.root=$HOME/Reset_station/data/$DS \
  --policy.device=cuda --policy.push_to_hub=false \
  --policy.scheduler_decay_steps=$STEPS \
  --batch_size=$BS --num_workers=12 --steps=$STEPS --save_freq=$SAVE --log_freq=50 \
  --output_dir=$HOME/Reset_station/outputs/$OUT
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "TRAIN DONE"
