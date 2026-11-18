#!/bin/bash
# 服务器上（后台用）：在 lerobot 环境、仓库根目录下跑一条 python 命令，结尾打印 PY DONE。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1
cd ~/Reset_station
nice -n 10 python "$@" 2>&1 | grep -vE "Error while downloading|download.py|^$|Exception ignored|NoneType|renderer.py|egl/__init__|^Traceback \(most recent call last\):$" | cut -c1-900
echo "PY DONE"
