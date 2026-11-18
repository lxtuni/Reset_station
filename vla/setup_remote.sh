#!/bin/bash
# 在 4090 服务器的用户目录里搭环境（不需要管理员权限，不影响其他用户）。可以重复运行。
# 服务器连不上 GitHub / huggingface.co，全部走国内镜像；LeRobot 源码和 Reset_station 由本机 rsync 过来。
set -e
cd ~
PIP_IDX=https://pypi.tuna.tsinghua.edu.cn/simple
if [ ! -d ~/miniforge3 ]; then
  F=Miniforge3-26.9.4-2-Linux-x86_64.sh
  curl -sL -o /tmp/$F https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/$F
  bash /tmp/$F -b -p ~/miniforge3 > /dev/null; rm /tmp/$F
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda config --set channel_alias https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud > /dev/null
conda config --set show_channel_urls yes
conda env list | grep -q "^lerobot " || conda create -y -q -n lerobot -c conda-forge python=3.12 ffmpeg > /dev/null
conda activate lerobot
python --version
pip config set global.index-url $PIP_IDX > /dev/null
pip install -q torch torchvision                              # PyPI 上的 Linux 版自带 CUDA
pip install -q -e "$HOME/lerobot[smolvla,training,dataset]"
pip install -q mujoco gymnasium stable-baselines3 onnx onnxscript onnxruntime-gpu matplotlib imageio
# 以后每次登录自动走 Hugging Face 镜像、MuJoCo 用 EGL 渲染
grep -q HF_ENDPOINT ~/.bashrc || printf '\nexport HF_ENDPOINT=https://hf-mirror.com\nexport MUJOCO_GL=egl\n' >> ~/.bashrc
python - <<'EOF'
import torch, lerobot, mujoco, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("lerobot", lerobot.__version__, "| mujoco", mujoco.__version__, "| transformers", transformers.__version__)
EOF
echo "SETUP DONE"
