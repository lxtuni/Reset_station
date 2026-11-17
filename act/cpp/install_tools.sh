#!/bin/bash
# 给 trt86 环境补上 C++/CUDA 开发要用的三样东西（都装在用户目录，不需要管理员权限）：
#   1. nvcc：CUDA 的编译器，把 .cu 文件里的 kernel 编译成显卡能跑的代码（版本和已装的 CUDA 运行库 12.9 对应）
#   2. g++ 13：nvcc 12.9 不认系统自带的 g++ 15，需要一个旧一点的 C++ 编译器配合
#   3. TensorRT 8.6 的 C++ 头文件：pip 装的 tensorrt 只有 .so 库，没有 NvInfer.h；头文件在 NVIDIA 的公开仓库里
source ~/miniforge3/etc/profile.d/conda.sh
conda activate trt86
pip install -q nvidia-cuda-nvcc-cu12==12.9.86 2>&1 | tail -2
conda install -y -q -n trt86 -c conda-forge gxx_linux-64=13 2>&1 | tail -3
cd ~/act_cpp
[ -d trt_headers ] || git clone -q --depth 1 -b release/8.6 --filter=blob:none --sparse https://github.com/NVIDIA/TensorRT.git trt_headers
(cd trt_headers && git sparse-checkout set include && ls include | head -20)
SP=$(python -c "import site; print(site.getsitepackages()[0])")
find $SP/nvidia/cuda_nvcc -maxdepth 2 | head -20
ls $CONDA_PREFIX/bin | grep -E "g\+\+|gcc$" | head
