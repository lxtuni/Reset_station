#!/bin/bash
# 编译并运行 C++ 推理程序。用法：bash build_run.sh [只运行不编译时加 run]
# 三步：nvcc 编译 kernel → g++ 编译主程序并链接 TensorRT 和 CUDA 运行库 → 运行
source ~/miniforge3/etc/profile.d/conda.sh
conda activate trt86
export GLIBC_TUNABLES=glibc.rtld.execstack=2
SP=$(python -c "import site; print(site.getsitepackages()[0])")
CUDA=$CONDA_PREFIX/targets/x86_64-linux                 # conda 装的 CUDA 头文件和 libcudart
TRT_INC=~/act_cpp/trt_headers/include                   # TensorRT 8.6 头文件
TRT_LIB=$SP/tensorrt_libs                               # pip 装的 TensorRT 库（libnvinfer.so.8）
CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
cd "$(dirname "$0")"
if [ "$1" != run ]; then
  set -e
  # 1. kernel：nvcc 把 __global__ 函数编译成显卡指令，其余部分交给 g++
  nvcc -O2 -ccbin $CXX -I$CUDA/include -c preprocess.cu -o preprocess.o
  # 2. 主程序。-l:libnvinfer.so.8 是因为 pip 包里的库文件名带版本号
  $CXX -O2 -std=c++17 main.cpp preprocess.o -I$TRT_INC -I$CUDA/include \
       -L$TRT_LIB -l:libnvinfer.so.8 -L$CUDA/lib -lcudart -o act_infer
  set +e
  ls -lh act_infer
fi
# 3. 运行时要能找到：TensorRT、cuDNN 8、cuBLAS、CUDA 运行库、conda 的 C++ 标准库
export LD_LIBRARY_PATH=$TRT_LIB:$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$CUDA/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
./act_infer ~/act_trt/deploy_final/act_rs_fp32.engine ~/Reset_station/outputs/cpp_sample
