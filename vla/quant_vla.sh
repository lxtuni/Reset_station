#!/bin/bash
# 服务器上：SmolVLA 最终模型的 FP32 / FP16 / INT8 引擎。
#   1. （lerobot 环境）从示教里抽 200 帧做校准数据
#   2. （ortgpu 环境）转 FP16、做 INT8 量化、编译三个引擎、测速、比精度
source ~/miniforge3/etc/profile.d/conda.sh
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2; Q=outputs/quant_vla
cd ~/Reset_station
[ -f outputs/calib_vla.npz ] || bash ~/vla/run_py.sh vla/calib_dump.py vla $M outputs/calib_vla.npz 200 | grep "校准数据"
conda activate ortgpu
pip install -q onnxconverter-common 2>&1 | tail -1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
rm -rf $Q                                                  # 上一次失败留下的几百个中间文件
timeout 3000 nice -n 10 python vla/quant.py $E/smolvla_n3.onnx outputs/calib_vla.npz $Q ${1:-Conv,MatMul,Gemm} 2>&1 \
  | grep -vE "ScatterND|colored|WARNING The logger|Loading bytes|DeprecationWarning|get_input_metadata" | cut -c1-400
ls -lh $Q | grep -E "engine|onnx$" | cut -c1-100
echo "QUANT VLA DONE"
