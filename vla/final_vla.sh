#!/bin/bash
# 服务器上：对最终模型（DAgger 之后）做部署收尾。显卡此时空闲，测出的时间可以直接用。
#   1. PyTorch 推理耗时（不同去噪次数）
#   2. 导出 ONNX（去噪 3 次，32 位）→ ONNX Runtime 测速 → 编译 TensorRT 引擎并测速
#   3. 闭环验证：TensorRT 引擎控制机械臂；以及 32 位的 PyTorch 模型控制机械臂
# 用法：bash final_vla.sh <检查点，如 vla_d1/005000>
source ~/miniforge3/etc/profile.d/conda.sh
C=$1; M=outputs/${C%/*}/checkpoints/${C#*/}/pretrained_model; E=outputs/export_final
echo "##### 1. PyTorch 推理耗时"
bash ~/vla/run_py.sh vla/bench_vla.py $M | grep -E "去噪|拟合|半精度|显存|参数量"
echo "##### 2. 导出"
bash ~/vla/run_py.sh vla/export_vla.py $M $E 3 | grep -E "转成 32 位|导出|节点|误差|providers"
conda activate ortgpu
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
cd ~/Reset_station
python vla/ort_trt_bench.py $E smolvla_n3.onnx 2>&1 | grep -E "^ONNX Runtime|^TensorRT|编译" | grep -v "失败 —— AttributeError" | cut -c1-300
conda deactivate; unset LD_LIBRARY_PATH
echo "##### 3. 闭环验证"
conda activate lerobot
# 在训练环境里加装 TensorRT 的运行库（--no-deps：只装它自己，不动 PyTorch 用的 CUDA 库）
pip install -q --no-deps tensorrt_cu13==11.5.0.63 tensorrt_cu13_libs==11.5.0.63 tensorrt_cu13_bindings==11.5.0.63 2>&1 | tail -1
python -c "import tensorrt, torch; print('lerobot 环境里 TensorRT', tensorrt.__version__, '| torch 仍可用显卡:', torch.cuda.is_available())"
cat > ~/vla/cfg_final.txt <<EOF
f_trt_n3e1|$C|--bins --exec_steps 1 --num_steps 3 --backend trt --engine $E/smolvla_n3.onnx.fp32.engine
EOF
rm -f outputs/ev/f_trt_n3e1.* outputs/ev/f_fp32_n3e1.*
bash ~/vla/par_eval.sh cfg_final.txt 1
grep -E "TensorRT|Error|Traceback" outputs/ev/f_trt_n3e1.log | grep -v downloading | head -5 | cut -c1-300
cat > ~/vla/cfg_final2.txt <<EOF
f_fp32_n3e1|$C|--bins --exec_steps 1 --num_steps 3 --fp32
EOF
bash ~/vla/par_eval.sh cfg_final2.txt 1
echo "FINAL DONE"
