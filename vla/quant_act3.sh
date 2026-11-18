#!/bin/bash
# 服务器上：定位 ACT 的 FP16 引擎坏在哪类运算上；找到后做闭环，并在显卡空闲时给全部引擎重新测速。
source ~/miniforge3/etc/profile.d/conda.sh
M=outputs/act_srv/checkpoints/040000/pretrained_model; E=outputs/deploy_act4090; Q=outputs/quant_act_r3
cd ~/Reset_station; rm -rf $Q
grep -E "^A\." ~/logs/quantact2.log | cut -c1-300
conda activate ortgpu
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
timeout 2400 python vla/quant3.py $E/act_rs.onnx outputs/calib_act.npz $Q 2>&1 \
  | grep -vE "colored|WARNING The logger|Loading bytes|DeprecationWarning|get_input_metadata|truncated|warnings.warn|TRT\] \[E\]|Begin node|End node|Node name|Invalid Node|^ITensor|^\[6\]|ModelImporter" | cut -c1-600
conda activate lerobot; unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
if [ -f $Q/fp16fix.engine ]; then
  echo "##### 闭环：修正后的 FP16 引擎"
  for S in 0 1000; do
    nice -n 10 python act/eval_act.py $M --bins --episodes 30 --exec_steps 1 --seed0 $S --backend trt --engine $Q/fp16fix.engine \
      --json outputs/ev/act4090_fp16fix_$S.json > outputs/ev/act4090_fp16fix_$S.log 2>&1
    echo "=== fp16fix，种子 $S 起 ==="; grep -E "合计|时间分解" outputs/ev/act4090_fp16fix_$S.log | cut -c1-200
  done
fi
echo "##### 新种子（1000 起）上的对照：PyTorch、FP32 引擎、INT8 引擎"
nice -n 10 python act/eval_act.py $M --bins --episodes 30 --exec_steps 1 --seed0 1000 > outputs/ev/act4090_torch_1000.log 2>&1
echo "=== PyTorch ==="; grep -E "合计" outputs/ev/act4090_torch_1000.log | cut -c1-120
for P in "fp32 outputs/quant_act_r2/fp32.engine" "int8pct outputs/quant_act_r2/int8pct.engine"; do
  set -- $P
  nice -n 10 python act/eval_act.py $M --bins --episodes 30 --exec_steps 1 --seed0 1000 --backend trt --engine $2 > outputs/ev/act4090_$1_1000.log 2>&1
  echo "=== $1 ==="; grep -E "合计" outputs/ev/act4090_$1_1000.log | cut -c1-120
done
echo "QUANT ACT3 DONE"
