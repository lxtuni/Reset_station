#!/bin/bash
# 服务器上：ACT 量化第二轮（诊断 FP16、混合精度、每层一个刻度的 INT8），然后对能编译的引擎做闭环。
source ~/miniforge3/etc/profile.d/conda.sh
M=outputs/act_srv/checkpoints/040000/pretrained_model; E=outputs/deploy_act4090; Q=outputs/quant_act_r2
cd ~/Reset_station; rm -rf $Q
conda activate ortgpu
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
timeout 2400 python vla/quant2.py $E/act_rs.onnx outputs/calib_act.npz $Q 2>&1 \
  | grep -vE "colored|WARNING The logger|Loading bytes|DeprecationWarning|get_input_metadata|truncated|warnings.warn|Begin node|End node|ModelImporter" | cut -c1-420
conda activate lerobot; unset LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
echo "##### 闭环（每步重新推理，种子 0–29；PyTorch 和 FP32 引擎都是 159/180）"
for P in fp16mix int8 int8pct; do
  [ -f $Q/$P.engine ] || continue
  nice -n 10 python act/eval_act.py $M --bins --episodes 30 --exec_steps 1 --backend trt --engine $Q/$P.engine \
    --json outputs/ev/act4090_r2_$P.json > outputs/ev/act4090_r2_$P.log 2>&1
  echo "=== $P ==="; grep -E "合计|时间分解" outputs/ev/act4090_r2_$P.log | cut -c1-200
done
echo "QUANT ACT2 DONE"
