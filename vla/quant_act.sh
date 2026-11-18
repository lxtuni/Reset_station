#!/bin/bash
# 服务器上：ACT（第二轮 DAgger 后的模型）在 RTX 4090 上的 PyTorch / FP32 / FP16 / INT8 对比，含闭环成功率。
#   1. （lerobot 环境）导出 ONNX，抽校准数据，测 PyTorch 推理耗时
#   2. （ortgpu 环境）转 FP16、INT8 量化、编译引擎、测速、比精度
#   3. （lerobot 环境）每种引擎各跑 180 回合闭环
source ~/miniforge3/etc/profile.d/conda.sh
M=outputs/act_srv/checkpoints/${1:-040000}/pretrained_model; E=outputs/deploy_act4090; Q=outputs/quant_act
rm -rf $E ${Q}_Conv ${Q}_Conv_MatMul_Gemm
cd ~/Reset_station
cp vla/eval_act.py act/eval_act.py; cp vla/export_onnx.py act/export_onnx.py   # 带 TensorRT 后端的评测脚本、相机名取自模型配置的导出脚本
mkdir -p outputs/ev
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1
echo "##### 1. 导出和校准数据"
python act/export_onnx.py $M $E 2>&1 | grep -E "相机顺序|最大误差"
python vla/calib_dump.py act $M outputs/calib_act.npz 200 2>&1 | grep "校准数据"
python act/profile_torch.py $M $E 2>&1 | grep -E "整网|backbone|encoder|decoder"
echo "##### 2. 量化和引擎"
conda activate ortgpu
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')
for OPS in Conv Conv,MatMul,Gemm; do
  echo "--- INT8 量化的运算：$OPS ---"
  timeout 1500 python vla/quant.py $E/act_rs.onnx outputs/calib_act.npz ${Q}_${OPS//,/_} $OPS 2>&1 \
    | grep -vE "colored|WARNING The logger|Loading bytes|DeprecationWarning|get_input_metadata" | cut -c1-400
done
echo "##### 3. 闭环（每步重新推理，种子 0–29）"
conda activate lerobot; unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
ev() {  # 名字 其余参数
  N=$1; shift
  nice -n 10 python act/eval_act.py $M --bins --episodes 30 --exec_steps 1 "$@" --json outputs/ev/$N.json > outputs/ev/$N.log 2>&1
  echo "=== $N ==="; grep -E "cm/s|合计|时间分解" outputs/ev/$N.log | cut -c1-200; grep -E "Error:" outputs/ev/$N.log | tail -1 | cut -c1-200
}
ev act4090_torch
for P in fp32 fp16 int8; do
  [ -f ${Q}_Conv/$P.engine ] && ev act4090_$P --backend trt --engine ${Q}_Conv/$P.engine
done
[ -f ${Q}_Conv_MatMul_Gemm/int8.engine ] && ev act4090_int8all --backend trt --engine ${Q}_Conv_MatMul_Gemm/int8.engine
echo "QUANT ACT DONE"
