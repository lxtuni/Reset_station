#!/bin/bash
# 把复位台 ACT 的 ONNX 编译成 TensorRT 8.6.1 引擎并测速。用法：bash trt_rs.sh <deploy 目录名，如 deploy_30k>
# 在单独的文件夹 ~/act_trt/<名字> 里做，避免和官方 ACT 的 sample.npz 混在一起。
source ~/miniforge3/etc/profile.d/conda.sh
export GLIBC_TUNABLES=glibc.rtld.execstack=2
D=${1:-deploy}
W=~/act_trt/$D; mkdir -p $W; cd $W
cp ~/Reset_station/outputs/$D/* .
cp ~/Reset_station/act/trt_data.py ~/Reset_station/act/trt_bench.py .
conda activate trt
polygraphy surgeon sanitize act_rs.onnx --fold-constants -o act_rs_folded.onnx 2>&1 | grep -E "Folded|Error|\[E\]" | cut -c1-200 | tail -4
conda activate trt86
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
for P in fp32 fp16; do
  echo "=== TensorRT 8.6.1 $P ==="
  FLAG=""; [ $P = fp16 ] && FLAG="--fp16"
  polygraphy run act_rs_folded.onnx --trt $FLAG --onnxrt --data-loader-script trt_data.py \
    --save-engine act_rs_$P.engine --atol 1e-2 --rtol 1e-2 2>&1 \
    | grep -E "PASSED|FAILED|not supported|Unsupported|Error|\[E\]|DCHECK" | cut -c1-260 | tail -6
done
python trt_bench.py act_rs_fp32.engine act_rs_fp16.engine 2>&1 | grep -v "^\[" | tail -4
echo "TRT RS DONE"
