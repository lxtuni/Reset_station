#!/bin/bash
# 服务器上：训练残差策略（SmolVLA 冻结），然后评测几个检查点。
# 用法：bash resid_run.sh [总步数=200000] [并行环境=4] [alpha=0.3] [输出名=resid] [修正量代价=0.02] [熵系数=0.003]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine
TOTAL=${1:-200000}; NENV=${2:-4}; ALPHA=${3:-0.3}; OUT=outputs/${4:-resid}
rm -rf $OUT
F="Error while downloading|download.py|^$|Exception ignored|NoneType|renderer.py|egl/__init__|^Traceback \(most recent call last\):$|Loading|Reducing|torch_dtype"
nice -n 5 python vla/train_resid.py $M $E $OUT $TOTAL $NENV $ALPHA ${5:-0.02} ${6:-0.003} 2>&1 | grep --line-buffered -vE "$F" | stdbuf -oL cut -c1-300
ls $OUT
echo "##### 评测（种子 0–29；同一个引擎不加修正是 160/180）"
for C in $(ls $OUT | grep -E "resid_(50000|100000)_steps.zip|resid_final.zip"); do
  python vla/eval_resid.py $M $E $OUT/$C --alpha $ALPHA --json outputs/ev/${4:-resid}_${C%.zip}.json > outputs/ev/${4:-resid}_${C%.zip}.log 2>&1 &
done
wait
for C in $(ls $OUT | grep -E "resid_(50000|100000)_steps.zip|resid_final.zip"); do
  echo "=== $C ==="; grep -E "cm/s|合计" outputs/ev/${4:-resid}_${C%.zip}.log | cut -c1-200; grep "Error" outputs/ev/${4:-resid}_${C%.zip}.log | grep -v downloading | tail -1 | cut -c1-200
done
echo "##### 最终模型在新种子上（1000–1029；不加修正是 154/180）"
python vla/eval_resid.py $M $E $OUT/resid_final.zip --alpha $ALPHA --seed0 1000 2>&1 | grep -E "cm/s|合计" | cut -c1-200
echo "RESID RUN DONE"
