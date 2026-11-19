#!/bin/bash
# 服务器上：方案 F。先冒烟（永远选“继续”，前两档各 6 回合），再训练 DQN，最后评测。
# 用法：bash prim_run.sh [决策步数=20000] [并行环境=6] [输出名=prim]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine
TOTAL=${1:-20000}; NENV=${2:-6}; NAME=${3:-prim}; OUT=outputs/$NAME
F="Error while downloading|download.py|^$|Exception ignored|NoneType|renderer.py|egl/__init__|^Traceback \(most recent call last\):$|Loading|Reducing|torch_dtype"
rm -rf $OUT
python vla/eval_prim.py $M $E none --episodes 4 2>&1 | grep --line-buffered -vE "$F" | stdbuf -oL cut -c1-200 | tail -3
echo "##### 训练"
nice -n 5 python vla/train_prim.py $M $E $OUT $TOTAL $NENV 2>&1 | grep --line-buffered -vE "$F" | stdbuf -oL cut -c1-260
echo "##### 评测（同一个引擎不加选择器：种子 0–29 是 160/180，种子 1000–1029 是 154/180）"
for C in prim_10000_steps prim_final; do
  for S in 0 1000; do
    [ -f $OUT/$C.zip ] && python vla/eval_prim.py $M $E $OUT/$C.zip --seed0 $S --json outputs/ev/${NAME}_${C}_$S.json > outputs/ev/${NAME}_${C}_$S.log 2>&1 &
  done
done
wait
for C in prim_10000_steps prim_final; do for S in 0 1000; do echo "=== $C，种子 $S 起 ==="; grep -E "cm/s|合计|Error" outputs/ev/${NAME}_${C}_$S.log | grep -v downloading | cut -c1-230; done; done
echo "PRIM RUN DONE"
