#!/bin/bash
# 服务器上：搜索加蒸馏的一轮。
#   1. 采集：SmolVLA 正常跑；失败的回合用同一个种子重放并插入脱身动作，找到能成功的做法就留下
#   2. 和原来的 820 回合合并，从第二轮 DAgger 的模型接着训练动作头 5000 步
#   3. 评测两个检查点，各两批种子（对照 314/360）
# 用法：bash search_rl.sh <进程数> <每个进程的回合数>
# 进程数先用 3 个：第三轮 DAgger 正在训练，显存要留够。等它结束再训练这一轮。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine; R=sr1
NP=${1:-3}; NE=${2:-500}
T0=$(date +%s)
for i in $(seq 0 $((NP - 1))); do
  nice -n 10 python vla/collect_search.py $M $E data/raw_${R}_$i $NE $((300000 + i * 10000)) > data/collect_${R}_$i.log 2>&1 &
done
wait
grep -ah "可复现性|完成：|各种救法" data/collect_${R}_*.log | cut -c1-260
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
while ! grep -qa "DAGGER3B DONE" ~/logs/dagger3b.log; do sleep 30; done      # 等第三轮 DAgger 完全结束再占显卡训练
DIRS=""; for i in $(seq 0 $((NP - 1))); do DIRS="$DIRS data/raw_${R}_$i"; done
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast \
  $DIRS data/vla_$R 2>&1 | tail -1
echo "数据就绪，用时 $(( $(date +%s) - T0 )) 秒"
unset HF_HUB_OFFLINE OMP_NUM_THREADS
bash ~/vla/train_vla.sh vla_$R 5000 2500 64 $HOME/Reset_station/$M vla_$R 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -a -E "step:[0-9K]+ |End of training|Traceback|Error" | grep --line-buffered -a -vE "downloading" | stdbuf -oL cut -c1-140 | awk 'NR % 20 == 0 || /End|Trace|Error/'
echo "训练结束，总用时 $(( $(date +%s) - T0 )) 秒"
export PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cat > ~/vla/cfg_$R.txt <<EOF
${R}_2500|vla_$R/002500|--bins --exec_steps 1 --num_steps 3
${R}_5000|vla_$R/005000|--bins --exec_steps 1 --num_steps 3
${R}_5000_b|vla_$R/005000|--bins --exec_steps 1 --num_steps 3 --seed0 1000
${R}_2500_b|vla_$R/002500|--bins --exec_steps 1 --num_steps 3 --seed0 1000
EOF
bash ~/vla/par_eval.sh cfg_$R.txt 4 > /dev/null
for f in ${R}_2500 ${R}_5000 ${R}_2500_b ${R}_5000_b; do echo "== $f"; grep -a -E "cm/s|合计" outputs/ev/$f.log | cut -c1-110; done
echo "SEARCH RL DONE，总用时 $(( $(date +%s) - T0 )) 秒"
