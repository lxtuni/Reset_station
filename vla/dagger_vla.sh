#!/bin/bash
# 服务器上：SmolVLA 的一轮 DAgger，全流程。
#   1. 4 个进程并行采集 160 回合（学生 70% / 教师 30% 轮流控制，标签是教师动作）
#   2. 和原来的 400 回合示教合并成新数据集 vla_d1
#   3. 从微调好的模型接着训练 5000 步
#   4. 评测
# 用法：bash dagger_vla.sh <起点检查点，如 vla_ft2/010000>
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
cd ~/Reset_station
C=$1; M=outputs/${C%/*}/checkpoints/${C#*/}/pretrained_model
T0=$(date +%s)
for i in 0 1 2 3; do
  nice -n 10 python vla/collect_dagger_vla.py $M --episodes 40 --seed0 $((40000 + i * 1000)) --out data/raw_vd1_$i \
    > data/collect_vd1_$i.log 2>&1 &
done
wait
grep -h "完成" data/collect_vd1_*.log; grep -h "40/40" data/collect_vd1_*.log | cut -c1-60
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/vla_d1 2>&1 | tail -1
echo "数据就绪，用时 $(( $(date +%s) - T0 )) 秒"
unset HF_HUB_OFFLINE OMP_NUM_THREADS
bash ~/vla/train_vla.sh vla_d1 5000 2500 64 $HOME/Reset_station/$M vla_d1 2>&1 \
  | tr '\r' '\n' | grep -E "step:[0-9K]+ |End of training|Traceback|Error" | grep -vE "downloading" | cut -c1-140 | awk 'NR % 10 == 0 || /End|Trace|Error/'
echo "训练结束，总用时 $(( $(date +%s) - T0 )) 秒"
bash ~/vla/par_eval.sh cfg_d1.txt 4
echo "DAGGER DONE"
