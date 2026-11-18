#!/bin/bash
# 服务器上：SmolVLA 的第二轮 DAgger，这次偏向快船（失败最集中的地方）。
#   1. DAgger 采集 160 回合：1 个进程慢船（0–10 cm/s），3 个进程快船（10–20 cm/s）
#   2. 另外补 100 个快船的教师示教
#   3. 和之前的全部数据（400 示教 + 第一轮 160）合并成 vla_d2
#   4. 从第一轮 DAgger 的模型接着训练 5000 步，评测
# 用法：bash dagger_vla2.sh <起点检查点，如 vla_d1/005000>
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd ~/Reset_station
C=$1; M=outputs/${C%/*}/checkpoints/${C#*/}/pretrained_model
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
T0=$(date +%s)
VMIN=(0.0 0.10 0.10 0.10); VMAX=(0.10 0.20 0.20 0.20)
for i in 0 1 2 3; do
  nice -n 10 python vla/collect_dagger_vla.py $M --episodes 40 --seed0 $((50000 + i * 1000)) \
    --v_min ${VMIN[$i]} --v_max ${VMAX[$i]} --out data/raw_vd2_$i > data/collect_vd2_$i.log 2>&1 &
done
nice -n 10 python act/collect_demos.py --v_min 0.10 --v_max 0.20 --episodes 100 --seed0 60000 --out data/raw_fast > data/collect_fast.log 2>&1 &
wait
grep -h "完成" data/collect_vd2_*.log data/collect_fast.log; grep -h "40/40" data/collect_vd2_*.log | cut -c1-60
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast \
  data/vla_d2 2>&1 | tail -1
echo "数据就绪，用时 $(( $(date +%s) - T0 )) 秒"
unset HF_HUB_OFFLINE OMP_NUM_THREADS
bash ~/vla/train_vla.sh vla_d2 5000 2500 64 $HOME/Reset_station/$M vla_d2 2>&1 \
  | tr '\r' '\n' | grep -E "step:[0-9K]+ |End of training|Traceback|Error" | grep -vE "downloading" | cut -c1-140 | awk 'NR % 20 == 0 || /End|Trace|Error/'
echo "训练结束，总用时 $(( $(date +%s) - T0 )) 秒"
bash ~/vla/par_eval.sh cfg_d2.txt 3
echo "DAGGER2 DONE"
