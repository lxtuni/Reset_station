#!/bin/bash
# 服务器上：SmolVLA 的第三轮 DAgger，只针对卡住的状态。
#   1. 6 个进程各跑 300 回合，只保存“到第 60 步还没夹住”的回合的后半段，标签是教师的动作
#   2. 和原来的 820 回合合并（不含那批自我模仿的数据），从第二轮的模型接着训练 5000 步
#   3. 评测两个检查点，各两批种子（对照：314/360）
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine; R=d3
T0=$(date +%s)
for i in 0 1 2 3 4 5; do
  nice -n 10 python vla/collect_stuck.py $M $E data/raw_${R}_$i 300 $((80000 + i * 1000)) > data/collect_${R}_$i.log 2>&1 &
done
wait
grep -h "300/300" data/collect_${R}_*.log | cut -c1-220
grep -h "完成" data/collect_${R}_*.log
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast \
  data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_2 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 data/vla_$R 2>&1 | tail -1
echo "数据就绪，用时 $(( $(date +%s) - T0 )) 秒"
unset HF_HUB_OFFLINE OMP_NUM_THREADS
bash ~/vla/train_vla.sh vla_$R 5000 2500 64 $HOME/Reset_station/$M vla_$R 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -E "step:[0-9K]+ |End of training|Traceback|Error" | grep --line-buffered -vE "downloading" | stdbuf -oL cut -c1-140 | awk 'NR % 20 == 0 || /End|Trace|Error/'
echo "训练结束，总用时 $(( $(date +%s) - T0 )) 秒"
cat > ~/vla/cfg_$R.txt <<EOF
${R}_2500|vla_$R/002500|--bins --exec_steps 1 --num_steps 3
${R}_5000|vla_$R/005000|--bins --exec_steps 1 --num_steps 3
${R}_5000_b|vla_$R/005000|--bins --exec_steps 1 --num_steps 3 --seed0 1000
${R}_2500_b|vla_$R/002500|--bins --exec_steps 1 --num_steps 3 --seed0 1000
EOF
bash ~/vla/par_eval.sh cfg_$R.txt 4 > /dev/null
for f in ${R}_2500 ${R}_5000 ${R}_2500_b ${R}_5000_b; do echo "== $f"; grep -E "cm/s|合计" outputs/ev/$f.log | cut -c1-110; done
echo "DAGGER3 DONE，总用时 $(( $(date +%s) - T0 )) 秒"
