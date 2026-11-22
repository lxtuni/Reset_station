#!/bin/bash
# 服务器上：按状态触发的分叉搜索 + 蒸馏（collect_branch.py）。
#   1. 6 个进程采集：原策略跑一遍，停滞时分叉，试 8 个候选动作，留下成对比较后确实更好的片段
#   2. 和原有的 820 回合合并，接着训练 5000 步
#   3. 评测（对照：314/360；第三轮 DAgger：334/360）
# 用法：bash branch_run.sh <轮次名> <种子起点> <每个进程的回合数> [pilot：只采集不训练 | full] ["合并时额外加入的数据目录"]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
R=$1; S=$2; N=$3
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine
BASE="data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast"
T0=$(date +%s)
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for i in 0 1 2 3 4 5; do
  rm -rf data/raw_${R}_$i
  nice -n 10 python vla/collect_branch.py $M $E data/raw_${R}_$i $N $((S + i * 10000)) > data/collect_${R}_$i.log 2>&1 &
done
wait
grep -ah "可复现" data/collect_${R}_*.log | sort | uniq -c
for i in 0 1 2 3 4 5; do grep -a -E "回合：|各候选|搜索层数|完成：" data/collect_${R}_$i.log | tail -n 4 | cut -c1-330; done
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
[ "$4" = pilot ] && { echo "BRANCH PILOT DONE"; exit 0; }
python vla/convert_vla.py $BASE $5 data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_2 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 data/vla_$R 2>&1 | tail -n 1
[ -f data/vla_$R/awr_weights.npy ] || { echo "合并数据失败，停止"; exit 1; }
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
for f in ${R}_2500 ${R}_2500_b ${R}_5000 ${R}_5000_b; do echo "== $f $(grep -a 合计 outputs/ev/$f.log | cut -c1-40)"; done
echo "BRANCH DONE，总用时 $(( $(date +%s) - T0 )) 秒"
