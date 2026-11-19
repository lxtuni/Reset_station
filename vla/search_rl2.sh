#!/bin/bash
# 服务器上：接管“搜索加蒸馏”这一轮的后续流程。
# 情况：第一次启动了 3 个采集进程，其中 1 个因为显存不够没起来（第三轮 DAgger 的训练正占着 15 GB），剩 2 个在跑。
# 这里：等第三轮 DAgger 的训练结束、显存空出来之后，再补 4 个采集进程；全部采完后合并、训练、评测。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
pkill -f "[v]la/search_rl.sh" 2>/dev/null          # 只停掉旧的调度脚本，正在跑的采集进程不受影响
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine; R=sr1
T0=$(date +%s)
while ! grep -qa "训练结束" ~/logs/dagger3b.log; do sleep 30; done
for i in 3 4 5 6; do
  nice -n 10 python vla/collect_search.py $M $E data/raw_${R}_$i 400 $((300000 + i * 10000)) > data/collect_${R}_$i.log 2>&1 &
done
for i in 0 1 3 4 5 6; do
  while ! grep -qaE "完成：|Traceback" data/collect_${R}_$i.log 2>/dev/null; do sleep 30; done
done
grep -ah -E "可复现性|完成：|各种救法" data/collect_${R}_[013456].log | cut -c1-260
echo "采集阶段结束，用时 $(( $(date +%s) - T0 )) 秒"
while ! grep -qa "DAGGER3B DONE" ~/logs/dagger3b.log; do sleep 30; done
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast \
  data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 data/raw_${R}_6 data/vla_$R 2>&1 | tail -1
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
