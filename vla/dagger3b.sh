#!/bin/bash
# 服务器上：第三轮 DAgger 的后半段（数据已经合并好）。上一次训练因为和方案 F 抢显存而中断，这里等方案 F 全部结束后再训练和评测。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/Reset_station
M=outputs/vla_d2/checkpoints/005000/pretrained_model; R=d3
while ! grep -qa "PRIM RUN DONE" ~/logs/prim.log; do sleep 30; done
sleep 20
T0=$(date +%s)
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl
bash ~/vla/train_vla.sh vla_$R 5000 2500 64 $HOME/Reset_station/$M vla_$R 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -a -E "step:[0-9K]+ |End of training|Traceback|Error" | grep --line-buffered -a -vE "downloading" | stdbuf -oL cut -c1-140 | awk 'NR % 20 == 0 || /End|Trace|Error/'
echo "训练结束，用时 $(( $(date +%s) - T0 )) 秒"
export PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cat > ~/vla/cfg_$R.txt <<EOF
${R}_2500|vla_$R/002500|--bins --exec_steps 1 --num_steps 3
${R}_5000|vla_$R/005000|--bins --exec_steps 1 --num_steps 3
${R}_5000_b|vla_$R/005000|--bins --exec_steps 1 --num_steps 3 --seed0 1000
${R}_2500_b|vla_$R/002500|--bins --exec_steps 1 --num_steps 3 --seed0 1000
EOF
rm -f outputs/ev/${R}_*.json outputs/ev/${R}_*.log
bash ~/vla/par_eval.sh cfg_$R.txt 4 > /dev/null
for f in ${R}_2500 ${R}_5000 ${R}_2500_b ${R}_5000_b; do echo "== $f"; grep -a -E "cm/s|合计" outputs/ev/$f.log | cut -c1-110; done
echo "DAGGER3B DONE"
