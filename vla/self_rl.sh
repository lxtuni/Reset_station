#!/bin/bash
# 服务器上：用“按成功筛选的自我模仿”对 SmolVLA 的动作专家做一轮强化学习后训练。
#   1. 当前策略带探索自己跑 6×200 回合，只留成功的
#   2. 和已有的 820 回合合并，从当前模型接着训练 5000 步
#   3. 评测（PyTorch，去噪 3 次，每步推理；对照：157/180 和 157/180）
# 用法：bash self_rl.sh <起点检查点，如 vla_d2/005000> <轮次名，如 s1> <种子起点> [上一轮的自采数据目录前缀...]
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
C=$1; R=$2; S=$3; shift 3
M=outputs/${C%/*}/checkpoints/${C#*/}/pretrained_model
E=outputs/export_$R/smolvla_n3.onnx.fp32.engine
T0=$(date +%s)
if [ ! -f $E ]; then                                        # 这一轮的起点模型还没有引擎：导出并编译一个（采集时用引擎，快一倍）
  [ "$C" = "vla_d2/005000" ] && { mkdir -p outputs/export_$R; ln -sf $HOME/Reset_station/outputs/export_d2/smolvla_n3.onnx.fp32.engine $E; } || {
    bash ~/vla/run_py.sh vla/export_vla.py $M outputs/export_$R 3 | grep -E "导出"
    conda activate ortgpu; SP=$(python -c "import site; print(site.getsitepackages()[0])")
    LD_LIBRARY_PATH=~/vla/cudalib:$(ls -d $SP/nvidia/*/lib 2>/dev/null | tr '\n' ':') python vla/ort_trt_bench.py outputs/export_$R smolvla_n3.onnx 2>&1 | grep -E "编译"
    conda activate lerobot; }
fi
for i in 0 1 2 3 4 5; do
  nice -n 10 python vla/collect_self.py $M $E data/raw_${R}_$i 200 $((S + i * 1000)) > data/collect_${R}_$i.log 2>&1 &
done
wait
grep -h "200/200" data/collect_${R}_*.log | cut -c1-200
grep -h "完成" data/collect_${R}_*.log
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
python vla/convert_vla.py data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5 \
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast \
  "$@" data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_2 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 data/vla_$R 2>&1 | tail -1
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
bash ~/vla/par_eval.sh cfg_$R.txt 4 | grep -E "===|合计"
echo "SELF RL DONE，总用时 $(( $(date +%s) - T0 )) 秒"
