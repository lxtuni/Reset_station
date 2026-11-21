#!/bin/bash
# 服务器上：方案 B——用奖励直接更新 SmolVLA 的动作专家（优势加权回归）。
#   1. 当前策略带探索自己跑 6×200 回合，成功失败都留下，记下每一步的奖励
#   2. 用奖励算出每一帧的优势和权重（awr_weights.py）
#   3. 和原有的 820 回合合并（原有数据权重为 1，自采数据的总权重和它相等），按权重抽样接着训练 5000 步
#   4. 评测（对照：314/360）
# 用法：bash awr_run.sh <轮次名，如 b1> <种子起点>
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
R=$1; S=$2
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine
BASE="data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast"
T0=$(date +%s)
[ "$3" = skip ] || [ "$3" = train ] || for i in 0 1 2 3 4 5; do            # 第 3 个参数是 skip 时跳过采集（后面的步骤出错重跑时用）
  rm -rf data/raw_${R}_$i
  nice -n 10 python vla/collect_awr.py $M $E data/raw_${R}_$i 200 $((S + i * 1000)) > data/collect_${R}_$i.log 2>&1 &
done
wait
grep -ah -E "完成|Traceback|Error" data/collect_${R}_*.log | cut -c1-200
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
if [ "$3" != train ]; then                              # 第 3 个参数是 train 时数据已经合并好，直接训练
NB=$(python -c "import json; print(json.load(open('data/vla_d2/meta/info.json'))['total_frames'])")
python vla/awr_weights.py $NB data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_2 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 2>&1 | cut -c1-260
python vla/convert_vla.py $BASE data/raw_${R}_0 data/raw_${R}_1 data/raw_${R}_2 data/raw_${R}_3 data/raw_${R}_4 data/raw_${R}_5 data/vla_$R 2>&1 | tail -n 2
fi
[ -f data/vla_$R/awr_weights.npy ] || { echo "合并数据失败，停止"; exit 1; }
echo "数据就绪，用时 $(( $(date +%s) - T0 )) 秒"
unset HF_HUB_OFFLINE OMP_NUM_THREADS
rm -rf outputs/vla_$R
AWR_WEIGHTS=data/vla_$R/awr_weights.npy python vla/train_awr.py \
  --policy.path=$HOME/Reset_station/$M \
  --dataset.repo_id=local/vla_$R --dataset.root=$HOME/Reset_station/data/vla_$R \
  --policy.device=cuda --policy.push_to_hub=false --policy.scheduler_decay_steps=5000 \
  --batch_size=64 --num_workers=12 --steps=5000 --save_freq=2500 --log_freq=50 \
  --output_dir=$HOME/Reset_station/outputs/vla_$R 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -a -E "step:[0-9K]+ |End of training|Traceback|Error|按权重抽样|对不上" | grep --line-buffered -a -vE "downloading" | stdbuf -oL cut -c1-140 | awk 'NR % 20 == 1 || /End|Trace|Error|权重|对不上/'
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
echo "AWR DONE，总用时 $(( $(date +%s) - T0 )) 秒"
