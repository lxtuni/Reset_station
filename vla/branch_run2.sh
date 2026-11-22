#!/bin/bash
# 服务器上：接管分叉搜索这一轮的后续流程。
# 情况：第一批（6×200 回合）按进度只能得到约 115 段，偏少。这里等第一批采完后再采第二批（6×200 回合，种子不同），
# 两批一起和原有的 820 回合合并。分叉片段的抽样权重设为 2（原有数据为 1），让它在训练里占到约三分之一。
# 然后接着训练 5000 步、评测（对照：314/360；第三轮 DAgger：334/360）。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
pkill -u $USER -f "[v]la/branch_run.sh" 2>/dev/null       # 只停掉旧的调度脚本，正在跑的采集进程不受影响
R=br1; W=2.0
M=outputs/vla_d2/checkpoints/005000/pretrained_model; E=outputs/export_d2/smolvla_n3.onnx.fp32.engine
BASE="data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 data/raw_full_4 data/raw_full_5
  data/raw_vd1_0 data/raw_vd1_1 data/raw_vd1_2 data/raw_vd1_3 data/raw_vd2_0 data/raw_vd2_1 data/raw_vd2_2 data/raw_vd2_3 data/raw_fast"
T0=$(date +%s)
for i in 0 1 2 3 4 5; do
  while ! grep -qaE "完成：|Traceback \(most" data/collect_br1_$i.log 2>/dev/null; do sleep 30; done
done
echo "第一批采完，用时 $(( $(date +%s) - T0 )) 秒"
for i in 0 1 2 3 4 5; do
  rm -rf data/raw_br2_$i
  nice -n 10 python vla/collect_branch.py $M $E data/raw_br2_$i 200 $((900000 + i * 10000)) > data/collect_br2_$i.log 2>&1 &
done
wait
grep -ah "可复现" data/collect_br[12]_*.log | sort | uniq -c
for b in br1 br2; do for i in 0 1 2 3 4 5; do grep -a -E "回合：|各候选|搜索层数" data/collect_${b}_$i.log | tail -n 3 | cut -c1-330; done; done
echo "采集结束，用时 $(( $(date +%s) - T0 )) 秒"
BR=$(ls -d data/raw_br1_[0-5] data/raw_br2_[0-5])
python - $W $BR <<'EOF'
import glob, sys, numpy as np
w, n, fr = float(sys.argv[1]), 0, 0
for d in sys.argv[2:]:
    for f in sorted(glob.glob(f"{d}/ep_*.npz")):
        T = len(np.load(f)["action"]); np.save(f[:-4] + ".w.npy", np.full(T, w, np.float32)); n += 1; fr += T
print(f"分叉片段共 {n} 段，{fr} 帧，抽样权重 {w}")
EOF
python vla/convert_vla.py $BASE $BR data/vla_$R 2>&1 | tail -n 1
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
for f in ${R}_2500 ${R}_2500_b ${R}_5000 ${R}_5000_b; do echo "== $f $(grep -a 合计 outputs/ev/$f.log | cut -c1-40)"; done
echo "BRANCH DONE，总用时 $(( $(date +%s) - T0 )) 秒"
