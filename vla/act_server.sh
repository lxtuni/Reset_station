#!/bin/bash
# 服务器上：在 4090 上训练一个 ACT，供量化对比用。
# 原计划是把本机训练好的 ACT 传上来，但上行只有约 0.01 MB/s，197 MB 要传几个小时，所以改为在服务器上直接训练。
# 数据用服务器上现成的 vla_d2（820 回合：示教 500 + SmolVLA 两轮 DAgger 的 320）。ACT 不看任务描述，相机名无所谓。
# 训练完评测每个检查点（每步重新推理，种子 0–29），供挑选。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1
cd ~/Reset_station
cp vla/eval_act.py act/eval_act.py; cp vla/export_onnx.py act/export_onnx.py
rm -rf outputs/act_srv
T0=$(date +%s)
lerobot-train --dataset.repo_id=local/vla_d2 --dataset.root=$HOME/Reset_station/data/vla_d2 \
  --policy.type=act --policy.chunk_size=20 --policy.n_action_steps=5 --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --num_workers=8 --steps=40000 --save_freq=10000 --log_freq=1000 \
  --output_dir=$HOME/Reset_station/outputs/act_srv 2>&1 | tr '\r' '\n' | grep -E "step:[0-9K]+ |End of training|Traceback|Error" | cut -c1-120 | awk 'NR % 5 == 0 || /End|Trace|Error/'
echo "训练用时 $(( $(date +%s) - T0 )) 秒"
export OMP_NUM_THREADS=2
for C in 010000 020000 030000 040000; do
  nice -n 10 python act/eval_act.py outputs/act_srv/checkpoints/$C/pretrained_model --bins --episodes 30 --exec_steps 1 \
    --json outputs/ev/actsrv_$C.json > outputs/ev/actsrv_$C.log 2>&1 &
done
wait
for C in 010000 020000 030000 040000; do echo "=== act_srv $C ==="; grep -E "cm/s|合计|时间分解" outputs/ev/actsrv_$C.log | cut -c1-170; grep "Error:" outputs/ev/actsrv_$C.log | tail -1 | cut -c1-200; done
echo "ACT SRV DONE"
