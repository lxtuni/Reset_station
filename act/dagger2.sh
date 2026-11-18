#!/bin/bash
# ACT 的第二轮 DAgger，偏向快船（失败最集中的地方）。在本机 WSL 的仓库根目录运行。
#   1. DAgger 采集 160 回合：1 个进程慢船（0–10 cm/s），3 个进程快船（10–20 cm/s）
#   2. 另外补 100 个快船的教师示教
#   3. 和之前的全部数据（400 示教 + 第一轮 160）合并，从第一轮 DAgger 的模型接着训练 2 万步
#   4. 评测两个检查点（每步都重新推理）
# 用法：bash act/dagger2.sh <起点模型目录>
M=${1:-outputs/act_d1/checkpoints/030000/pretrained_model}
T0=$(date +%s)
VMIN=(0.0 0.10 0.10 0.10); VMAX=(0.10 0.20 0.20 0.20)
for i in 0 1 2 3; do
  python act/collect_dagger.py $M --episodes 40 --seed0 $((50000 + i * 1000)) --v_min ${VMIN[$i]} --v_max ${VMAX[$i]} \
    --out data/raw_d2_$i > outputs/dagger_d2_$i.log 2>&1 &
done
python act/collect_demos.py --v_min 0.10 --v_max 0.20 --episodes 100 --seed0 60000 --out data/raw_fast > outputs/collect_fast.log 2>&1 &
wait
grep -h "完成" outputs/dagger_d2_*.log outputs/collect_fast.log; grep -h "40/40" outputs/dagger_d2_*.log | cut -c1-60
echo "采集用时 $(( $(date +%s) - T0 )) 秒"
bash act/dagger_train.sh $M lerobot_d2 act_d2 20000 data/raw_static data/raw_full_0 data/raw_full_1 data/raw_full_2 data/raw_full_3 \
  data/raw_full_4 data/raw_full_5 data/raw_d1_0 data/raw_d1_1 data/raw_d1_2 data/raw_d1_3 \
  data/raw_d2_0 data/raw_d2_1 data/raw_d2_2 data/raw_d2_3 data/raw_fast > outputs_train_d2.log 2>&1
grep -E "End of training" outputs_train_d2.log | tail -1 | cut -c1-80
echo "训练结束，总用时 $(( $(date +%s) - T0 )) 秒"
bash act/eval_ckpts.sh act_d2 1 010000 020000
echo "ACT DAGGER2 DONE，总用时 $(( $(date +%s) - T0 )) 秒"
