#!/bin/bash
# 并行采集 DAgger 数据：4 个进程 × 40 回合。用法（仓库根目录）：bash act/dagger_collect.sh MODEL_DIR [轮次名，默认 d1] [种子起点]
M=$1; R=${2:-d1}; S=${3:-40000}
for i in 0 1 2 3; do
  python act/collect_dagger.py $M --episodes 40 --seed0 $((S + i * 1000)) --out data/raw_${R}_$i \
    > outputs/dagger_${R}_$i.log 2>&1 &
done
wait
grep -h "完成" outputs/dagger_${R}_*.log; grep -h "40/40" outputs/dagger_${R}_*.log
