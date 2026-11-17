#!/bin/bash
# 全速度范围（0–20 cm/s）的示教：6 个进程并行，各采 50 个成功回合，种子互不重叠。
cd ~/Reset_station
for i in 0 1 2 3 4 5; do
  python act/collect_demos.py --v_max 0.20 --episodes 50 --seed0 $((20000 + i * 5000)) --out data/raw_full_$i \
    > data/collect_full_$i.log 2>&1 &
done
wait
grep -h "完成" data/collect_full_*.log
echo "COLLECT DONE"
