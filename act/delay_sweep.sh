#!/bin/bash
# 延迟注入实验：快船（10–20 cm/s）上，人为加 0/50/100/150 ms 推理延迟，比较同步和异步执行。
# 用法（仓库根目录）：bash act/delay_sweep.sh MODEL_DIR [回合数]    结果追加到 outputs/delay_sweep.csv
M=$1; N=${2:-60}
rm -f outputs/delay_sweep.csv
for D in 0 50 100 150; do
  for A in "" "--async_exec"; do
    [ $D = 0 ] && [ -n "$A" ] && continue          # 没有延迟时同步和异步是一回事
    python act/eval_act.py $M --v_min 0.10 --v_max 0.20 --episodes $N --delay_ms $D $A --exec_steps 4 \
      --csv outputs/delay_sweep.csv 2>&1 | grep -E "^合计"
  done
done
echo "backend,delay_ms,exec_steps,async,success,walls,infer_ms,episodes"; cat outputs/delay_sweep.csv
