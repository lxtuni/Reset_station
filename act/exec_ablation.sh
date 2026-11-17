#!/bin/bash
# 消融：每次推理后执行几步（exec_steps）对成功率的影响。船在动（3–20 cm/s）。
# 用法（仓库根目录）：bash act/exec_ablation.sh MODEL_DIR [回合数] [exec 列表]
M=$1; N=${2:-30}; LIST=${3:-"1 2 5 10 20"}
for E in $LIST; do
  echo "--- exec_steps=$E ---"
  python act/eval_act.py $M --v_min 0.03 --v_max 0.20 --episodes $N --exec_steps $E 2>&1 | grep -E "cm/s|时间分解"
done
