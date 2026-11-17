#!/bin/bash
# 并行评测一次训练的多个检查点（6 档速度 × 30 回合）。用法（仓库根目录）：bash act/eval_ckpts.sh <输出名> <exec_steps> <检查点...>
OUT=$1; E=$2; shift 2
for C in "$@"; do
  python act/eval_act.py outputs/$OUT/checkpoints/$C/pretrained_model --bins --episodes 30 --exec_steps $E \
    --json outputs/eval_${OUT}_${C}_e$E.json > outputs/eval_${OUT}_${C}_e$E.log 2>&1 &
done
wait
for C in "$@"; do echo "=== $OUT $C exec_steps=$E ==="; grep -E 'cm/s|合计|时间分解' outputs/eval_${OUT}_${C}_e$E.log; done
