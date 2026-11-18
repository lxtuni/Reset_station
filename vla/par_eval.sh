#!/bin/bash
# 服务器上：跑多组评测。配置文件每行一组，格式 “名字|检查点|eval_vla.py 的参数”。
# 用法：bash par_eval.sh <配置文件，在 ~/vla 里> [同时跑几组，默认 4]
# 服务器是多人共用的：每个评测进程只用 2 个 CPU 线程，同时最多跑 4 组，并用 nice 降低优先级。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd ~/Reset_station; mkdir -p outputs/ev
P=${2:-4}
run() {
  IFS='|' read -r NAME C ARGS <<< "$1"
  [ -f outputs/ev/$NAME.json ] && return                 # 已有结果就跳过，方便中断后续跑
  nice -n 10 python vla/eval_vla.py outputs/${C%/*}/checkpoints/${C#*/}/pretrained_model $ARGS --json outputs/ev/$NAME.json \
    > outputs/ev/$NAME.log 2>&1
}
while read -r line; do
  [ -z "$line" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  run "$line" &
done < ~/vla/$1
wait
while read -r line; do
  [ -z "$line" ] && continue
  NAME=${line%%|*}
  echo "=== $NAME ==="; grep -E "cm/s|合计|时间分解" outputs/ev/$NAME.log | cut -c1-200
  grep -E "Error:" outputs/ev/$NAME.log | grep -v "downloading" | tail -2
done < ~/vla/$1
echo "PAR EVAL DONE"
