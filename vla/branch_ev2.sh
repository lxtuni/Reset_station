#!/bin/bash
# 服务器上：新模型和原模型在“每次规划后连续执行 2 步 / 4 步”下的成功率（各两组种子），
# 和已有的 1 步、8 步连起来，看两者的差距是否随连续执行的步数变大。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
: > ~/vla/cfg_bry.txt
for e in 2 4; do for m in br1 d2; do
  echo "${m}_5000_e$e|vla_$m/005000|--bins --exec_steps $e --num_steps 3" >> ~/vla/cfg_bry.txt
  echo "${m}_5000_e${e}_b|vla_$m/005000|--bins --exec_steps $e --num_steps 3 --seed0 1000" >> ~/vla/cfg_bry.txt
done; done
bash ~/vla/par_eval.sh cfg_bry.txt 4 > /dev/null
for e in 2 4; do for m in br1 d2; do for s in "" _b; do f=${m}_5000_e$e$s; echo "$f $(grep -a 合计 outputs/ev/$f.log | cut -c1-36)"; done; done; done
echo "BRANCH EV2 DONE"
