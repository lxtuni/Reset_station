#!/bin/bash
# 服务器上：自适应执行的评测。平时每步重新推理；预测的动作块开头明显上抬时，连续执行 8 步。
# 新模型（分叉搜索后）和原模型各测两个门槛、两组种子。对照：每步重新推理时 313/360 和 314/360。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
cd ~/Reset_station
: > ~/vla/cfg_cm.txt
for z in 0.3 0.5; do for m in br1 d2; do
  echo "${m}_cm$z|vla_$m/005000|--bins --exec_steps 1 --num_steps 3 --commit 8 --commit_z $z" >> ~/vla/cfg_cm.txt
  echo "${m}_cm${z}_b|vla_$m/005000|--bins --exec_steps 1 --num_steps 3 --commit 8 --commit_z $z --seed0 1000" >> ~/vla/cfg_cm.txt
done; done
bash ~/vla/par_eval.sh cfg_cm.txt 4 > /dev/null
for z in 0.3 0.5; do for m in br1 d2; do for s in "" _b; do f=${m}_cm$z$s
  echo "$f $(grep -a 合计 outputs/ev/$f.log | cut -c1-36) | $(grep -a 自适应 outputs/ev/$f.log | cut -c20-70)"; grep -a "Error" outputs/ev/$f.log | tail -n 1; done; done; done
echo "COMMIT EV DONE"
