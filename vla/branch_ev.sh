#!/bin/bash
# 服务器上：分叉搜索训练出的模型的补充评测。
# 检验一个假设：脱身动作需要连续执行若干步，而评测时每一步都重新规划（exec_steps=1），学到的脱身动作可能被打断。
# 所以把“每次规划后连续执行 8 步”也测一下，新模型和原模型各测两组种子。
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com MUJOCO_GL=egl PYTHONPATH=act:vla HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
cd ~/Reset_station
echo "--- 已有评测的分速度结果 ---"
for f in br1_2500 br1_5000 br1_5000_b; do echo "== $f"; grep -a -E "cm/s|合计" outputs/ev/$f.log | cut -c1-90; done
ls outputs/ev | grep -a -E "d2|d3" | tr '\n' ' '; echo
cat > ~/vla/cfg_brx.txt <<EOF
br1_5000_e8|vla_br1/005000|--bins --exec_steps 8 --num_steps 3
br1_5000_e8_b|vla_br1/005000|--bins --exec_steps 8 --num_steps 3 --seed0 1000
d2_5000_e8|vla_d2/005000|--bins --exec_steps 8 --num_steps 3
d2_5000_e8_b|vla_d2/005000|--bins --exec_steps 8 --num_steps 3 --seed0 1000
EOF
bash ~/vla/par_eval.sh cfg_brx.txt 4 > /dev/null
for f in br1_5000_e8 br1_5000_e8_b d2_5000_e8 d2_5000_e8_b; do echo "== $f"; grep -a -E "cm/s|合计" outputs/ev/$f.log | cut -c1-90; done
echo "BRANCH EV DONE"
