#!/bin/bash
# 香橙派上：停掉旧的推理服务，启动一个新的。用法：bash start_serve.sh <方案> <名字> <端口>
# 单独放在一个脚本里：这样“停掉旧服务”的匹配不会误伤正在执行启动命令的 shell。
cd ~/act_opi
pkill -f "python serve_opi.py" 2>/dev/null; sleep 1
setsid nohup ./venv/bin/python serve_opi.py model $HOME/act_opi/lib/librknnrt.so $1 $3 > log_$2.txt 2>&1 < /dev/null &
for i in $(seq 1 20); do grep -q "推理服务就绪" log_$2.txt 2>/dev/null && break; sleep 1; done
grep -v device_discovery log_$2.txt | tail -2
