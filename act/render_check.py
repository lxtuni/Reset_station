"""让老师跑一个回合，每 4 步把两路图像拼在一起存成 PNG，用来检查相机画面。

用法（仓库根目录）：python act/render_check.py [输出文件夹] [种子]
"""
import os, sys
import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from common import make_env, Cams, TEACHER

out = sys.argv[1] if len(sys.argv) > 1 else "act_preview"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
os.makedirs(out, exist_ok=True)

env = make_env(v_range=(0.10, 0.20)); cams = Cams(env); teacher = SAC.load(TEACHER)
obs, _ = env.reset(seed=seed); t = 0
while True:
    if t % 4 == 0:
        top, wrist = cams()
        Image.fromarray(np.concatenate([top, wrist], 1)).save(f"{out}/step_{t:03d}.png")
    obs, r, term, trunc, info = env.step(teacher.predict(obs, deterministic=True)[0])
    t += 1
    if term or trunc:
        break
top, wrist = cams(); Image.fromarray(np.concatenate([top, wrist], 1)).save(f"{out}/step_{t:03d}_end.png")
print(f"回合结束：{t} 步，成功={info['success']}，船速 v0={info['v0']:.3f} m/s，图像存在 {out}/")
