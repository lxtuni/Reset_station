"""回放检查：用同一个种子重置环境，不用老师，直接按记录的动作执行，回合应该照样成功。

用法（仓库根目录）：python act/replay_check.py data/raw_static [回合数]
"""
import glob, sys
import numpy as np
from common import make_env

files = sorted(glob.glob(f"{sys.argv[1]}/ep_*.npz"))[: int(sys.argv[2]) if len(sys.argv) > 2 else 10]
env = make_env(); ok = 0
for f in files:
    d = np.load(f)
    env.v_range = (0.0, 0.0) if float(d["v0"]) == 0.0 else env.v_range
    obs, _ = env.reset(seed=int(d["seed"]))
    for a in d["action"]:
        obs, r, term, trunc, info = env.step(a.copy())
        if term or trunc:
            break
    ok += bool(info["success"])
    print(f"{f}: 重放 {'成功' if info['success'] else '失败'}（记录 {len(d['action'])} 步，重放 {env.steps} 步）")
print(f"{ok}/{len(files)} 个回合重放成功")
