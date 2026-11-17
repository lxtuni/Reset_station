"""让老师（stage9 SAC）在拦截环境里跑，把成功的回合存成示教数据。

每个回合一个 npz：top / wrist 图像 (T,240,320,3) uint8，state (T,6) 关节角度，action (T,5) 老师的原始输出。
用法（仓库根目录）：
    python act/collect_demos.py --v_max 0.0  --episodes 100 --out data/raw_static
    python act/collect_demos.py --v_max 0.20 --episodes 300 --out data/raw_full
"""
import argparse, os, time
import numpy as np
from stable_baselines3 import SAC
from common import make_env, Cams, joints, TEACHER

ap = argparse.ArgumentParser()
ap.add_argument("--v_min", type=float, default=0.0)
ap.add_argument("--v_max", type=float, default=0.20)
ap.add_argument("--episodes", type=int, default=100)      # 要采集的成功回合数
ap.add_argument("--seed0", type=int, default=10000)       # 和评测用的种子（0–29、1000 起）错开
ap.add_argument("--out", default="data/raw_full")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

env = make_env(v_range=(args.v_min, args.v_max)); cams = Cams(env); teacher = SAC.load(TEACHER)
saved, tried, frames_total, t0 = 0, 0, 0, time.time()
while saved < args.episodes:
    seed = args.seed0 + tried; tried += 1
    obs, _ = env.reset(seed=seed)
    tops, wrists, states, actions = [], [], [], []
    while True:
        top, wrist = cams()                                # 先拍照，再走一步：图像和动作对应同一时刻
        a = teacher.predict(obs, deterministic=True)[0]
        tops.append(top); wrists.append(wrist); states.append(joints(env))
        actions.append(np.asarray(a, np.float32).copy())   # 记录滤波前的原始动作（env.step 里还会做 0.8 的低通滤波）
        obs, r, term, trunc, info = env.step(np.array(a, dtype=np.float32))
        if term or trunc:
            break
    if not info["success"]:
        continue                                           # 只保留成功的回合
    np.savez_compressed(f"{args.out}/ep_{saved:04d}.npz", top=np.array(tops), wrist=np.array(wrists),
                        state=np.array(states), action=np.array(actions), seed=seed, v0=info["v0"])
    saved += 1; frames_total += len(actions)
    if saved % 10 == 0:
        print(f"{saved}/{args.episodes} 个成功回合（老师成功率 {saved / tried * 100:.0f}%），"
              f"{frames_total} 帧，已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{saved} 个成功回合 / 尝试 {tried} 个，共 {frames_total} 帧，存在 {args.out}")
