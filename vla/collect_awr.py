"""方案 B 的数据采集：SmolVLA 带探索自己跑，成功和失败的回合全部留下，并记下每一步的奖励。

和 collect_self.py 的探索方式相同（一半回合加连续的小扰动；到第 60 步还没夹住就朝随机方向推 12 步），区别是：
  - 不按成败筛选回合。每一步动作好不好，交给后面的“优势”来判断（awr_weights.py）
  - 多存两样东西：每一步的环境奖励，以及仿真器里的真实状态（只给打分网络当输入，SmolVLA 永远看不到）
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/collect_awr.py MODEL_DIR ENGINE OUT_DIR 回合数 种子起点
"""
import os, sys, time
import numpy as np
from resid_env import ResidEnv

M, ENG, OUT, N, S0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
env = ResidEnv(M, ENG, alpha=1.0, fast_bias=0.5, pen=0.0)   # pen=0：奖励就是环境原本的奖励
rng = np.random.default_rng(S0)
frames = n_ok = n_stuck = n_rescued = 0
t0 = time.time()
for ep in range(N):
    env.reset(seed=S0 + ep)
    noisy = rng.random() < 0.5
    ou = np.zeros(3); kick, kick_left, cool, stuck = np.zeros(3), 0, 0, False
    tops, wrists, states, actions, rews, sobs, kicked, grasp_seen = [], [], [], [], [], [], [], False
    while True:
        delta = np.zeros(5, np.float32)
        if noisy:
            ou += 0.2 * (0 - ou) + 0.12 * rng.standard_normal(3); delta[:3] += ou
        t = len(actions)
        if kick_left == 0 and cool == 0 and t >= 60 and not grasp_seen:
            d = rng.standard_normal(3); d[2] = abs(d[2]) * 0.5
            kick = d / np.linalg.norm(d); kick_left = 12; stuck = True
        in_kick = kick_left > 0
        if in_kick:
            delta[:3] = kick - env.a_vla[:3]
            kick_left -= 1
            if kick_left == 0:
                cool = 30
        elif cool > 0:
            cool -= 1
        tops.append(env.last_top); wrists.append(env.last_wrist); states.append(env.last_q)
        sobs.append(np.asarray(env.last_env_obs, np.float32).copy()); kicked.append(in_kick)
        obs, r, term, trunc, info = env.step(delta)
        actions.append(env.prev_a.copy()); rews.append(r)
        grasp_seen = bool(env.last_info["grasped"])
        if term or trunc:
            break
    succ = bool(info["success"])
    n_ok += int(succ); n_stuck += int(stuck); n_rescued += int(stuck and succ)
    np.savez_compressed(f"{OUT}/ep_{ep:04d}.npz", top=np.array(tops), wrist=np.array(wrists), state=np.array(states),
                        action=np.array(actions, np.float32), reward=np.array(rews, np.float32), sobs=np.array(sobs),
                        kicked=np.array(kicked), success=succ, seed=S0 + ep, v0=env.last_info["v0"], stuck=stuck, noisy=noisy)
    frames += len(actions)
    if (ep + 1) % 50 == 0:
        print(f"{ep + 1}/{N} 回合：成功 {n_ok}，共 {frames} 帧 | 到第 60 步还没夹住的 {n_stuck} 个，推一把之后成功 {n_rescued} 个 | "
              f"已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{N} 回合，成功 {n_ok}，{frames} 帧，存在 {OUT}")
