"""强化学习后训练的数据采集：SmolVLA 自己开（不问教师），加一点探索，只留下成功的回合。

思路（按成功与否筛选的自我模仿，是最简单的一种离线强化学习）：
  - 让当前策略带着随机扰动去做任务，有的回合成功、有的失败
  - 只把成功回合里“实际执行的动作”当作新的训练数据：等于告诉模型“这样做过，结果是好的，以后多这样做”
  - 没有教师参与。唯一的信号是环境给的结果（成功 / 失败），这就是强化学习和模仿学习的区别
失败分析的结论是：76% 的失败是第一次没夹住之后停在 8–10 cm 外不再尝试。所以探索专门针对这一点：
  - 一半的回合全程加小幅的、时间上连续的扰动（环境会平滑动作，逐步独立的噪声会被滤掉）
  - 任何回合如果到第 60 步还没夹住，就给一次“推一把”：朝一个随机方向连续移动 12 步，再交还给 SmolVLA。
    推完之后成功了的回合留下，其中就包含了“卡住之后怎么脱身并重新靠近”的例子
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/collect_self.py MODEL_DIR ENGINE OUT_DIR 回合数 种子起点
"""
import os, sys, time
import numpy as np
from resid_env import ResidEnv

M, ENG, OUT, N, S0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
env = ResidEnv(M, ENG, alpha=1.0, fast_bias=0.5)            # alpha=1：delta 直接加在 SmolVLA 的动作上
rng = np.random.default_rng(S0)
saved = frames = n_stuck = n_rescued = n_plain = ok_plain = 0
t0 = time.time()
for ep in range(N):
    env.reset(seed=S0 + ep)
    noisy = rng.random() < 0.5
    ou = np.zeros(3); kick, kick_left, cool, stuck = np.zeros(3), 0, 0, False
    tops, wrists, states, actions, grasp_seen = [], [], [], [], False
    while True:
        delta = np.zeros(5, np.float32)
        if noisy:                                            # 时间上连续的小扰动（OU 噪声），只加在末端位移上
            ou += 0.2 * (0 - ou) + 0.12 * rng.standard_normal(3); delta[:3] += ou
        t = len(actions)
        if kick_left == 0 and cool == 0 and t >= 60 and not grasp_seen:
            d = rng.standard_normal(3); d[2] = abs(d[2]) * 0.5   # 随机方向，略微偏向上抬
            kick = d / np.linalg.norm(d); kick_left = 12; stuck = True
        if kick_left > 0:
            delta[:3] = kick - env.a_vla[:3]                  # 这几步完全按“推一把”的方向走
            kick_left -= 1
            if kick_left == 0:
                cool = 30
        elif cool > 0:
            cool -= 1
        tops.append(env.last_top); wrists.append(env.last_wrist); states.append(env.last_q)
        obs, r, term, trunc, info = env.step(delta)
        actions.append(env.prev_a.copy())                    # 存的是实际执行的动作
        grasp_seen = bool(env.last_info["grasped"])
        if term or trunc:
            break
    succ = bool(info["success"])
    if stuck:
        n_stuck += 1; n_rescued += int(succ)
    elif not noisy:
        n_plain += 1; ok_plain += int(succ)
    if succ:
        np.savez_compressed(f"{OUT}/ep_{saved:04d}.npz", top=np.array(tops), wrist=np.array(wrists), state=np.array(states),
                            action=np.array(actions, np.float32), seed=S0 + ep, v0=env.last_info["v0"], stuck=stuck, noisy=noisy)
        saved += 1; frames += len(actions)
    if (ep + 1) % 50 == 0:
        print(f"{ep + 1}/{N} 回合：留下 {saved} 个成功回合（{frames} 帧）| 到第 60 步还没夹住的 {n_stuck} 个，推一把之后成功 {n_rescued} 个 | "
              f"不加任何扰动的 {n_plain} 个里成功 {ok_plain} 个 | 已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{N} 回合，留下 {saved} 个成功回合，{frames} 帧，存在 {OUT}")
