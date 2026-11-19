"""强化学习后训练的第五种做法：用仿真器做搜索，把搜到的更好的做法教回给模型（搜索加蒸馏）。

前四次都没成，原因归结为两点：模型卡住时缺一种没见过的动作；随机探索很难碰到它，碰到了也被平均掉。
这次利用仿真器能“重来一遍”的特点，把探索变成有目的的搜索：
  1. 让 SmolVLA 正常跑一个回合（去噪起点的随机数由种子决定，所以同一个种子可以原样重放）
  2. 如果成功，什么都不做
  3. 如果失败，就用同一个种子重放，在某个时刻插入一个脱身动作（上抬 / 后退并上抬 / 向左并上抬 / 向右并上抬），
     其余时间仍然是 SmolVLA。按固定顺序逐个尝试，第一个让回合变成功的就留下
  4. 留下的回合就是“在这个失败情形下，这样做能成功”的确切例子，之后用来训练动作头
没有教师参与。唯一的信号是回合成功与否，但它是通过“同一情形下换一种做法的结果对比”得到的，比单纯筛选成功回合有针对性得多。
和方案 F 的区别：F 让小网络自己猜什么时候该插入；这里由搜索直接找到答案，再让 SmolVLA 自己从图像里学会。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/collect_search.py MODEL_DIR ENGINE OUT_DIR 回合数 种子起点
"""
import os, sys, time
import numpy as np
from resid_env import ResidEnv

M, ENG, OUT, N, S0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
env = ResidEnv(M, ENG, alpha=1.0, fast_bias=0.7)
PRIMS = {1: (0.0, 0.0, 1.0), 2: (-1.0, 0.0, 0.6), 3: (0.0, 1.0, 0.4), 4: (0.0, -1.0, 0.4)}
NAMES = {1: "上抬", 2: "后退并上抬", 3: "向左并上抬", 4: "向右并上抬"}
DUR = 12
CANDS = [(60, 1), (60, 2), (60, 3), (60, 4), (90, 1), (90, 2), (90, 3), (90, 4), (45, 1), (120, 1)]   # (第几步插入, 哪个动作)，按这个顺序试


def rollout(seed, interv=None, keep=False):
    rng = np.random.default_rng(seed * 7919 + 13)
    env.noise_fn = lambda: rng.standard_normal((env.chunk, env.adim)).astype(np.float32)
    env.reset(seed=seed)
    tops, wrists, states, actions = [], [], [], []
    while True:
        t = len(actions)
        delta = np.zeros(5, np.float32)
        if interv is not None and interv[0] <= t < interv[0] + DUR:
            delta = np.array([*PRIMS[interv[1]], 0.0, 1.0], np.float32) - env.a_vla
        if keep:
            tops.append(env.last_top); wrists.append(env.last_wrist); states.append(env.last_q)
        _, r, term, trunc, info = env.step(delta)
        actions.append(env.prev_a.copy())
        if term or trunc:
            break
    return bool(info["success"]), len(actions), (tops, wrists, states, actions), float(env.last_info["v0"])


# 先确认可复现：同一个种子跑两遍，结果要一样，否则“重放后插入动作”的对比没有意义
same = 0
for s in range(S0, S0 + 4):
    a = rollout(s); b = rollout(s); same += int(a[0] == b[0] and a[1] == b[1])
print(f"可复现性检查：4 个种子各跑两遍，结果完全相同的有 {same} 个", flush=True)

n_ok = n_fail = n_resc = frames = tries = 0
used = {}
t0 = time.time()
for ep in range(N):
    seed = S0 + ep
    ok, steps, _, v0 = rollout(seed)
    if ok:
        n_ok += 1
    else:
        n_fail += 1
        for (t_int, k) in CANDS:
            tries += 1
            ok2, steps2, traj, _ = rollout(seed, (t_int, k), keep=True)
            if ok2:
                tops, wrists, states, actions = traj
                sl = slice(max(0, t_int - 40), steps2)                  # 留下：插入前 2 秒 + 脱身动作 + 之后重新靠近并抓住
                np.savez_compressed(f"{OUT}/ep_{n_resc:04d}.npz", top=np.array(tops[sl]), wrist=np.array(wrists[sl]), state=np.array(states[sl]),
                                    action=np.array(actions[sl], np.float32), seed=seed, v0=v0, t_int=t_int, prim=k)
                n_resc += 1; frames += sl.stop - sl.start; used[(t_int, k)] = used.get((t_int, k), 0) + 1
                break
    if (ep + 1) % 25 == 0:
        print(f"{ep + 1}/{N} 回合：直接成功 {n_ok}，失败 {n_fail}，其中搜索后救回 {n_resc}（{frames} 帧，平均试 {tries / max(n_fail, 1):.1f} 次）| "
              f"已用 {(time.time() - t0) / 60:.0f} 分钟", flush=True)
print(f"完成：{N} 回合，直接成功 {n_ok}，失败 {n_fail}，救回 {n_resc}，保存 {frames} 帧到 {OUT}")
print("各种救法被采用的次数：", {f"第{t}步{NAMES[k]}": v for (t, k), v in sorted(used.items(), key=lambda x: -x[1])})
