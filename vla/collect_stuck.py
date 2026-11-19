"""第三轮 DAgger，专门针对“第一次没夹住之后不会重来”。

卡住分析的结论：在失败回合的后半段，SmolVLA 的动作和教师在同一状态下的动作几乎不相关（方向一致程度 +0.17，
正常靠近阶段是 +0.76）。教师在那种状态下会明显上抬、绕开再重新对准；SmolVLA 则一直朝桅杆的方向低位推。
也就是说它缺的是一种没见过的动作，而不是把已有的动作做得更准。这种情况靠“只留下自己成功的回合”学不出来，
最直接的办法是让教师在这些状态下给出示范。
做法：SmolVLA 自己开（不加扰动）。每一步都记录教师的动作作为标签。
  - 到第 60 步还没夹住的回合才保存，而且只保存第 40 步以后的部分（卡住和脱身的阶段），最多 130 帧
  - 其中一半的回合，从第 60 步起把控制权交给教师：这样能采到“教师脱身并重新抓住”的整段过程
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/collect_stuck.py MODEL_DIR ENGINE OUT_DIR 回合数 种子起点
"""
import os, sys, time
import numpy as np
from stable_baselines3 import SAC
from common import TEACHER
from resid_env import ResidEnv

M, ENG, OUT, N, S0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
env = ResidEnv(M, ENG, alpha=1.0, fast_bias=0.6); inner = env.env
teacher = SAC.load(TEACHER, device="cpu")
rng = np.random.default_rng(S0)
saved = frames = n_stuck = ok_takeover = n_takeover = ok_student = n_student = 0
t0 = time.time()
for ep in range(N):
    env.reset(seed=S0 + ep)
    takeover = rng.random() < 0.5
    tops, wrists, states, actions, grasp_seen, stuck = [], [], [], [], False, False
    while True:
        t = len(actions)
        a_t = np.asarray(teacher.predict(inner._obs(), deterministic=True)[0], np.float32)
        if t == 60 and not grasp_seen:
            stuck = True
        tops.append(env.last_top); wrists.append(env.last_wrist); states.append(env.last_q); actions.append(a_t.copy())
        delta = (a_t - env.a_vla) if (stuck and takeover) else np.zeros(5, np.float32)   # 卡住后：一半回合交给教师开
        obs, r, term, trunc, info = env.step(delta)
        grasp_seen |= bool(env.last_info["grasped"])
        if term or trunc:
            break
    if stuck:
        n_stuck += 1
        if takeover:
            n_takeover += 1; ok_takeover += int(info["success"])
        else:
            n_student += 1; ok_student += int(info["success"])
        sl = slice(40, min(len(actions), 170))
        np.savez_compressed(f"{OUT}/ep_{saved:04d}.npz", top=np.array(tops[sl]), wrist=np.array(wrists[sl]), state=np.array(states[sl]),
                            action=np.array(actions[sl], np.float32), seed=S0 + ep, v0=env.last_info["v0"], takeover=takeover)
        saved += 1; frames += sl.stop - sl.start
    if (ep + 1) % 50 == 0:
        print(f"{ep + 1}/{N} 回合：到第 60 步还没夹住的 {n_stuck} 个（保存 {frames} 帧）| 其中交给教师的 {n_takeover} 个，最后成功 {ok_takeover} 个 | "
              f"留给 SmolVLA 自己的 {n_student} 个，最后成功 {ok_student} 个 | 已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{N} 回合，保存 {saved} 个卡住的回合，{frames} 帧，存在 {OUT}")
