"""SmolVLA 的输出对“去噪起点的那串随机数”有多敏感？

这决定了一种强化学习方案是否可行：不改 SmolVLA 的任何参数，只训练一个小网络来“挑选起点噪声”。
如果换一份噪声输出的动作几乎不变，那挑噪声就没有意义；如果变化明显，小网络就有调节的余地，
而且无论怎么挑，输出的仍然是 SmolVLA 学过的那类动作，不会像直接加修正量那样把动作带偏。
做法：让 SmolVLA 正常跑若干回合。每隔几步，在同一个画面上换 16 份不同的噪声各推理一次，看第 1 步动作的离散程度。
分两种状态统计：正常靠近阶段（前 40 步），和卡住阶段（第 80 步之后还没夹住）。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/noise_sens.py MODEL_DIR ENGINE
"""
import sys
import numpy as np
from common import BINS
from resid_env import ResidEnv

M, ENG = sys.argv[1], sys.argv[2]
env = ResidEnv(M, ENG, v_range=(0.0, 0.0))
rng = np.random.default_rng(0)
S = {"正常靠近": [], "卡住": []}
for vr in [BINS[2], BINS[4], BINS[5]]:
    env.fixed_v = vr
    for ep in range(12):
        env.reset(seed=3000 + ep); t = 0; grasp = False
        while True:
            phase = "正常靠近" if t < 40 else ("卡住" if (t >= 80 and not grasp) else None)
            if phase and t % 6 == 0:
                A = []
                for k in range(16):
                    env.noise = rng.standard_normal((env.chunk, env.adim)).astype(np.float32)
                    _, acts = env._vla(); A.append(acts[0])
                env.noise = None
                A = np.array(A)                              # (16, 5)
                mean = A.mean(0); sd = A.std(0)
                unit = A[:, :3] / (np.linalg.norm(A[:, :3], axis=1, keepdims=True) + 1e-9)
                S[phase].append((np.linalg.norm(mean[:3]), sd[:3].mean(), sd[4], float(np.linalg.norm(unit.mean(0))),
                                 float(np.ptp(A[:, 4]))))
            _, r, term, trunc, info = env.step(np.zeros(5, np.float32)); t += 1
            grasp |= bool(env.last_info["grasped"])
            if term or trunc:
                break
for k, v in S.items():
    if not v:
        print(f"[{k}] 没有采到样本"); continue
    v = np.array(v)
    print(f"[{k}] {len(v)} 个画面，每个换 16 份噪声：位移动作的平均大小 {v[:, 0].mean():.2f}，换噪声造成的波动（标准差）{v[:, 1].mean():.3f}，"
          f"方向的一致程度 {v[:, 3].mean():.2f}（1 = 16 次完全同向）；夹爪动作的波动 {v[:, 2].mean():.3f}，16 次里最大和最小相差 {v[:, 4].mean():.2f}")
print("（动作每个分量的范围是 -1 到 1）")
print("NOISE DONE")
