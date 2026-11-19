"""卡住的时候，SmolVLA 到底在输出什么？

失败分析的结论是：大多数失败是第一次没夹住之后停在 8–10 cm 外。这里进一步看“停住”期间模型的动作：
  - 动作有多大（是不是接近 0）
  - 方向对不对（和“从夹爪指向桅杆”的方向的夹角，用仿真器里的真实位置算，只用于分析）
  - 同一时刻教师会输出多大的动作、朝哪个方向
这决定下一步该怎么做：如果方向对但幅度太小，放大动作就可能解决；如果方向本身是乱的，就需要新的数据。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/stuck_analysis.py MODEL_DIR ENGINE [种子起点=0]
"""
import sys
import numpy as np
from stable_baselines3 import SAC
from common import BINS, TEACHER
from resid_env import ResidEnv

M, ENG = sys.argv[1], sys.argv[2]; S0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
env = ResidEnv(M, ENG, v_range=(0.0, 0.0)); inner = env.env
teacher = SAC.load(TEACHER, device="cpu")
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
rows = {"stuck": [], "normal": []}
n_fail = 0
for vr in BINS:
    env.fixed_v = vr
    for ep in range(30):
        env.reset(seed=S0 + ep)
        obs_t = inner._obs()                                 # 教师用的观测（含船的位置估计），只用于分析
        log, grasped_ever = [], False
        while True:
            rel = inner._grip_point() - inner.ik.tcp(inner.d)                # 从夹爪指向桅杆（真实值）
            a_s = env.a_vla.copy(); a_t = np.asarray(teacher.predict(obs_t, deterministic=True)[0], np.float32)
            log.append((len(log), float(np.linalg.norm(rel)), np.linalg.norm(a_s[:3]), cos(a_s[:2], rel[:2]), np.linalg.norm(a_t[:3]),
                        cos(a_t[:2], rel[:2]), cos(a_s[:3], a_t[:3]), a_s[4], a_t[4], a_s[2], a_t[2]))
            _, r, term, trunc, info = env.step(np.zeros(5, np.float32))
            obs_t = inner._obs(); grasped_ever |= bool(env.last_info["grasped"])
            if term or trunc:
                break
        L = np.array(log)
        if not info["success"]:
            n_fail += 1
            rows["stuck"].append(L[L[:, 0] >= 80])           # 失败回合的第 80 步之后：停住的阶段
        else:
            rows["normal"].append(L[L[:, 0] < 25])           # 成功回合的前 25 步：正常靠近的阶段
print(f"种子 {S0} 起：失败 {n_fail} 回合")
for k, name in [("normal", "成功回合的靠近阶段"), ("stuck", "失败回合第 80 步之后")]:
    A = np.concatenate(rows[k]) if rows[k] else np.zeros((1, 11))
    print(f"[{name}] {len(A)} 步：离桅杆 {np.median(A[:, 1]) * 100:.1f} cm | SmolVLA 位移动作的大小 {np.median(A[:, 2]):.2f}，方向和“指向桅杆”的一致程度 {np.mean(A[:, 3]):+.2f} | "
          f"教师：大小 {np.median(A[:, 4]):.2f}，一致程度 {np.mean(A[:, 5]):+.2f} | 两者方向的一致程度 {np.mean(A[:, 6]):+.2f} | "
          f"夹爪动作 SmolVLA {np.mean(A[:, 7]):+.2f} / 教师 {np.mean(A[:, 8]):+.2f} | 上下方向 SmolVLA {np.mean(A[:, 9]):+.2f} / 教师 {np.mean(A[:, 10]):+.2f}")
print("（一致程度：+1 = 完全同向，0 = 无关，-1 = 反向。动作每个分量的范围是 -1 到 1）")
print("STUCK DONE")
