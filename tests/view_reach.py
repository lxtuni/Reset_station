"""在 viewer 里看 ReachEnv 的回合：默认用脚本策略（直接朝目标走），也可以换成随机策略对比。
绿球 = 目标（桅杆顶上方 2 cm）。每回合结束停 1 秒再开下一回合，终端打印结果。

运行（项目根目录，激活 .venv）：
    python tests/view_reach.py            # 脚本策略
    python tests/view_reach.py random     # 随机策略（看看"什么都没学会"是什么样）
以后训练好了，这里再加一个"加载 SAC 模型"的选项。
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.reset_env import ReachEnv, scripted_action

mode = sys.argv[1] if len(sys.argv) > 1 else "scripted"
env = ReachEnv()
rng = np.random.default_rng()
policy = scripted_action if mode == "scripted" else (lambda o: rng.uniform(-1, 1, 3))
obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    R = 0.0
    while viewer.is_running():
        t0 = time.time()
        obs, r, term, trunc, info = env.step(policy(obs))
        R += r
        with viewer.lock():
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]),
                                env._target(), np.eye(3).ravel(), np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float32))
            viewer.user_scn.ngeom = 1
        viewer.sync()
        if term or trunc:
            print(f"{mode}: {'成功' if info['success'] else '失败'}，{env.steps} 步，结束距离 {1000*info['dist']:.0f} mm，"
                  f"回报 {R:.1f}{'，碰撞过' if info['collided'] else ''}")
            time.sleep(1.0)
            obs, _ = env.reset(seed=int(rng.integers(1_000_000))); R = 0.0
        dt = env.n_sub * env.m.opt.timestep - (time.time() - t0)   # 按真实时间：一个控制步 = 50 ms
        if dt > 0:
            time.sleep(dt)
