"""在 viewer 里看阶段 2（Grasp）的回合。绿球 = 当前夹持点（随夹爪开合移动：张开时让桅杆在两指空隙正中，合拢后回到 TCP）。
每回合结束停 1 秒，终端打印：是否成功、步数、夹住保持了几步、船被推开多少。

运行（项目根目录，激活 .venv）：
    python tests/view_grasp.py            # 脚本策略（状态机：对准 → 下降 → 合拢）
    python tests/view_grasp.py random     # 随机策略
    python tests/view_grasp.py results/stage2_grasp/sac_seed0/best_model.zip   # 训练好的 SAC
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.grasp_env import GraspEnv, scripted_grasp

mode = sys.argv[1] if len(sys.argv) > 1 else "scripted"
env = GraspEnv()
rng = np.random.default_rng()
if mode == "scripted":
    policy = scripted_grasp
elif mode == "random":
    policy = lambda o: rng.uniform(-1, 1, 5)
else:
    from stable_baselines3 import SAC
    model = SAC.load(mode)
    policy = lambda o: model.predict(o, deterministic=True)[0]
    mode = "SAC"
obs, _ = env.reset(seed=int(rng.integers(1_000_000)))

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    max_hold = 0
    while viewer.is_running():
        t0 = time.time()
        obs, r, term, trunc, info = env.step(policy(obs))
        max_hold = max(max_hold, info["hold"])
        with viewer.lock():
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]),
                                env._grip_point(), np.eye(3).ravel(), np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float32))
            viewer.user_scn.ngeom = 1
        viewer.sync()
        if term or trunc:
            print(f"{mode}: {'成功' if info['success'] else '失败'}，{env.steps} 步，最长连续夹住 {max_hold} 步，"
                  f"船被推开 {1000*info['push']:.0f} mm，倾斜 {info['tilt']:.1f}°{'，撞到过东西' if info['collided'] else ''}")
            time.sleep(1.0)
            obs, _ = env.reset(seed=int(rng.integers(1_000_000))); max_hold = 0
        dt = env.n_sub * env.m.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
