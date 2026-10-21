"""看 IK 在仿真里怎么动：每 3 秒在船区随机取一个目标点（小球），机械臂用 IK 一步步追过去。

小球颜色：绿 = IK 解到了（误差 < 1 mm、夹爪偏离竖直 < 1°）；红 = 没解到（太远或太近）。
目标点有 70 % 取在可抓区内（离底座 ≤ 0.30 m），30 % 故意取在外面，让你看到够不着时手臂会怎样。
控制方式和以后的 RL 环境一样：每 20 ms 一个控制步，目标最多移动 1 cm，IK 从上一步的解出发迭代 3 次，
解出的关节角写进 data.ctrl，由 SO-101 的位置执行器（模拟舵机）去追，所以手臂动作有真实的滞后。
BODY2 放在缸的远端，开着水（不会沉），不参与。

运行（项目根目录，激活 .venv）：python tests/view_ik.py
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.ik import IK
from envs.water import Water

m = mujoco.MjModel.from_xml_path("assets/robotstudio_so101/reset_station.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
jid = m.joint("body2_free").id
qa = m.jnt_qposadr[jid]
d.qpos[qa:qa + 7] = [0.75, 0.0, -0.0318, 1, 0, 0, 0]       # BODY2 停在远端，不挡路
mujoco.mj_forward(m, d)
water = Water(m)
ik = IK(m)                    # 完整求解：用来判断目标到底够不够得着（决定小球颜色）
ik_step = IK(m, max_iter=3)   # 每个控制步用的快速版：只迭代 3 次
rng = np.random.default_rng()
N_PHYS = int(round(0.02 / m.opt.timestep))                  # 一个控制步 = 20 ms = 4 个物理步

tip = d.site_xpos[ik.sid].copy()                            # 当前"指令目标"（每步向真正的目标挪 ≤ 1 cm）
q_cmd = d.qpos.copy()


def new_target():
    r = rng.uniform(0.12, 0.30) if rng.random() < 0.7 else rng.uniform(0.30, 0.42)
    th = rng.uniform(-1.2, 1.2)
    p = np.array([r * np.cos(th), r * np.sin(th), rng.uniform(0.0, 0.08)])
    p[0] = max(p[0], 0.10)                                  # 不进隔板后面
    p[1] = np.clip(p[1], -0.22, 0.22)                       # 不超出两侧缸壁（y = ±0.25）
    _, info = ik.solve(d.qpos, p)
    print(f"目标 ({p[0]:.2f}, {p[1]:+.2f}, {p[2]:.2f})  离底座 {np.hypot(p[0], p[1]):.2f} m → "
          + ("能解" if info["success"] else f"解不到（{info['reason']}，剩 {1000*info['err_pos']:.0f} mm / {np.degrees(info['err_rot']):.0f}°）"))
    return p, info["success"]


goal, ok = new_target()
with mujoco.viewer.launch_passive(m, d) as viewer:
    t_goal = time.time()
    while viewer.is_running():
        t0 = time.time()
        # 1. 控制步：指令目标向 goal 挪最多 1 cm，再用 IK 解出关节角
        step = goal - tip
        if np.linalg.norm(step) > 0.01:
            step *= 0.01 / np.linalg.norm(step)
        tip = tip + step
        q_cmd, _ = ik_step.solve(q_cmd, tip)
        d.ctrl[ik.qadr] = q_cmd[ik.qadr]                     # 写进 4 个 IK 关节的位置执行器
        # 2. 物理步（20 ms）
        for _ in range(N_PHYS):
            mujoco.mj_step1(m, d)
            water.apply(d)
            mujoco.mj_step2(m, d)
        # 3. 画目标小球（user_scn：只画给你看的额外几何体，不参与物理）
        with viewer.lock():
            g = viewer.user_scn.geoms[0]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]), goal, np.eye(3).ravel(),
                                np.array([0.1, 0.8, 0.2, 0.9] if ok else [0.9, 0.15, 0.1, 0.9], dtype=np.float32))
            viewer.user_scn.ngeom = 1
        viewer.sync()
        if time.time() - t_goal > 3.0:
            real = d.site_xpos[ik.sid]
            print(f"   3 s 后实际尖端离目标 {1000*np.linalg.norm(real - goal):.1f} mm")
            goal, ok = new_target(); t_goal = time.time()
            tip = d.site_xpos[ik.sid].copy(); q_cmd = d.qpos.copy()   # 从实际姿态重新出发
        dt = 0.02 - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
