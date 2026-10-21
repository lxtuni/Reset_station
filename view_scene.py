"""打开复位台场景：机械臂停在 home；BODY2 从缸的远端出发，朝可抓区游过来，进区后停桨漂行；每 15 秒重新随机一次。

复位的第一步：BODY2 自己游进机械臂的可抓区（离底座约 0.06–0.37 m 能竖直朝下夹桅杆，见 docs/reach_map.png）。
这里还没有真正的游泳模型，用"沿船头方向的推力"代替：推力大小正好抵消该速度下的水阻力，所以船保持匀速；
进入可抓区就把推力关掉（相当于 BODY2 停止划水），之后靠水的阻力慢慢停下。以后换成 MA 的 PPO 航向控制。

运行（WSL，项目根目录，先激活 .venv）：
    python view_scene.py        # 每次随机都不一样
    python view_scene.py 0      # 指定随机种子 0：每次运行都重现同一串回合
"""
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from envs.water import Water

XML = "assets/robotstudio_so101/reset_station.xml"
EPISODE_S = 15.0                     # 每隔多少秒重新随机一次
GRASP_R = 0.30                       # 可抓区：桅杆离底座的水平距离 ≤ 0.30 m（夹爪严格竖直最远约 0.32 m，
                                     # 见 tests/test_ik.py；reach_map.png 的 0.37 m 允许夹爪斜 20°，偏乐观）
GRASP_XMIN = 0.14                    #         且在隔板前面（隔板 0.085 + 半个船宽 0.058）
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else None   # 命令行第 1 个参数当随机种子；没给就是 None（每次不同）
rng = np.random.default_rng(SEED)    # 随机数生成器：同一个种子 → 同一串随机数

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
water = Water(m)

# BODY2 的 freejoint 在 qpos / qvel 里从哪一位开始
jid = m.joint("body2_free").id
qa, va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
bid = m.body("body2").id
body2_geoms = {m.geom("body2_hull").id, m.geom("body2_mast").id}
swim = dict(on=False, speed=0.0, t_in=None)   # 当前回合的"游泳"状态


def in_grasp_zone(d):
    """桅杆（= 船体中心）是否已经在可抓区里。"""
    x, y = d.qpos[qa], d.qpos[qa + 1]
    return np.hypot(x, y) <= GRASP_R and x >= GRASP_XMIN


def randomize(d):
    """机械臂回到 home；BODY2 放在缸的远端，船头大致对准可抓区里的一个随机目标点，以随机速度出发。"""
    for _ in range(100):                                     # 拒绝采样：摆放后碰到东西就重抽
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        x = rng.uniform(0.42, 0.86)                          # 出发点：可抓区外面、缸的远端
        y = rng.uniform(-0.18, 0.18)
        tx = rng.uniform(0.18, 0.30)                         # 目标点：可抓区里面
        ty = rng.uniform(-0.12, 0.12)
        yaw = np.arctan2(ty - y, tx - x) + np.radians(rng.uniform(-10, 10))   # 对准目标，再加 ±10° 的偏差
        z = water.water_z - 0.0168 + 0.025                   # 平衡吃水 ≈ 0.51 kg /(ρ·0.262·0.116) = 16.8 mm
        d.qpos[qa:qa + 7] = [x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]   # 位置 + 绕 z 转 yaw 的四元数
        mujoco.mj_forward(m, d)                              # 算一次几何和接触，但不推进时间
        if not any(c.geom1 in body2_geoms or c.geom2 in body2_geoms for c in d.contact[:d.ncon]):
            break
    speed = rng.uniform(0.05, 0.20)                          # 游速 5–20 cm/s（9-29 实测旧机 0.14–0.26 m/s）
    yaw_rate = rng.uniform(-0.1, 0.1)                        # 小的初始转动扰动 rad/s
    d.qvel[va:va + 6] = [speed * np.cos(yaw), speed * np.sin(yaw), 0, 0, 0, yaw_rate]  # 线速度（世界系）+ 角速度（船体系）
    swim.update(on=True, speed=speed, t_in=None)
    print(f"BODY2: 出发 ({x:.2f}, {y:+.2f})  目标 ({tx:.2f}, {ty:+.2f})  朝向 {np.degrees(yaw):6.1f}°  速度 {100*speed:.1f} cm/s")


def apply_swim(d):
    """游泳推力：沿船头方向（水平），大小 = 该速度下的纵向水阻力，所以能保持匀速。进可抓区后关掉。"""
    if not swim["on"]:
        return
    if in_grasp_zone(d):
        swim.update(on=False, t_in=d.time)
        print(f"       {d.time:.1f} s 进入可抓区，停桨")
        return
    v = swim["speed"]
    T = water.c1_lin[0] * v + water.c2_lin[0] * v * v        # 纵向阻力公式（和 water.py 一样）
    head = d.xmat[bid].reshape(3, 3)[:, 0].copy()            # 船体 x 轴（船头方向）在世界里的方向
    head[2] = 0.0
    head /= np.linalg.norm(head)
    d.xfrc_applied[bid, :3] += T * head                      # 加在 water.apply 写好的力上面


randomize(d)
with mujoco.viewer.launch_passive(m, d) as viewer:          # 被动 viewer：画面由我们的循环推进
    t_ep = time.time()
    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step1(m, d)                                # 1. 用当前状态算位置、速度
        water.apply(d)                                       # 2. 算水的力（读的是刚算好的位置，没有延迟）
        apply_swim(d)                                        # 3. 再加上游泳推力
        mujoco.mj_step2(m, d)                                # 4. 加上这些力，推进一步（5 ms）
        viewer.sync()                                        # 把新状态画出来
        if time.time() - t_ep > EPISODE_S:                   # 到时间就重新随机
            randomize(d); t_ep = time.time()
        dt = m.opt.timestep - (time.time() - t0)             # 按真实时间节奏跑
        if dt > 0:
            time.sleep(dt)
