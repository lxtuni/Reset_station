"""阶段 1 的 Gymnasium 环境：Reach（预抓取）—— 夹爪的 TCP 移到 BODY2 桅杆正上方，夹爪竖直、张开，
并且两指合拢的方向垂直于桅杆的宽面（这样往下一降，两指正好跨在桅杆两侧）。

Gymnasium 规定的接口（SB3 只调用这些）：
    observation_space / action_space   观测、动作的形状和范围
    reset(seed) -> (obs, info)          开始新回合
    step(action) -> (obs, reward, terminated, truncated, info)

设计（10-21 定，10-22 改：TCP、加手腕转角）：
- 动作 (4,)：TCP 位移 (dx, dy, dz) 每维 [-1, 1] × 1 cm，手腕转角增量 d_roll [-1, 1] × 0.1 rad。
  位置由 IK（envs/ik.py，4 个关节）换算成关节目标角；wrist_roll 直接给目标角；夹爪固定张开 GRIP_OPEN。
  都写进 data.ctrl，由 SO-101 的位置执行器（模拟舵机）去追。
- 控制频率 20 Hz：一个动作执行 10 个物理步（10 × 5 ms）。
- 观测 (28,)：关节角 6、关节角速度 6、TCP 位置 3、目标相对 TCP 3、BODY2 朝向 cos/sin 2、BODY2 水平速度 2、
  转角误差 sin(2e)/cos(2e) 2、上一步动作 4。都是真机测得到的量（关节角来自舵机反馈，BODY2 位姿来自顶置相机 + AprilTag）。
  转角误差 e 用 2e 的 sin/cos：两指从桅杆哪一侧夹都一样（e 和 e+180° 等价），2e 正好把它们变成同一个值。
- 目标：TCP 在桅杆顶正上方 3.5 cm。
- 奖励：每步 -（TCP 到目标的距离，m）- 0.1·|e|（rad）- 0.01·|动作|²；碰到缸壁/隔板/平台/缸底/BODY2 每步再 -1；
  成功（水平误差 < 1 cm、高度误差 < 1.5 cm、|e| < 10°）：+10 并结束回合。最多 100 步（5 s）后截断。
- BODY2：本阶段已经在可抓区里（桅杆离底座 0.16–0.28 m），随机位置和朝向，静止；水开着（浮着不动）。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from envs.ik import IK
from envs.water import Water

XML = "assets/robotstudio_so101/reset_station.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OBSTACLES = ["tank_wall_back", "tank_wall_front", "tank_wall_yneg", "tank_wall_ypos", "tank_bottom",
             "fence", "mount_plate", "table", "body2_hull", "body2_mast"]
MAST_TOP = 0.085                     # 桅杆顶相对 BODY2 原点（船体中心）的高度：船体半高 0.025 + 桅杆 0.06
HOVER = 0.035                        # 目标 = TCP 在桅杆顶上方 3.5 cm：夹爪最低点（指尖和碰撞盒）在 TCP 下方约 1.75 cm，
                                     # 所以指尖离桅杆顶约 1.75 cm（10-22：原先 2 cm 时指尖只高 2.5 mm，悬停时常擦到桅杆）
GRIP_OPEN = 0.35                     # 夹爪张开角（rad）：两指尖相距约 4 cm，跨 1 cm 厚的桅杆两边各留约 1.5 cm
ROLL_STEP = 0.1                      # 手腕转角动作 ±1 对应 ±0.1 rad（约 5.7°）
WS_LO = np.array([0.09, -0.24, -0.02])   # 指令 TCP 位置的允许范围（工作空间盒子），超出就截断
WS_HI = np.array([0.40, 0.24, 0.15])


def wrap_half_pi(x):
    """把角度折到 (-90°, 90°]：两指合拢方向反过来也一样能夹。"""
    return (x + np.pi / 2) % np.pi - np.pi / 2


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, xml=XML, max_steps=100, n_substeps=10, step_size=0.01, render_mode=None):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.water = Water(self.m)
        self.ik = IK(self.m, max_iter=3)                     # 每个控制步只迭代 3 次（从上一步的解热启动）
        self.max_steps, self.n_sub, self.step_size = max_steps, n_substeps, step_size
        self.render_mode = render_mode
        self._renderer = None
        # 编号
        self.key_home = self.m.key("home").id
        jid = self.m.joint("body2_free").id
        self.bqa, self.bva = self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]   # BODY2 在 qpos / qvel 里的起始位置
        self.bid = self.m.body("body2").id
        aj = [self.m.joint(n).id for n in ARM_JOINTS]
        self.arm_qa = np.array([self.m.jnt_qposadr[j] for j in aj])
        self.arm_va = np.array([self.m.jnt_dofadr[j] for j in aj])
        rj, gj = self.m.joint("wrist_roll").id, self.m.joint("gripper").id
        self.roll_qa, self.grip_qa = self.m.jnt_qposadr[rj], self.m.jnt_qposadr[gj]
        self.roll_lo, self.roll_hi = self.m.jnt_range[rj]
        self.roll_act, self.grip_act = self.m.actuator("wrist_roll").id, self.m.actuator("gripper").id
        self.obst = {self.m.geom(n).id for n in OBSTACLES}
        self.body2_geoms = {self.m.geom("body2_hull").id, self.m.geom("body2_mast").id}
        self.arm_geoms = {g for g in range(self.m.ngeom)
                          if self.m.geom_bodyid[g] != 0 and g not in self.body2_geoms}
        # 两指合拢方向（在 gripper 刚体坐标系里）：张开时从固定指尖指向转动指尖
        self.gbody = self.m.body("gripper").id
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key_home)
        self.d.qpos[self.grip_qa] = GRIP_OPEN
        mujoco.mj_forward(self.m, self.d)
        tips = lambda p: np.mean([self.d.geom_xpos[self.m.geom(f"{p}_sph_tip{i}").id] for i in (1, 2, 3)], 0)
        v = tips("moving_jaw") - tips("fixed_jaw")
        self.close_dir_local = self.d.xmat[self.gbody].reshape(3, 3).T @ (v / np.linalg.norm(v))
        # 空间：动作 4 维 [-1, 1]；观测 28 维，范围不限（SB3 会自己处理）
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)

    # ---------- 内部工具 ----------
    def _physics_step(self):
        mujoco.mj_step1(self.m, self.d)
        self.water.apply(self.d)                              # 水的力必须插在 step1 和 step2 之间（见 water.py）
        mujoco.mj_step2(self.m, self.d)

    def _target(self):
        return self.d.xpos[self.bid] + np.array([0.0, 0.0, MAST_TOP + HOVER])

    def _boat_yaw(self):
        R = self.d.xmat[self.bid].reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])                   # 船头方向（船体 x 轴）在水平面上的角度

    def _angle_err(self):
        """两指合拢方向（水平投影）与"桅杆宽面的法线"（船体 y 轴）之间的夹角，折到 (-90°, 90°]。"""
        c = self.d.xmat[self.gbody].reshape(3, 3) @ self.close_dir_local
        return float(wrap_half_pi(np.arctan2(c[1], c[0]) - (self._boat_yaw() + np.pi / 2)))

    def _arm_collision(self):
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.arm_geoms and g2 in self.obst) or (g2 in self.arm_geoms and g1 in self.obst):
                return True
        return False

    def _obs(self):
        tcp = self.ik.tcp(self.d)
        yaw, e = self._boat_yaw(), self._angle_err()
        return np.concatenate([
            self.d.qpos[self.arm_qa], self.d.qvel[self.arm_va], tcp, self._target() - tcp,
            [np.cos(yaw), np.sin(yaw)], self.d.qvel[self.bva:self.bva + 2],
            [np.sin(2 * e), np.cos(2 * e)], self.prev_action,
        ]).astype(np.float32)

    # ---------- Gymnasium 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                              # 设好 self.np_random（Gymnasium 统一的随机数生成器）
        rng = self.np_random
        for _ in range(100):                                  # 拒绝采样：BODY2 摆好后不能碰到任何东西
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key_home)
            self.d.qpos[self.grip_qa] = GRIP_OPEN             # 夹爪一开始就张开
            self.d.ctrl[self.grip_act] = GRIP_OPEN
            r, th = rng.uniform(0.16, 0.28), rng.uniform(-0.9, 0.9)
            x, y = max(r * np.cos(th), 0.15), r * np.sin(th)
            yaw = rng.uniform(-np.pi, np.pi)
            self.d.qpos[self.bqa:self.bqa + 7] = [x, y, self.water.water_z - 0.0168 + 0.025,
                                                   np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.m, self.d)
            if not any(c.geom1 in self.body2_geoms or c.geom2 in self.body2_geoms for c in self.d.contact[:self.d.ncon]):
                break
        self.q_cmd = self.d.qpos.copy()                       # IK 的"指令姿态"，每步从它热启动
        self.tcp_cmd = self.ik.tcp(self.d).copy()             # 指令 TCP 位置（动作在它上面累加）
        self.roll_cmd = float(self.d.qpos[self.roll_qa])      # 指令手腕转角
        self.prev_action = np.zeros(4)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # 手腕转角：直接累加，截到关节限位
        self.roll_cmd = float(np.clip(self.roll_cmd + a[3] * ROLL_STEP, self.roll_lo, self.roll_hi))
        self.q_cmd[self.roll_qa] = self.roll_cmd              # 先告诉 IK 新的转角（TCP 不在转轴上，转角会让它挪几毫米）
        # 位置：TCP 指令累加 → IK
        self.tcp_cmd = np.clip(self.tcp_cmd + a[:3] * self.step_size, WS_LO, WS_HI)
        self.q_cmd, _ = self.ik.solve(self.q_cmd, self.tcp_cmd)
        self.d.ctrl[self.ik.qadr] = self.q_cmd[self.ik.qadr]
        self.d.ctrl[self.roll_act] = self.roll_cmd
        collided = False
        for _ in range(self.n_sub):
            self._physics_step()
            collided |= self._arm_collision()
        self.steps += 1
        self.prev_action = a
        rel = self._target() - self.ik.tcp(self.d)
        dist, e = float(np.linalg.norm(rel)), self._angle_err()
        success = np.hypot(rel[0], rel[1]) < 0.01 and abs(rel[2]) < 0.015 and abs(e) < np.radians(10)
        reward = -dist - 0.1 * abs(e) - 0.01 * float(a @ a) - (1.0 if collided else 0.0) + (10.0 if success else 0.0)
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = dict(dist=dist, angle_err_deg=float(np.degrees(e)), success=bool(success), is_success=bool(success),
                    collided=collided)                       # is_success：SB3 的 EvalCallback 只认这个名字
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0.18, 0, 0.0]; self._cam.distance = 0.85; self._cam.azimuth = 140; self._cam.elevation = -25
        self._renderer.update_scene(self.d, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def scripted_action(obs):
    """手写的"脚本策略"：位置直接朝目标走；转角按误差反方向转。用来检查环境可解，也是 RL 的对照。"""
    e = 0.5 * np.arctan2(obs[22], obs[23])                    # 由 sin(2e)、cos(2e) 还原 e
    return np.clip(np.r_[obs[15:18] / 0.01, -e / ROLL_STEP], -1.0, 1.0)
