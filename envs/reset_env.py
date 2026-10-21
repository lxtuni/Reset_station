"""阶段 1 的 Gymnasium 环境：Reach —— 夹爪尖端移到 BODY2 桅杆顶上方 2 cm，夹爪保持竖直。

Gymnasium 规定的接口（SB3 只调用这些）：
    observation_space / action_space   观测、动作的形状和范围
    reset(seed) -> (obs, info)          开始新回合
    step(action) -> (obs, reward, terminated, truncated, info)

设计（10-21 定，见对话第 1 步的表）：
- 动作 (3,)：夹爪尖端的位移 (dx, dy, dz)，每维 [-1, 1] × 1 cm；由 IK（envs/ik.py）换算成 4 个关节的目标角，
  写进 data.ctrl，SO-101 的位置执行器（模拟舵机）去追。wrist_roll、夹爪本阶段不动。
- 控制频率 20 Hz：一个动作执行 10 个物理步（10 × 5 ms）。
- 观测 (25,)：关节角 6、关节角速度 6、尖端位置 3、目标相对尖端 3、BODY2 朝向 cos/sin 2、BODY2 水平速度 2、上一步动作 3。
  都是真机测得到的量（关节角来自舵机反馈，BODY2 位姿来自顶置相机 + AprilTag）；目标用相对位置。
- 奖励：每步 -（尖端到目标的距离，m）- 0.01·|动作|²；碰到缸壁/隔板/平台/缸底/BODY2 每步再 -1；
  距离 < 2 cm 算成功：+10 并结束回合。最多 100 步（5 s）后截断。
- BODY2：本阶段已经在可抓区里（桅杆离底座 ≤ 0.28 m），随机位置和朝向，静止；水开着（浮着不动）。
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
HOVER = 0.02                         # 目标 = 桅杆顶上方 2 cm
WS_LO = np.array([0.09, -0.24, -0.02])   # 指令尖端位置的允许范围（工作空间盒子），超出就截断
WS_HI = np.array([0.40, 0.24, 0.15])


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, xml=XML, max_steps=100, n_substeps=10, step_size=0.01, success_dist=0.02, render_mode=None):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.water = Water(self.m)
        self.ik = IK(self.m, max_iter=3)                     # 每个控制步只迭代 3 次（从上一步的解热启动）
        self.max_steps, self.n_sub, self.step_size, self.success_dist = max_steps, n_substeps, step_size, success_dist
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
        self.sid = self.ik.sid                                # 夹爪尖端
        self.obst = {self.m.geom(n).id for n in OBSTACLES}
        self.body2_geoms = {self.m.geom("body2_hull").id, self.m.geom("body2_mast").id}
        self.arm_geoms = {g for g in range(self.m.ngeom)
                          if self.m.geom_bodyid[g] != 0 and g not in self.body2_geoms}
        # 空间：动作 3 维 [-1, 1]；观测 25 维，范围不限（SB3 会自己处理）
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)

    # ---------- 内部工具 ----------
    def _physics_step(self):
        mujoco.mj_step1(self.m, self.d)
        self.water.apply(self.d)                              # 水的力必须插在 step1 和 step2 之间（见 water.py）
        mujoco.mj_step2(self.m, self.d)

    def _target(self):
        return self.d.xpos[self.bid] + np.array([0.0, 0.0, MAST_TOP + HOVER])

    def _arm_collision(self):
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.arm_geoms and g2 in self.obst) or (g2 in self.arm_geoms and g1 in self.obst):
                return True
        return False

    def _obs(self):
        tip = self.d.site_xpos[self.sid]
        R = self.d.xmat[self.bid].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])                    # 船头方向（船体 x 轴）在水平面上的角度
        return np.concatenate([
            self.d.qpos[self.arm_qa], self.d.qvel[self.arm_va], tip, self._target() - tip,
            [np.cos(yaw), np.sin(yaw)], self.d.qvel[self.bva:self.bva + 2], self.prev_action,
        ]).astype(np.float32)

    # ---------- Gymnasium 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                              # 设好 self.np_random（Gymnasium 统一的随机数生成器）
        rng = self.np_random
        for _ in range(100):                                  # 拒绝采样：BODY2 摆好后不能碰到任何东西
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key_home)
            r, th = rng.uniform(0.16, 0.28), rng.uniform(-0.9, 0.9)
            x, y = max(r * np.cos(th), 0.15), r * np.sin(th)
            yaw = rng.uniform(-np.pi, np.pi)
            self.d.qpos[self.bqa:self.bqa + 7] = [x, y, self.water.water_z - 0.0168 + 0.025,
                                                   np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.m, self.d)
            if not any(c.geom1 in self.body2_geoms or c.geom2 in self.body2_geoms for c in self.d.contact[:self.d.ncon]):
                break
        self.q_cmd = self.d.qpos.copy()                       # IK 的"指令姿态"，每步从它热启动
        self.tip_cmd = self.d.site_xpos[self.sid].copy()      # 指令尖端位置（动作在它上面累加）
        self.prev_action = np.zeros(3)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.tip_cmd = np.clip(self.tip_cmd + a * self.step_size, WS_LO, WS_HI)
        self.q_cmd, _ = self.ik.solve(self.q_cmd, self.tip_cmd)
        self.d.ctrl[self.ik.qadr] = self.q_cmd[self.ik.qadr]     # 只写 4 个 IK 关节；wrist_roll、夹爪保持 home 的指令
        collided = False
        for _ in range(self.n_sub):
            self._physics_step()
            collided |= self._arm_collision()
        self.steps += 1
        self.prev_action = a
        dist = float(np.linalg.norm(self._target() - self.d.site_xpos[self.sid]))
        success = dist < self.success_dist
        reward = -dist - 0.01 * float(a @ a) - (1.0 if collided else 0.0) + (10.0 if success else 0.0)
        terminated = success
        truncated = self.steps >= self.max_steps
        info = dict(dist=dist, success=success, collided=collided)
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
    """手写的"脚本策略"：直接朝目标走（相对位置 / 1 cm，截到 [-1, 1]）。用来检查环境可解，以后也可当 RL 的对照。"""
    return np.clip(obs[15:18] / 0.01, -1.0, 1.0)
