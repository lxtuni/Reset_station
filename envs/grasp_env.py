"""阶段 2 的 Gymnasium 环境：Grasp —— 从预抓取位姿附近出发，下降、让两指跨住桅杆、合拢、稳稳夹住。

继承 ReachEnv（envs/reset_env.py），复用场景、水、IK、碰撞检测和转角误差的定义。
设计（10-22 定）：
- 起点：机械臂在"预抓取位姿"附近（TCP 在桅杆顶上方 3.5 cm，位置随机偏 ±2 cm、转角随机偏 ±20°），夹爪张开 0.35 rad。
  阶段 1 的策略负责把夹爪带到这里；部署时先跑阶段 1 再跑阶段 2（分层）。
- 动作 (5,)：(dx, dy, dz) × 1 cm、d_roll × 0.1 rad、d_grip × 0.1 rad（负 = 合拢）。
- "夹住"：两根手指都碰到桅杆，并且夹爪实际角度比指令角度大 0.03 rad 以上（被桅杆挡住、合不拢）。
  后者在真机上靠 STS3215 的位置反馈就能测到。夹住 1 cm 厚的桅杆时，夹爪约停在 -0.08 rad。
- 成功：连续夹住 10 步（0.5 s），并且船被推开 < MAX_PUSH（5 cm）、倾斜 < 10°。
  （10-22 由 3 cm 放宽到 5 cm：阶段 3 本来就要拖船，夹取时被推开几厘米无妨，关键是夹稳。部署是分层的：先阶段 1 再阶段 2。）
- 观测 (30,)：阶段 1 的观测（上一步动作改为 5 维）+ 夹爪的"指令角 - 实际角"（1 维）。
- 奖励（分段，每步都 ≤ 0，只有成功时为正）：-（TCP 到夹持点的距离）- 0.1·|转角误差| - 0.5 × 没碰到桅杆的手指数 - 1（没夹住时）
  10-22 第二版：原来是 "+0.5 × 碰到的手指数 + 2（夹住时）"，每步最多 +3，而成功只给一次 +20 且回合结束 →
  "一直夹着但故意不满足成功条件"能刷到约 300 分，策略学会刷分（残差 SAC 回报上升、成功率反降到 65 %）。
  改成惩罚后，越早成功扣分越少。
  - 5 × 船被推开的距离 - 0.01·|动作|² - 1（撞缸壁/隔板/平台，或碰到船体）；成功 +20 并结束。
  夹持点 = 桅杆正中、桅杆顶往下 1.5 cm（两指跨住桅杆最上面约 3 cm）。最多 100 步（5 s）。
"""
import numpy as np
from gymnasium import spaces
import mujoco

from envs.reset_env import ReachEnv, MAST_TOP, HOVER, GRIP_OPEN, ROLL_STEP, WS_LO, WS_HI

GRIP_STEP = 0.1                      # 夹爪动作 ±1 对应 ±0.1 rad
GRIP_DEPTH = 0.015                   # 夹持点在桅杆顶下方 1.5 cm
HOLD_STEPS = 10                      # 连续夹住 10 步算成功
MAX_PUSH = 0.05                      # 成功时船最多被推开 5 cm
OBST_GRASP = ["tank_wall_back", "tank_wall_front", "tank_wall_yneg", "tank_wall_ypos", "tank_bottom",
              "fence", "mount_plate", "table", "body2_hull"]          # 和阶段 1 不同：桅杆不算障碍物，可以碰


class GraspEnv(ReachEnv):
    def __init__(self, stage1_states=None, p_stage1=0.5, **kw):
        """stage1_states：阶段 1 策略成功时的状态文件（.npz，由 eval/collect_states.py 生成）。给了的话，
        每回合以概率 p_stage1 从这些"真实接力起点"开始，其余回合仍用"预抓取 + 随机偏差"。
        10-22：残差 SAC 在随机偏差起点上 96 %，接到阶段 1 的 SAC 后面只有 80 %（起点分布不一样）→ 加了这个。"""
        super().__init__(**kw)
        m = self.m
        from envs.state_io import load_states
        self.stage1_states = load_states(stage1_states) if stage1_states else None
        self.p_stage1 = p_stage1
        self.obst = {m.geom(n).id for n in OBST_GRASP}
        self.mast = m.geom("body2_mast").id
        gb, mb = m.body("gripper").id, m.body("moving_jaw_so101_v1").id
        self.fixed_finger = {g for g in range(m.ngeom) if m.geom_bodyid[g] == gb}      # 固定指（含手掌）的碰撞体
        self.moving_finger = {g for g in range(m.ngeom) if m.geom_bodyid[g] == mb}     # 转动指的碰撞体
        self.grip_lo, self.grip_hi = m.jnt_range[m.joint("gripper").id]
        self.grip_cmd = GRIP_OPEN                             # 父类 reset 里会调用 _obs()，要先有这个值
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float32)

    # ---------- 内部工具 ----------
    def _grip_point(self):
        """TCP 应该去的位置：让桅杆处在两指当前空隙的正中间（随夹爪开合移动）。
        TCP 离固定指内表面只有 6 mm（= 桅杆半厚 5 mm + 1 mm），离转动指在张开时有 3 cm 多；如果始终让 TCP 对准桅杆中心，
        下降时水平偏 1 mm 固定指就会落在桅杆顶上，把浮着的船推走（10-22 调试发现）。
        所以：空隙 g（由夹爪角度线性估计：g ≈ 16.3 mm + 76 mm/rad × q，测于 10-22）的中点在 TCP 往转动指方向 g/2 - 6 mm 处；
        让这个中点对准桅杆 → TCP 的目标 = 桅杆中心 - (g/2 - 6 mm) × 合拢方向。合拢到 g ≈ 1 cm 时偏移为 0，正好 TCP 对准桅杆。"""
        base = self.d.xpos[self.bid] + np.array([0.0, 0.0, MAST_TOP - GRIP_DEPTH])
        gap = max(0.0046, 0.0163 + 0.076 * float(self.d.qpos[self.grip_qa]))
        shift = max(0.0, gap / 2 - 0.006)
        c = self.d.xmat[self.gbody].reshape(3, 3) @ self.close_dir_local
        c[2] = 0.0
        c /= np.linalg.norm(c)
        return base - shift * c

    def _finger_contacts(self):
        """返回 (固定指碰到桅杆, 转动指碰到桅杆)。"""
        f = mv = False
        for c in self.d.contact[:self.d.ncon]:
            pair = {c.geom1, c.geom2}
            if self.mast in pair:
                other = (pair - {self.mast}).pop() if len(pair) == 2 else self.mast
                f |= other in self.fixed_finger
                mv |= other in self.moving_finger
        return f, mv

    def _obs(self):
        tcp = self.ik.tcp(self.d)
        yaw, e = self._boat_yaw(), self._angle_err()
        grip_err = self.grip_cmd - float(self.d.qpos[self.grip_qa])            # 指令 - 实际：夹到东西时明显为负
        return np.concatenate([
            self.d.qpos[self.arm_qa], self.d.qvel[self.arm_va], tcp, self._grip_point() - tcp,
            [np.cos(yaw), np.sin(yaw)], self.d.qvel[self.bva:self.bva + 2],
            [np.sin(2 * e), np.cos(2 * e)], self.prev_action, [grip_err],
        ]).astype(np.float32)

    # ---------- Gymnasium 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                              # 阶段 1 的 reset：随机放船、机械臂回 home、夹爪张开
        rng = self.np_random
        if self.stage1_states and rng.random() < self.p_stage1:   # 从阶段 1 策略的真实终点开始
            from envs.state_io import restore
            restore(self, self.stage1_states[int(rng.integers(len(self.stage1_states)))])
            self.grip_cmd = GRIP_OPEN
            self.boat_xy0 = self.d.xpos[self.bid][:2].copy()
            self.hold = 0
            self.prev_action = np.zeros(5)
            return self._obs(), {}
        # 把机械臂摆到预抓取位姿附近：先定转角（对准 + 随机偏差），再用完整 IK 解位置（+ 随机偏差）
        from envs.ik import IK
        ik_full = getattr(self, "_ik_full", None) or IK(self.m)
        self._ik_full = ik_full
        for _ in range(20):
            q = self.d.qpos.copy()
            e0 = self._angle_err()
            q[self.roll_qa] = np.clip(q[self.roll_qa] - e0 + np.radians(rng.uniform(-20, 20)), self.roll_lo, self.roll_hi)
            # 水平 ±2 cm；高度只往上偏 0–2 cm（往下偏会让手指一开始就压进桅杆里）
            target = self._target() + np.r_[rng.uniform(-0.02, 0.02, 2), rng.uniform(0.0, 0.02)]
            q, info = ik_full.solve(q, target)
            self.d.qpos[:] = q
            self.d.ctrl[self.ik.qadr] = q[self.ik.qadr]
            self.d.ctrl[self.roll_act] = q[self.roll_qa]
            mujoco.mj_forward(self.m, self.d)
            touching_mast = any(self.mast in (c.geom1, c.geom2) for c in self.d.contact[:self.d.ncon])
            if info["success"] and not self._arm_collision() and not touching_mast:
                break
        self.q_cmd = self.d.qpos.copy()
        self.tcp_cmd = self.ik.tcp(self.d).copy()
        self.roll_cmd = float(self.d.qpos[self.roll_qa])
        self.grip_cmd = GRIP_OPEN
        self.boat_xy0 = self.d.xpos[self.bid][:2].copy()      # 记下船的起始位置，用来算"被推开多远"
        self.hold = 0
        self.prev_action = np.zeros(5)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.roll_cmd = float(np.clip(self.roll_cmd + a[3] * ROLL_STEP, self.roll_lo, self.roll_hi))
        self.grip_cmd = float(np.clip(self.grip_cmd + a[4] * GRIP_STEP, self.grip_lo, GRIP_OPEN))
        self.q_cmd[self.roll_qa] = self.roll_cmd
        self.tcp_cmd = np.clip(self.tcp_cmd + a[:3] * self.step_size, WS_LO, WS_HI)
        self.q_cmd, _ = self.ik.solve(self.q_cmd, self.tcp_cmd)
        self.d.ctrl[self.ik.qadr] = self.q_cmd[self.ik.qadr]
        self.d.ctrl[self.roll_act] = self.roll_cmd
        self.d.ctrl[self.grip_act] = self.grip_cmd
        collided = False
        for _ in range(self.n_sub):
            self._physics_step()
            collided |= self._arm_collision()
        self.steps += 1
        self.prev_action = a
        # 状态
        rel = self._grip_point() - self.ik.tcp(self.d)
        dist, e = float(np.linalg.norm(rel)), self._angle_err()
        f_touch, m_touch = self._finger_contacts()
        stalled = float(self.d.qpos[self.grip_qa]) - self.grip_cmd > 0.03
        grasped = f_touch and m_touch and stalled
        self.hold = self.hold + 1 if grasped else 0
        push = float(np.linalg.norm(self.d.xpos[self.bid][:2] - self.boat_xy0))
        Rb = self.d.xmat[self.bid].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1))))
        success = self.hold >= HOLD_STEPS and push < MAX_PUSH and tilt < 10
        reward = (-dist - 0.1 * abs(e) - 0.5 * (2 - f_touch - m_touch) - (0.0 if grasped else 1.0)
                  - 5.0 * push - 0.01 * float(a @ a) - (1.0 if collided else 0.0) + (20.0 if success else 0.0))
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = dict(dist=dist, angle_err_deg=float(np.degrees(e)), touch=(f_touch, m_touch), grasped=grasped,
                    hold=self.hold, push=push, tilt=tilt, success=bool(success), is_success=bool(success), collided=collided)
        return self._obs(), reward, terminated, truncated, info


def scripted_grasp(obs, phase=None):
    """手写的分阶段策略（状态机），用来检查环境可解：
    对准（水平到位、转角 < 5°）→ 下降到夹持点 → 合拢并保持。只看观测，不偷看仿真内部。"""
    rel, e, grip_err = obs[15:18], 0.5 * np.arctan2(obs[22], obs[23]), obs[29]
    horiz_ok = np.hypot(rel[0], rel[1]) < 0.006 and abs(e) < np.radians(5)
    a = np.zeros(5)
    a[3] = np.clip(-e / ROLL_STEP, -1, 1)
    if not horiz_ok and rel[2] < -0.012:                       # 还没对准、而且还在高处：只做水平移动和转角
        a[0:2] = np.clip(rel[0:2] / 0.01, -1, 1)
        a[4] = 1.0                                            # 保持张开
    else:                                                      # 对准了：下降到夹持点；到了就合拢，合拢时继续跟踪夹持点
        a[0:3] = np.clip(rel / 0.01, -0.5, 0.5)                # （夹持点随夹爪合拢移向桅杆，手臂把固定指带过去）；慢一点，减少跟踪滞后
        closing = obs[5] < 0.33                                # obs[5] = 夹爪实际角度：一旦开始合拢就一直合下去，不来回开合
        a[4] = -1.0 if (closing or np.linalg.norm(rel) < 0.006) else 1.0
    return a
