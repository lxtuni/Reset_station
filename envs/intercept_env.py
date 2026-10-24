"""拦截抓取：船从缸的远端朝可抓区漂过来，机械臂预测它的轨迹、追上去、同速对准、下降夹住，再把船停下来。
在船撞到缸壁或隔板之前完成。替代"BODY2 进区后自己刹车"的假设（阶段 0）以及阶段 1、2。

两种"预测"方式（对比实验）：
- predictor="kf"：卡尔曼滤波（envs/kalman.py）把带噪声、有延迟的相机测量变成位置 / 速度 / 朝向 / 角速度的估计，
  再预测 0.25 / 0.5 / 1.0 s 后桅杆的位置和"还有多久撞墙"，放进观测。
- predictor="history"：不做任何处理，观测里直接放最近 4 帧的原始测量，让网络自己学会"预测"。
两种观测含有同样的原始信息，区别只在有没有显式的估计和预测。

设计（10-23 定，可行性分析见对话：夹爪跟到约 0.2 m/s 误差还在几毫米；随机方向漂移只有 20–30 % 的轨迹经过可抓区）：
- 出生：在可抓区里选一个目标点（x 0.24–0.29、y ±0.10）和来的方向（从 +x 方向 ±50°），初速度在 v_range 内随机
  （默认 0–8 cm/s），按 速度 × 到达时间（1.5–4 s）往回推出生点（限制在 x 0.24–0.80、y ±0.17）；漂移方向对准目标点 ±10°；
  船头大致朝运动方向（±30°，慢船随机），船体离墙至少 2 cm，角速度 ±0.1 rad/s。机械臂从 home 出发、夹爪张开。
- 相机模型：位置噪声 1 mm、朝向噪声 0.5°，延迟 1 帧（50 ms）。
- 动作 (5,)：同阶段 2：(dx, dy, dz) × 1 cm、d_roll × 0.1 rad、d_grip × 0.1 rad（每步 1 cm = 最快 0.2 m/s）。
- 成功：夹住（两指都碰到桅杆、夹爪合不拢），并且船速 < 2 cm/s，连续 10 步（0.5 s）。
- 失败：夹住之前船撞到缸壁或隔板（回合结束，-10）；超时 200 步（10 s）。
- 奖励（每步都 ≤ 0，只有成功为正）：-（TCP 到真实夹持点的距离）- 0.1·|转角误差| - 0.5 ×（2 - 碰到桅杆的手指数）
  - 1（没夹住）- 0.5·|TCP 与桅杆的相对速度|（离桅杆 5 cm 以内时，鼓励"同速跟上"）- 0.01·|动作|²
  - 1（机械臂撞缸壁 / 隔板 / 平台 / 船体）- (2 + 100·撞前船速)（船每撞一次墙，不结束回合，船反弹后继续；
  反弹出可抓区后 BODY2 自己游回来：推力 ≤ 0.3 N，速度 max(初速度, 3 cm/s)，进可抓区后关推力）
  - 1/步（夹着或碰着船时船贴墙）- 20·船速增量（没夹住时手指把船推快）；成功的 0.5 s 静止期内船不能碰墙。
  成功（v3）：连续 0.5 s 夹着、不碰墙，且 0.5 s 内位移 < 1 cm（不看瞬时速度：夹住后船会振动）。
  - 0.5·|Δa|²（和上一步动作的差，夹爪那一维 ×0.2；v3 加，防止动作每步来回反向）。
  10-23 v2：v1 里撞墙只扣 2，SAC 学会了"夹住后拖船撞隔板当刹车"（静止船也平均撞 1.2 次），所以加重。- 夹住后：10·|高度偏差| + 0.5·|倾角|；成功 +20。
- 观测里有船的高度偏差和横倾 / 纵倾（10-23）：高速夹住时桅杆在两指间被扭，船被夹成横倾 7–15°，
  浮力要扶正、夹爪不放，船在夹爪下来回振动（竖直速度 ±16 cm/s），船速一直降不到 2 cm/s。
"""
from collections import deque
import numpy as np
from gymnasium import spaces
import mujoco

from envs.grasp_env import GraspEnv, GRIP_STEP, GRIP_DEPTH
from envs.reset_env import MAST_TOP, GRIP_OPEN, ROLL_STEP, WS_LO, WS_HI, wrap_half_pi
from envs.kalman import BoatKF, wrap_pi
from envs.ik import smooth_tcp_step

BOAT_Z_EQ = -0.04 - 0.0168 + 0.025
HORIZONS = (0.25, 0.5, 1.0)
WALL_GEOMS = ["tank_wall_back", "tank_wall_front", "tank_wall_yneg", "tank_wall_ypos", "fence", "mount_plate"]
HOLD_STEPS = 10
HOVER_GP = 0.05                      # 脚本拦截器接近时 TCP 在夹持点上方的高度（= 桅杆顶上方 3.5 cm，和阶段 1 一样）。
                                     # 10-23：曾用 6.5 cm 避免扫到桅杆（当时 z 振荡）；底层振荡修好后降回来，6.5 cm 时 x ≈ 0.30 m 的船竖直够不到
# 顶部相机遮挡模型（10-24）：相机在缸中央正上方 1 m；AprilTag（4 × 4 cm）贴在甲板上、桅杆前方 7 cm（船头一侧）。
# 从相机向标签中心和 4 个角各打一条光线，被机械臂挡住任意一条 = 这一帧识别不到（AprilTag 要看到完整四边形）。
TOP_CAM = np.array([0.44, 0.0, 1.0])
TAG_LOCAL = np.array([0.07, 0.0, 0.0251])                 # 单标签时的位置（甲板面 = 船体半高 0.025），已不用
# 10-24：真机船头、船尾各贴一个（ID 不同，不会认错头尾），甲板上距中心 ±9 cm。任意一个完整可见就有读数
TAG_LOCALS = [np.array([0.09, 0.0, 0.0251]), np.array([-0.09, 0.0, 0.0251])]
TAG_HALF = 0.02
# 腕部相机（10-24）：模型自带的 wrist_cam（装在夹爪上的 camera_mount，竖直视场 61°，按 4:3 画幅）。
# 测桅杆顶部中心的位置和桅杆朝向（扁长方体分不出船头船尾 → 朝向只知道对 180° 取模）；离得近，比顶部相机准。
WRIST_SIGMA_POS, WRIST_SIGMA_YAW, WRIST_RANGE = 0.0005, np.radians(1.0), 0.35
AGE_CAP = 20                                              # "距上次有读数的帧数"的上限（观测里除以它归一化）
ZONE_LO, ZONE_HI = np.array([0.22, -0.15]), np.array([0.30, 0.15])   # 可抓区（船体中心）：桅杆在手臂够得着、船体放得下的范围
                                     # 10-23：x 上限原为 0.38；夹爪竖直、在悬停高度时手臂只够到约 0.30 m
SWIM_TAU, SWIM_FMAX = 0.3, 0.3                        # 游回可抓区：速度跟踪时间常数 0.3 s，推力上限 0.3 N
N_HIST = 4


def _hull_extent(yaw):
    """船体（半长 0.131、半宽 0.058）在世界 x / y 方向上伸出的半宽。"""
    c, sn = abs(np.cos(yaw)), abs(np.sin(yaw))
    return 0.131 * c + 0.058 * sn, 0.131 * sn + 0.058 * c


class InterceptEnv(GraspEnv):
    def __init__(self, predictor="kf", v_range=(0.0, 0.08), max_steps=200, noise=True, latency=1, noise_range=None,
                 occlusion=False, wrist=False, action_filter=0.0, two_tags=True, rate_limit=0.0, smooth_penalty=0.5, **kw):
        super().__init__(max_steps=max_steps, **kw)
        self.predictor, self.v_range, self.noise, self.latency = predictor, v_range, noise, latency
        self.sigma_pos, self.sigma_yaw = 0.001, np.radians(0.5)   # 相机测量噪声（AprilTag：1 mm、0.5°）；评估抗干扰时可以调大
        # 10-23：每回合随机相机噪声（例如 (0.001, 0.003)）。KF 按当回合的噪声配置（真机上相机噪声可以标定），
        # 突变门限 = 6 σ（v2 评估里门限固定 6 mm、噪声 3 mm 时 KF 把噪声当成突变，A 方案成功率掉到 2 %）
        self.noise_range = noise_range
        # 动作低通滤波（10-24）：实际执行 = α·上一步执行的 + (1-α)·策略输出（TCP 位移和转腕 4 维；夹爪不滤，开合要果断）。
        # SAC 的动作相邻两步平均变化是脚本的 3 倍，动作变化惩罚只压下一半；滤波后物理上不可能每步来回反向。
        # 观测里的"上一步动作"是实际执行的那个，策略能看到滤波器的状态
        self.action_filter = action_filter
        self.smooth_penalty = smooth_penalty
        self.block_outward = True                             # 边界处不往外推（envs/ik.py::smooth_tcp_step）；v4 之前训练的模型可以关掉                  # 奖励里动作变化惩罚的系数：-smooth_penalty·|Δa|²（默认 0.5；stage9 用 2.0）
        self.rate_limit = rate_limit                          # >0：执行的动作每步最多变化这么多（TCP 位移和转腕 4 维），硬约束
        self.tag_locals = TAG_LOCALS if two_tags else [TAG_LOCAL]   # 船头船尾两个标签（默认）；False = 只有桅杆前一个（stage7 的设定）
        self.occlusion = occlusion                            # True：顶部相机会被机械臂挡住（见 _tag_visible）
        self.wrist = wrist                                    # True：加腕部相机（见 _wrist_measure）
        self.ext_obs = occlusion or wrist                     # 扩展观测（两个相机的"距上次读数帧数"/腕部读数），维数变了要重训
        self.top_age, self.wrist_age, self.wrist_last = 0, AGE_CAP, None   # 父类 reset 会先读一次观测，先给初值
        self.ramp_ctrl = True                                 # 舵机目标在每个控制步内线性过渡（见 step）
        m = self.m
        self.walls = {m.geom(n).id for n in WALL_GEOMS}
        self.boat_mass = float(m.body_subtreemass[self.bid])
        self.swim_target = None
        for g in self.walls:
            m.geom_solmix[g] = 1000.0                         # 墙和船接触时用墙的 solref（MuJoCo 按 solmix 加权混合两边参数）
        self.boat_geoms = {m.geom("body2_hull").id, m.geom("body2_mast").id}
        self.kf = BoatKF()
        self.kf.reset((0.0, 0.0), 0.0)                       # 父类 reset 里会先调用一次 _obs()，要先有这些
        self.hist = deque([(np.zeros(2), 0.0)] * N_HIST, maxlen=N_HIST)
        n_obs = (39 + 2 * (occlusion or wrist)) if predictor == "kf" else (43 + 6 * (occlusion or wrist))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    # ---------- 相机测量 ----------
    def _measure(self):
        """当前真实位姿 + 噪声（还没加延迟）；开了遮挡模型且标签被挡住时返回 None（这一帧没有测量）。"""
        if self.occlusion and not self._tag_visible():
            return None
        p = self.d.xpos[self.bid][:2].copy()
        yaw = self._boat_yaw()
        if self.noise:
            p += self.np_random.normal(0, self.sigma_pos, 2)
            yaw = wrap_pi(yaw + self.np_random.normal(0, self.sigma_yaw))
        return p, yaw

    def _one_tag_visible(self, local):
        """顶部相机能不能完整看到这一个 AprilTag（中心和 4 个角都不被机械臂的可视几何体 group 2 挡住）。"""
        R = self.d.xmat[self.bid].reshape(3, 3); c = self.d.xpos[self.bid]
        mask = np.array([0, 0, 1, 0, 0, 0], np.uint8)
        gid = np.zeros(1, np.int32)
        for dx, dy in [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            pt = c + R @ (local + np.array([dx * TAG_HALF, dy * TAG_HALF, 0.0]))
            t = mujoco.mj_ray(self.m, self.d, TOP_CAM, pt - TOP_CAM, mask, 1, -1, gid)
            if 0 <= t < 0.999:                                # 在到达标签之前打到了机械臂
                return False
        return True

    def tags_visible(self):
        """每个标签是否可见（GIF 画图也用）。"""
        return [self._one_tag_visible(t) for t in self.tag_locals]

    def _tag_visible(self):
        """至少一个标签完整可见 → 这一帧有读数（两个标签 ID 不同，任意一个都给出完整位姿）。"""
        return any(self.tags_visible())

    def _wrist_measure(self):
        """腕部相机：桅杆顶部中心在视场内、距离 < WRIST_RANGE 时返回 (xy, 朝向 mod π) + 噪声，否则 None。"""
        if not self.wrist:
            return None
        cid = self.m.camera("wrist_cam").id
        cpos, cR = self.d.cam_xpos[cid], self.d.cam_xmat[cid].reshape(3, 3)
        top = self.d.xpos[self.bid] + self.d.xmat[self.bid].reshape(3, 3) @ np.array([0.0, 0.0, MAST_TOP])
        v = cR.T @ (top - cpos)                               # 相机坐标系：看向 -z，y 朝上
        if v[2] >= 0 or np.linalg.norm(v) > WRIST_RANGE:
            return None
        tv = np.tan(np.radians(self.m.cam_fovy[cid]) / 2); th = tv * 4 / 3
        if abs(v[0] / -v[2]) > th or abs(v[1] / -v[2]) > tv:
            return None
        p = top[:2].copy(); yaw = self._boat_yaw()
        if self.noise:
            p += self.np_random.normal(0, WRIST_SIGMA_POS, 2)
            yaw += self.np_random.normal(0, WRIST_SIGMA_YAW)
        return p, float(wrap_pi(2 * yaw) / 2)                 # 朝向只知道对 180° 取模

    def _lag(self):
        """延迟补偿（10-24，"延迟状态滤波"）：扩展观测模式下 KF 运行在读数拍下的时刻（latency 帧以前），读数直接用来更新；
        给策略 / 脚本的"当前估计"只在输出时按匀速往前推 latency 帧，不回到滤波器里，没有反馈。
        （试过两种把读数往前推的写法：读数 + 速度 × 延迟、读数 + KF 这几帧的位移，延迟 3 帧时都会形成反馈、KF 发散到无穷大。）"""
        return self.latency * 0.05 if self.ext_obs else 0.0

    def _kf_now(self):
        """KF 估计往前推到"现在"：(位置, 朝向)。"""
        lag = self._lag()
        return self.kf.pos + self.kf.vel * lag, float(wrap_pi(self.kf.yaw + self.kf.yaw_rate * lag))

    def _camera(self):
        """把这一帧放进延迟队列，返回 latency 帧之前的测量。"""
        self.meas_q.append(self._measure())
        return self.meas_q[0]

    # ---------- 用估计值算的量（观测用；奖励用真实值） ----------
    def _close_dir(self):
        c = self.d.xmat[self.gbody].reshape(3, 3) @ self.close_dir_local
        c[2] = 0.0
        return c / np.linalg.norm(c)

    def _gap_shift(self):
        gap = max(0.0046, 0.0163 + 0.076 * float(self.d.qpos[self.grip_qa]))
        return max(0.0, gap / 2 - 0.006)

    def _grip_point_from(self, xy):
        """给定桅杆的水平位置，算 TCP 应该去的点（让桅杆在两指空隙中间；见 GraspEnv._grip_point）。"""
        z = BOAT_Z_EQ + MAST_TOP - GRIP_DEPTH
        return np.r_[xy, z] - self._gap_shift() * self._close_dir()

    def _angle_err_from(self, yaw):
        c = self._close_dir()
        return float(wrap_half_pi(np.arctan2(c[1], c[0]) - (yaw + np.pi / 2)))

    def _time_to_wall(self, pos, vel, yaw):
        """沿当前速度直线走，多久碰到缸壁或隔板（上限 5 s）。
        船体是 26 × 11.6 cm 的长方形：按当前朝向算它在 x / y 方向伸出多少（10-23：原来固定按 6 cm 算，船头朝隔板时偏乐观 7 cm）。"""
        ex, ey = _hull_extent(yaw)
        lim = [(0.085 + ex, 0.94 - ex), (-0.25 + ey, 0.25 - ey)]
        t = 5.0
        for i, (lo, hi) in enumerate(lim):
            if vel[i] > 1e-4:
                t = min(t, max(0.0, (hi - pos[i]) / vel[i]))
            elif vel[i] < -1e-4:
                t = min(t, max(0.0, (lo - pos[i]) / vel[i]))
        return t

    def _tilt(self):
        """船的高度偏差和横倾 / 纵倾（AprilTag 的 6 维位姿里就有；夹住后要\"托着船\"，不让它被夹歪、压沉）。"""
        R = self.d.xmat[self.bid].reshape(3, 3)
        return np.array([self.d.xpos[self.bid][2] - BOAT_Z_EQ, np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1))])

    def _obs(self):
        q = self.d.qpos[self.arm_qa]; qd = self.d.qvel[self.arm_va]; tcp = self.ik.tcp(self.d)
        grip_err = self.grip_cmd - float(self.d.qpos[self.grip_qa])
        tilt = self._tilt()
        if self.predictor == "kf":
            kf = self.kf
            pos_now, yaw_now = self._kf_now()
            gp = self._grip_point_from(pos_now)
            e = self._angle_err_from(yaw_now)
            preds = np.concatenate([kf.predict(t + self._lag())[0] - tcp[:2] for t in HORIZONS])
            return np.concatenate([q, qd, tcp, gp - tcp, kf.vel, [kf.yaw_rate], [np.sin(2 * e), np.cos(2 * e)], preds,
                                   [self._time_to_wall(pos_now, kf.vel, yaw_now) / 5.0], self.prev_action, [grip_err], tilt]
                                  + ([[self.top_age / AGE_CAP, self.wrist_age / AGE_CAP]] if self.ext_obs else [])).astype(np.float32)
        hist = list(self.hist)                                # 最近 N_HIST 帧原始测量（旧 → 新）
        p_now, y_now = hist[-1]
        gp = self._grip_point_from(p_now)
        rel = np.concatenate([p - tcp[:2] for p, _ in hist])
        ang = np.concatenate([[np.sin(2 * self._angle_err_from(y)), np.cos(2 * self._angle_err_from(y))] for _, y in hist])
        ext = []
        if self.ext_obs:                                      # 顶部相机这一帧有没有读数 + 腕部相机最新读数（相对 TCP）和是否新鲜
            if self.wrist_last is not None:
                ew = self._angle_err_from(self.wrist_last[1])
                wr = [self.wrist_last[0] - tcp[:2], [np.sin(2 * ew), np.cos(2 * ew)]]
            else:
                wr = [np.zeros(2), [0.0, 1.0]]
            ext = [[float(self.top_age == 0)], *wr, [float(self.wrist_age == 0)]]
        return np.concatenate([q, qd, tcp, gp - tcp, rel, ang, self.prev_action, [grip_err], tilt] + ext).astype(np.float32)

    # ---------- Gymnasium 接口 ----------
    def reset(self, seed=None, options=None):
        super(GraspEnv, self).reset(seed=seed)                # ReachEnv.reset：机械臂回 home、夹爪张开（船位置下面重新放）
        rng = self.np_random
        e = self
        for _ in range(100):
            mujoco.mj_resetDataKeyframe(e.m, e.d, e.key_home)
            e.d.qpos[e.grip_qa] = GRIP_OPEN; e.d.ctrl[e.grip_act] = GRIP_OPEN
            # 出生点按"多久后到达可抓区"倒推（10-23：直接在远端出生时，慢船 10 s 内根本到不了可抓区，任务不可能完成）：
            # 先选可抓区里的目标点和来的方向（大致从 +x 方向来，±50°），再按 速度 × 到达时间（1.5–4 s）往回推
            v = rng.uniform(*self.v_range)
            tx, ty = rng.uniform(0.24, 0.29), rng.uniform(-0.10, 0.10)   # 10-23：0.18–0.28 太靠隔板；> 0.29 竖直够不到
            come = np.radians(rng.uniform(-50, 50))           # 从哪个方向来（0 = 从 +x 来）
            back = max(v * rng.uniform(1.5, 4.0), 0.0)
            x = float(np.clip(tx + back * np.cos(come), 0.24, 0.80))   # 下限原为 0.30：静止的船被卡在可达范围外
            y = float(np.clip(ty + back * np.sin(come), -0.17, 0.17))
            ang = np.arctan2(ty - y, tx - x) + np.radians(rng.uniform(-10, 10)) if back > 0.01 else rng.uniform(-np.pi, np.pi)
            # BODY2 是自己游的：船头大致朝运动方向（±30°）；慢船（几乎不动）朝向随机。
            # 10-23：原来朝向完全随机，横着漂的船贴着侧壁一转就撞墙
            yaw = ang + np.radians(rng.uniform(-30, 30)) if back > 0.01 else rng.uniform(-np.pi, np.pi)
            ex, ey = _hull_extent(yaw)
            if not (0.085 + ex + 0.02 < x < 0.94 - ex - 0.02 and abs(y) < 0.25 - ey - 0.02):
                continue                                      # 出生时船体离墙 / 隔板至少 2 cm
            e.d.qpos[e.bqa:e.bqa + 7] = [x, y, BOAT_Z_EQ, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(e.m, e.d)
            if not any(c.geom1 in e.body2_geoms or c.geom2 in e.body2_geoms for c in e.d.contact[:e.d.ncon]):
                break
        e.d.qvel[e.bva:e.bva + 2] = [v * np.cos(ang), v * np.sin(ang)]
        e.d.qvel[e.bva + 5] = rng.uniform(-0.1, 0.1)         # 绕船体 z 轴的初始角速度
        # 撞墙反弹的"弹性"每回合随机（10-23）：阻尼比 0.3–1.0 ≈ 恢复系数 0.1–0.6（实测），真实的塑料船撞玻璃缸是多少不知道
        self.wall_dampratio = float(rng.uniform(0.3, 1.0))
        for g in self.walls:
            e.m.geom_solref[g] = [0.02, self.wall_dampratio]
        self.in_wall = False; self.wall_hits = 0; self.prev_speed = float(v)
        self.swim_target = None; self.swim_speed = 0.0; self.reaims = 0
        self.q_cmd = e.d.qpos.copy(); self.tcp_cmd = self.ik.tcp(e.d).copy(); self.roll_cmd = float(e.d.qpos[e.roll_qa])
        self.limit_dir = None
        self.grip_cmd = GRIP_OPEN
        if self.noise_range is not None:
            self.sigma_pos = float(rng.uniform(*self.noise_range))
            self.sigma_yaw = np.radians(0.5) * self.sigma_pos / 0.001
            self.kf = BoatKF(sigma_pos=self.sigma_pos, sigma_yaw=self.sigma_yaw, gate=6 * self.sigma_pos)
        occ, self.occlusion = self.occlusion, False            # 开始时手臂在 home、标签可见
        self.meas_q = deque(maxlen=self.latency + 1)
        for _ in range(self.latency + 1):
            self.meas_q.append(self._measure())
        self.occlusion = occ
        self.n_occluded = 0
        self.wq = deque([None] * (self.latency + 1), maxlen=self.latency + 1)   # 腕部相机的延迟队列
        self.top_age, self.wrist_age, self.wrist_last, self.n_wrist = 0, AGE_CAP, None, 0
        p0, y0 = self.meas_q[0]
        self.kf.reset(p0, y0)
        self.hist = deque([(p0, y0)] * N_HIST, maxlen=N_HIST)
        self.hold = 0; self.steps = 0; self.prev_action = np.zeros(5)
        self.pos_hist = deque(maxlen=HOLD_STEPS + 1)          # 最近 0.5 s 船的位置（判断"停稳"用位移，不用瞬时速度）
        self.prev_tcp = self.ik.tcp(e.d).copy()
        self.boat_xy0 = e.d.xpos[e.bid][:2].copy()
        self.v0 = v
        return self._obs(), {}

    def _physics_step(self):
        mujoco.mj_step1(self.m, self.d)
        self.water.apply(self.d)                              # 水的力（会覆盖 xfrc_applied）
        if getattr(self, "swim_target", None) is not None:    # BODY2 自己游：推力把水平速度拉向目标速度（加在水的力之后）
            pos, v = self.d.xpos[self.bid][:2], self.d.qvel[self.bva:self.bva + 2]
            d = self.swim_target - pos
            v_des = d / (np.linalg.norm(d) + 1e-9) * self.swim_speed
            F = self.boat_mass * (v_des - v) / SWIM_TAU
            F *= min(1.0, SWIM_FMAX / (np.linalg.norm(F) + 1e-9))
            self.d.xfrc_applied[self.bid, :2] += F
        mujoco.mj_step2(self.m, self.d)

    def _update_swim(self, touched):
        """反弹出可抓区后，船重新游回可抓区（10-23）。"""
        pos, v = self.d.xpos[self.bid][:2], self.d.qvel[self.bva:self.bva + 2]
        inside = bool(np.all(pos > ZONE_LO) and np.all(pos < ZONE_HI))
        if touched or inside:
            self.swim_target = None                           # 被夹爪碰到 / 已经回到可抓区：关推力，自由漂
            return
        if self.swim_target is None and self.wall_hits > 0:
            center = (ZONE_LO + ZONE_HI) / 2
            toward = float(v @ (center - pos) / (np.linalg.norm(center - pos) + 1e-9))
            speed = max(self.v0, 0.03)
            if toward < 0.5 * speed:                          # 没在往可抓区走（被弹开 / 贴墙滑 / 停住）
                self.swim_target = np.array([self.np_random.uniform(0.24, 0.32), self.np_random.uniform(-0.10, 0.10)])
                self.swim_speed = speed
                self.reaims += 1

    def kf_obs(self):
        """不管 predictor 是什么，都返回 KF 版观测（卡尔曼滤波每步都在后台跑）。
        方案 B 的示范用它：脚本拦截器看 KF 观测出动作，回放池里存的是 B 的原始测量观测，A / B 示范动作完全相同。"""
        keep, self.predictor = self.predictor, "kf"
        o = self._obs()
        self.predictor = keep
        return o[:39]                                         # 脚本拦截器只用前 39 维

    def _boat_hits_wall(self):
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.boat_geoms and c.geom2 in self.walls) or (c.geom2 in self.boat_geoms and c.geom1 in self.walls):
                return True
        return False

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_filter:
            a[:4] = self.action_filter * self.prev_action[:4] + (1 - self.action_filter) * a[:4]
        if self.rate_limit:
            a[:4] = self.prev_action[:4] + np.clip(a[:4] - self.prev_action[:4], -self.rate_limit, self.rate_limit)
        self.roll_cmd = float(np.clip(self.roll_cmd + a[3] * ROLL_STEP, self.roll_lo, self.roll_hi))
        self.grip_cmd = float(np.clip(self.grip_cmd + a[4] * GRIP_STEP, self.grip_lo, GRIP_OPEN))
        self.q_cmd[self.roll_qa] = self.roll_cmd
        smooth_tcp_step(self, a[:3] * self.step_size, WS_LO, WS_HI, self.block_outward)   # IK 线搜索 + 防积分饱和 + 边界处不往外推（envs/ik.py）
        ctrl0 = self.d.ctrl.copy()
        self.d.ctrl[self.ik.qadr] = self.q_cmd[self.ik.qadr]
        self.d.ctrl[self.roll_act] = self.roll_cmd
        self.d.ctrl[self.grip_act] = self.grip_cmd
        ctrl1 = self.d.ctrl.copy()
        collided = wall = False
        for i in range(self.n_sub):
            # 舵机目标在 50 ms 内从旧值线性过渡到新值（10-23 v3）。原来每 50 ms 跳一次：SO-101 的舵机模型欠阻尼
            # （kp 998、kv 2.7，阻尼比约 0.2，约 20 Hz），每次阶跃都冲过头再弹回，关节速度每步变号，看起来就是抖
            if self.ramp_ctrl:
                self.d.ctrl[:] = ctrl0 + (ctrl1 - ctrl0) * (i + 1) / self.n_sub
            self._physics_step()
            collided |= self._arm_collision()
            wall |= self._boat_hits_wall()
        self.steps += 1
        da = a - self.prev_action                             # 和上一步动作的差（抖动 = 每步来回反向）
        da2 = float(da[:4] @ da[:4]) + 0.2 * float(da[4] ** 2)  # 夹爪开合本来就是突变，权重小
        self.prev_action = a
        # 相机 → 卡尔曼滤波 / 历史
        z = self._camera()
        occluded = z is None
        if occluded:                                          # 丢帧：KF 只预测不更新；历史缓存重复上一次的有效测量
            self.n_occluded += 1
            self.kf.step(None, None)
            self.hist.append(self.hist[-1])
        else:
            self.kf.step(*z)                                  # 扩展观测模式下 KF 本来就在读数的时刻上（见 _lag）
            self.hist.append(z)
        self.top_age = min(self.top_age + 1, AGE_CAP) if occluded else 0
        self.wq.append(self._wrist_measure())
        zw = self.wq[0]
        if zw is not None:                                    # 腕部相机：同一时刻再更新一次 KF（朝向对 180° 取模）
            self.kf.update(zw[0], zw[1], sigma_pos=WRIST_SIGMA_POS, sigma_yaw=WRIST_SIGMA_YAW, gate=0.006, half_turn=True)
            self.wrist_last, self.wrist_age = zw, 0
            self.n_wrist += 1
        else:
            self.wrist_age = min(self.wrist_age + 1, AGE_CAP)
        if self.ext_obs and self.grip_cmd - float(self.d.qpos[self.grip_qa]) < -0.02:
            # 本体感觉（10-24）：夹爪指令比实际更合、合不拢 = 桅杆在两指之间。桅杆位置 ≈ TCP（关节角 + 正运动学，无延迟），
            # 桅杆长边 ⟂ 合拢方向（朝向对 180° 取模）。夹住后两个相机都常常看不到船，没有它 KF 会按原速度一直往前推、误差到几十厘米
            cd = self._close_dir()
            lag = self._lag()                                 # 本体感觉没有延迟：往回推到滤波器所在的时刻
            self.kf.update(self.ik.tcp(self.d)[:2] - self.kf.vel * lag, float(np.arctan2(cd[1], cd[0]) + np.pi / 2) - self.kf.yaw_rate * lag,
                           sigma_pos=0.003, sigma_yaw=np.radians(3), gate=1.0, half_turn=True)
        # 真实状态（奖励、成功判断用）
        tcp = self.ik.tcp(self.d)
        rel = self._grip_point() - tcp
        dist, e = float(np.linalg.norm(rel)), self._angle_err()
        f_touch, m_touch = self._finger_contacts()
        grasped = f_touch and m_touch and (float(self.d.qpos[self.grip_qa]) - self.grip_cmd > 0.03)
        boat_v = self.d.qvel[self.bva:self.bva + 2]
        speed = float(np.linalg.norm(boat_v))
        tcp_v = (tcp[:2] - self.prev_tcp[:2]) / (self.n_sub * self.m.opt.timestep)
        self.prev_tcp = tcp.copy()
        rel_v = float(np.linalg.norm(tcp_v - boat_v))
        tilt = self._tilt()
        impact = wall and not self.in_wall                    # 新的一次撞墙（上一步没贴着墙）
        self.in_wall = wall
        self.wall_hits += int(impact)
        self._update_swim(touched=bool(f_touch or m_touch))
        # 停稳 = 连续 0.5 s 夹着、不碰墙，且这 0.5 s 内船的位移 < 1 cm（平均速度 < 2 cm/s）。
        # 10-23 v3：原来要求每一步瞬时速度 < 2 cm/s；快船夹住后船在夹爪里振动，位置几乎不动但瞬时速度在 0–9 cm/s 乱跳，
        # 计数一再归零（残差 SAC 在 15–20 cm/s 档的失败几乎都是这个）。靠着墙停下仍然不算。
        self.pos_hist.append(self.d.xpos[self.bid][:2].copy())
        self.hold = self.hold + 1 if (grasped and not wall) else 0
        moved = float(np.linalg.norm(self.pos_hist[-1] - self.pos_hist[0])) if len(self.pos_hist) > HOLD_STEPS else 1.0
        success = self.hold >= HOLD_STEPS and moved < 0.01
        reward = (-dist - 0.1 * abs(e) - 0.5 * (2 - f_touch - m_touch) - (0.0 if grasped else 1.0)
                  - (0.5 * rel_v if dist < 0.05 else 0.0) - 0.01 * float(a @ a) - (1.0 if collided else 0.0)
                  - self.smooth_penalty * da2                         # 动作变化惩罚（10-23 v3：没有它 SAC 学出 10 Hz 来回抖动）
                  - (10.0 * abs(tilt[0]) + 0.5 * float(np.hypot(tilt[1], tilt[2])) if grasped else 0.0)
                  - ((2.0 + 100.0 * self.prev_speed) if impact else 0.0)          # 撞墙：按撞前船速加重（15 cm/s ≈ −17）
                  - (1.0 if wall and (grasped or f_touch or m_touch) else 0.0)     # 夹着 / 碰着船时船贴墙：每步 −1
                  - (20.0 * max(0.0, speed - self.prev_speed) if (f_touch or m_touch) and not grasped else 0.0)  # 没夹住就把船推快
                  + (20.0 if success else 0.0))
        self.prev_speed = speed
        terminated = bool(success)                            # 10-23：撞墙不再结束回合，船反弹后继续追
        truncated = self.steps >= self.max_steps
        info = dict(dist=dist, angle_err_deg=float(np.degrees(e)), grasped=grasped, boat_speed=speed, rel_v=rel_v,
                    wall=impact, wall_hits=self.wall_hits, reaims=self.reaims, collided=collided, success=bool(success), is_success=bool(success),
                    kf_pos_err=float(np.linalg.norm(self._kf_now()[0] - self.d.xpos[self.bid][:2])),
                    kf_vel_err=float(np.linalg.norm(self.kf.vel - boat_v)), v0=self.v0, da2=da2, occluded=occluded, n_occluded=self.n_occluded, n_wrist=self.n_wrist)
        return self._obs(), reward, terminated, truncated, info


def scripted_intercept(obs):
    """手写拦截器（只用卡尔曼观测，predictor="kf"）：
    预测拦截点 + 船速前馈 → 在上方 3.5 cm 跟上、转腕对准 → 下降到夹持点 → 合拢 → 夹住后手臂停住，让船慢慢停下。"""
    gp_rel, v, yaw_rate = obs[15:18], obs[18:20], obs[20]
    e = 0.5 * np.arctan2(obs[21], obs[22])
    grip_err, q_grip, dz = obs[35], obs[5], obs[36]
    a = np.zeros(5)
    ff = v * 0.05 / 0.01                                      # 速度前馈：船每步走 v·50 ms，换算成动作单位（1 = 1 cm）
    if grip_err < -0.02:                                      # 已经夹住（指令合拢、实际合不拢）
        # 不急停：继续跟着船走，每步只给 80 % 的船速 → 约 1 s 内平稳减速；
        # 同时把船慢慢拉回缸中间（≤ 1.5 cm/s，10-23 v2：靠着墙停下不算成功）
        pull = 0.5 * (np.array([0.30, 0.0]) - obs[12:14])
        pull *= min(1.0, 0.015 / (np.linalg.norm(pull) + 1e-9))
        a[0:2] = np.clip(0.8 * ff + pull * 0.05 / 0.01, -1, 1)
        a[2] = np.clip(-dz / 0.03, -0.5, 0.5)                 # 托着船：保持自然漂浮高度（10-23：急停会把船夹歪 7–15° 并引起振动）
        a[4] = -1.0
        return a
    tau = min(np.linalg.norm(gp_rel[:2]) / 0.15, 1.0)         # 提前量：按 0.15 m/s 追过去要多久
    aim = gp_rel[:2] + v * tau
    tcp_v = obs[30:32] * 0.01 / 0.05                          # 上一步的动作 = 指令 TCP 速度（m/s）
    matched = np.linalg.norm(tcp_v - v) < 0.015               # "同速跟上"：相对速度 < 1.5 cm/s
    aligned = np.linalg.norm(gp_rel[:2]) < 0.008 and abs(e) < np.radians(6)
    closing = q_grip < 0.33
    # 增益（10-23 v3 减半）：每步只补 1/4 的水平误差、1/2 的角度误差。原来补 1/2 和全部，加上相机 1 帧延迟和手臂跟随滞后，
    # 会冲过头再往回补，动作和关节速度每步反向（10 Hz 抖动），SAC 的示范和残差底子也继承了这个毛病
    a[0:2] = np.clip(aim / 0.04 + ff, -1, 1)
    a[3] = np.clip(-0.5 * e / ROLL_STEP + yaw_rate * 0.05 / ROLL_STEP, -1, 1)
    if aligned or closing:
        a[2] = np.clip(gp_rel[2] / 0.04, -0.6, 0.6)           # 对准了才下降（每步补 1/4，避免 z 振荡）
        urgent = obs[29] * 5.0 < 0.8                          # 不到 0.8 s 就撞墙：不等同速，够近就合拢
        a[4] = -1.0 if (closing or (np.linalg.norm(gp_rel) < 0.007 and (matched or urgent))) else 1.0   # 同速了才合拢（10-23）
    else:
        # 没对准：保持在夹持点上方 HOVER_GP。10-23：最早只高 3.5 cm（桅杆顶上方 2 cm），
        # 张开的活动夹爪最低点在 TCP 下方约 1.75 cm，再加上 z 振荡，接近时会扫到桅杆，把静止的船一下踢到 22 cm/s 撞墙
        a[2] = np.clip((gp_rel[2] + HOVER_GP) / 0.04, -0.6, 0.6)
        a[4] = 1.0
    return a
