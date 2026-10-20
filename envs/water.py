"""最简单的水：给 BODY2 的船体加浮力和阻力（阶段 1–3 用；阶段 4 再换成 swimopt 的水动力模块）。

用法（每个仿真步，必须插在 mj_step1 和 mj_step2 之间）：
    mujoco.mj_step1(model, data)   # 用当前状态算好位置、速度
    water.apply(data)              # 算水的力，写进 data.xfrc_applied
    mujoco.mj_step2(model, data)   # 用这些力算加速度、推进一步
  不要写成 water.apply(data); mujoco.mj_step(...)：那样 apply 读到的是上一步的旧位置，
  浮力相当于延迟一步的弹簧，上下起伏会越来越大（试过，会把船弹出水面）。

模型假设（都是低保真近似，系数是先验，不是标定值）：
- 浮力：把船体切成 nx × ny × nz 个小体素，每个体素在水下的比例 × ρ g × 体素体积 = 它的浮力，
  作用在体素中心。船倾斜时浸在水下的体素分布会变，自然产生回正力矩；翻到很大角度也照样算对。
- 阻力：在船体自己的坐标系里，每个方向 F = -(c1·v + c2·|v|·v)，转动同理；乘以浸没比例。
- 没有附加质量（MuJoCo 不自带，见 MA 的 CLAUDE.md §8），没有波浪和自由液面。
"""
import numpy as np
import mujoco

RHO = 1000.0   # 水的密度 kg/m³
G = 9.81       # 重力加速度 m/s²


class Water:
    def __init__(self, model, water_z=-0.04, body="body2", hull_geom="body2_hull", nx=12, ny=6, nz=6):
        self.m = model
        self.water_z = water_z                       # 水面高度（世界坐标 z）
        self.bid = model.body(body).id               # BODY2 这个刚体的编号
        g = model.geom(hull_geom)
        L, B, H = 2 * g.size                         # 船体全长、全宽、全高（size 是半长）
        # 体素中心（body 坐标系）：在船体盒子里均匀铺 nx × ny × nz 个点
        xs = (np.arange(nx) + 0.5) / nx * L - L / 2
        ys = (np.arange(ny) + 0.5) / ny * B - B / 2
        zs = (np.arange(nz) + 0.5) / nz * H - H / 2
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.vox = np.c_[X.ravel(), Y.ravel(), Z.ravel()] + g.pos   # 加上船体相对 body 原点的偏移
        self.dV = L * B * H / self.vox.shape[0]      # 每个体素的体积
        self.dz = H / nz                             # 体素高度，用来让"入水比例"平滑过渡
        self.V = L * B * H                           # 船体总体积
        self.A_wp = L * B                            # 水线面积（算平均吃水用）
        # 阻力系数（先验）：顺序 [纵荡 surge, 横荡 sway, 垂荡 heave] 和 [横摇 roll, 纵摇 pitch, 艏摇 yaw]
        # surge c2：SA 结论"整机 0.2 m/s 阻力 77.6 mN" → 0.0776 / 0.2² ≈ 1.94
        self.c1_lin = np.array([0.05, 0.2, 10.0])      # N·s/m；heave 的线性阻尼让上下起伏几个周期内停下
        self.c2_lin = np.array([1.94, 5.0, 5.0])       # N·s²/m²
        self.c1_rot = np.array([0.01, 0.02, 0.002])    # N·m·s/rad
        self.c2_rot = np.array([0.0, 0.0, 0.005])      # N·m·s²/rad²
        self.last = {}                                 # 上一步的诊断量，方便打印

    def apply(self, d):
        b = self.bid
        R = d.xmat[b].reshape(3, 3)                  # body 坐标系 → 世界坐标系的旋转矩阵
        p0 = d.xpos[b]                               # body 原点的世界坐标
        com = d.xipos[b]                             # 质心的世界坐标（力矩要对它算）

        # ---- 浮力：逐个体素 ----
        pts = p0 + self.vox @ R.T                    # 每个体素中心的世界坐标，形状 (体素数, 3)
        wet = np.clip((self.water_z - pts[:, 2]) / self.dz + 0.5, 0.0, 1.0)   # 在水下的比例：中心在水面 → 0.5
        fz = RHO * G * self.dV * wet                 # 每个体素的浮力（竖直向上）
        F = np.array([0.0, 0.0, fz.sum()])
        r = pts - com                                # 质心 → 体素的向量
        T = np.c_[r[:, 1] * fz, -r[:, 0] * fz, np.zeros_like(fz)].sum(0)   # r × (0, 0, fz)

        # ---- 阻力：在船体坐标系里算，再转回世界 ----
        frac = float(wet.mean())                     # 浸没比例 0..1，出水时阻力变小
        vel = np.zeros(6)
        # 0 = 世界坐标系方向。注意不能用 1（"局部"）：对 body，MuJoCo 的"局部"指的是惯性主轴系 ximat，
        # 它和 body 坐标系 xmat 不一定一样（这个盒子的主轴就是 x、z 对调的），用错会让阻力不再耗能、反而注入能量
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        w_loc, v_loc = R.T @ vel[:3], R.T @ vel[3:]  # MuJoCo 的顺序：先角速度，后线速度；再转到船体坐标系
        f_loc = -(self.c1_lin * v_loc + self.c2_lin * np.abs(v_loc) * v_loc) * frac
        t_loc = -(self.c1_rot * w_loc + self.c2_rot * np.abs(w_loc) * w_loc) * frac
        F += R @ f_loc
        T += R @ t_loc

        # ---- 写进 MuJoCo：xfrc_applied[body] = [力 (3), 力矩 (3)]，世界坐标，作用在质心 ----
        d.xfrc_applied[b, :3] = F
        d.xfrc_applied[b, 3:] = T
        self.last = dict(buoyancy=float(fz.sum()), draft=frac * self.V / self.A_wp, speed=float(np.linalg.norm(v_loc)))
