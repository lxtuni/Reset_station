"""SO-101 的数值逆运动学 (IK)：阻尼最小二乘 (DLS)，让夹爪尖端到目标点，同时夹爪竖直朝下。

原理（详见对话里的图）：
  ① 正运动学算当前尖端位置 p 和夹爪轴线方向 a
  ② 误差 e = [p* - p ; w·(a × a*)]           a* = (0, 0, -1) 竖直朝下
  ③ 雅可比 J = [J_p ; w·J_r]（6 × 4）           只取 4 个关节的列
  ④ Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e                 λ = 阻尼，防止奇异位形时 Δq 爆掉
  ⑤ q ← clip(q + Δq, 关节限位)，回到 ①，直到误差够小 / 卡住 / 次数用完

只用 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex 四个关节；wrist_roll 不参与（单独控制夹爪绕竖直轴的转角），
gripper（夹爪开合）也不参与。IK 用自己的一份 MjData，不改动仿真正在用的数据。
不检查碰撞：解出来的姿态可能撞墙或隔板，这由 RL 环境的惩罚项负责。
"""
import numpy as np
import mujoco

IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


class IK:
    # 默认参数来自 10-21 的参数扫描（λ 0.005–0.05 × 步长上限 0.1/0.3 × 朝向权重 0.05/0.1，150 个已知可达目标）：
    # λ 0.02、步长上限 0.3 rad、权重 0.1 → 成功 97 %、离底座 > 0.28 m 的远点 91 %、迭代中位 10 次、约 0.7 ms
    def __init__(self, model, site="gripperframe", w_rot=0.1, damping=0.02, max_dq=0.3,
                 tol_pos=1e-3, tol_rot=np.radians(1.0), max_iter=100):
        self.m = model
        self.d = mujoco.MjData(model)                       # IK 专用的数据副本
        self.sid = model.site(site).id                      # 夹爪尖端那个点（site）的编号
        jids = [model.joint(n).id for n in IK_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])   # 这 4 个关节在 qpos 里的位置
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])    # 在 qvel / 雅可比列里的位置
        self.lo, self.hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]   # 关节限位（弧度）
        rj = model.joint("wrist_roll").id
        self.roll_body = model.jnt_bodyid[rj]               # wrist_roll 所在的刚体
        self.roll_axis = model.jnt_axis[rj].copy()          # wrist_roll 的转轴（在该刚体坐标系里）
        self.a_des = np.array([0.0, 0.0, -1.0])             # 目标：夹爪轴线竖直朝下
        self.w, self.lam, self.max_dq = w_rot, damping, max_dq
        self.tol_pos, self.tol_rot, self.max_iter = tol_pos, tol_rot, max_iter
        self.jacp = np.zeros((3, model.nv))                 # mj_jacSite 的输出：位置雅可比（3 × 全部自由度）
        self.jacr = np.zeros((3, model.nv))                 #                    转动雅可比

    def _fk(self):
        """正运动学：返回尖端位置 p 和夹爪轴线方向 a（世界坐标）。"""
        mujoco.mj_kinematics(self.m, self.d)                # 只算位置和朝向，不算力，很快
        mujoco.mj_comPos(self.m, self.d)                    # 再算各关节的运动方向 (cdof)：mj_jacSite 要用它，
                                                            # 少了这一句雅可比全是 0，IK 一步也走不动（第一次测试就踩到了）
        p = self.d.site_xpos[self.sid].copy()
        R = self.d.xmat[self.roll_body].reshape(3, 3)
        a = -(R @ self.roll_axis)                           # 夹爪朝外的方向 = 腕转轴的反方向（模型里的约定）
        return p, a

    def solve(self, qpos, p_target):
        """qpos：整个模型当前的 qpos（只读，用作起点）；p_target：尖端目标位置 (3,)。
        返回 (新的 qpos 副本, info)。info 里有 success、迭代次数、最终误差、停止原因。"""
        self.d.qpos[:] = qpos
        q = self.d.qpos[self.qadr].copy()
        for it in range(1, self.max_iter + 1):
            p, a = self._fk()
            e_p = p_target - p                              # 位置误差 (3,)
            e_r = np.cross(a, self.a_des)                   # 朝向误差 (3,)：方向 = 转轴，长度 ≈ 偏差角
            err_pos, err_rot = np.linalg.norm(e_p), np.arcsin(min(1.0, np.linalg.norm(e_r)))
            if err_pos < self.tol_pos and err_rot < self.tol_rot:
                return self.d.qpos.copy(), dict(success=True, iters=it - 1, err_pos=err_pos, err_rot=err_rot, reason="converged")
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp[:, self.vadr], self.w * self.jacr[:, self.vadr]])   # (6, 4)
            e = np.concatenate([e_p, self.w * e_r])                                      # (6,)
            # 关节限位处理（"夹紧"法 / clamping）：DLS 本身不知道限位。若某关节已在限位上、而算出的 Δq 还要它往外走，
            # 只靠事后 clip 会把这一步"吃掉"，其他关节也几乎不动 → 卡住（10-21 测试 A 里 7/12 个失败就是 wrist_flex 顶在 95°）。
            # 正确做法：把这个关节冻结（它那一列置 0），用剩下的关节重新解，让别的关节分担它的活。
            free = np.ones(4, bool)
            for _ in range(4):
                Jf = J * free                                                            # 冻结的关节那一列为 0
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + self.lam ** 2 * np.eye(6), e)    # 阻尼最小二乘
                push_out = ((q <= self.lo + 1e-6) & (dq < 0)) | ((q >= self.hi - 1e-6) & (dq > 0))
                if not (push_out & free).any():
                    break
                free &= ~push_out
            n = np.linalg.norm(dq)
            if n > self.max_dq:                             # 每步最多转 max_dq 弧度，保持线性近似有效
                dq *= self.max_dq / n
            q_new = np.clip(q + dq, self.lo, self.hi)       # 关节限位
            if np.linalg.norm(q_new - q) < 1e-6:            # 走不动了：目标够不到或顶到限位
                return self.d.qpos.copy(), dict(success=False, iters=it, err_pos=err_pos, err_rot=err_rot, reason="stalled")
            q = q_new
            self.d.qpos[self.qadr] = q
        p, a = self._fk()
        err_pos = np.linalg.norm(p_target - p); err_rot = np.arcsin(min(1.0, np.linalg.norm(np.cross(a, self.a_des))))
        ok = err_pos < self.tol_pos and err_rot < self.tol_rot
        return self.d.qpos.copy(), dict(success=ok, iters=self.max_iter, err_pos=err_pos, err_rot=err_rot, reason="converged" if ok else "max_iter")
