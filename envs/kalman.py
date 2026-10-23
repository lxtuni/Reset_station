"""船的状态估计：卡尔曼滤波（Kalman filter），输入顶置相机 + AprilTag 测到的位置和朝向（带噪声、可能缺帧），
输出位置、速度、朝向、角速度的估计，并能往后预测。

运动模型：匀速（constant velocity）。位置 / 速度一组 4 维 [x, y, vx, vy]，朝向 / 角速度一组 2 维 [θ, ω]（角度单独处理回绕）。
水的阻力会让船慢慢减速，匀速模型不包含它，靠过程噪声 q 吸收；预测时可选一个指数衰减时间常数 tau_v 近似阻力。
"""
import numpy as np


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class BoatKF:
    def __init__(self, dt=0.05, sigma_pos=0.001, sigma_yaw=np.radians(0.5), q_acc=0.05, q_alpha=0.5, gate=0.006):
        self.dt = dt
        self.gate = gate                                      # 新息门限：位置偏差超过它就认为速度突变了
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.F = F
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.Q = G @ G.T * q_acc ** 2                         # 过程噪声：未建模的加速度（水阻力、碰撞）
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * sigma_pos ** 2                   # 测量噪声
        self.Fy = np.array([[1.0, dt], [0, 1.0]])
        gy = np.array([[dt ** 2 / 2], [dt]])
        self.Qy = gy @ gy.T * q_alpha ** 2
        self.Ry = sigma_yaw ** 2
        self.x = None

    def reset(self, pos, yaw):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.P = np.diag([1e-6, 1e-6, 0.01, 0.01])            # 一开始不知道速度：速度方差设大
        self.y = np.array([yaw, 0.0])
        self.Py = np.diag([1e-4, 0.25])
        self.jumps = 0

    def step(self, z_pos=None, z_yaw=None):
        """预测一步；有测量就更新。z_pos = (x, y)，z_yaw = θ；缺帧时传 None。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.y = self.Fy @ self.y
        self.Py = self.Fy @ self.Py @ self.Fy.T + self.Qy
        if z_pos is not None:
            innov = np.asarray(z_pos) - self.H @ self.x
            if np.linalg.norm(innov) > self.gate:             # 测量和预测差太多 = 速度突变（撞墙反弹、被夹爪碰到）
                self.P[2:, 2:] += np.eye(2) * 0.01            # 速度重新变"不确定"，几帧内就能重新估准（10-23）
                self.jumps += 1
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ innov
            self.P = (np.eye(4) - K @ self.H) @ self.P
        if z_yaw is not None:
            s = self.Py[0, 0] + self.Ry
            k = self.Py[:, 0] / s
            innov = wrap_pi(z_yaw - self.y[0])
            self.y = self.y + k * innov
            self.y[0] = wrap_pi(self.y[0])
            self.Py = self.Py - np.outer(k, self.Py[0, :])

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    @property
    def yaw(self):
        return float(self.y[0])

    @property
    def yaw_rate(self):
        return float(self.y[1])

    def predict(self, t, tau_v=None):
        """t 秒后的位置和朝向（匀速；给了 tau_v 就按速度指数衰减近似水阻力）。"""
        if tau_v:
            disp = self.vel * tau_v * (1 - np.exp(-t / tau_v))
        else:
            disp = self.vel * t
        return self.pos + disp, wrap_pi(self.yaw + self.yaw_rate * t)
