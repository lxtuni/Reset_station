"""阶段 3 的 Gymnasium 环境：Drag —— 夹着桅杆把船拖到终点、转到目标朝向，然后松开、抬起夹爪离开。

继承 GraspEnv（envs/grasp_env.py），复用动作执行、"夹住"判断、碰撞检测。
设计（10-22 定）：
- 终点：船体中心 TARGET_XY = (0.24, 0)，船头朝 +x（TARGET_YAW = 0，背对机械臂、朝缸的另一头）。
  理由：在可抓区中央（离底座 0.24 m，竖直夹持轻松）；船尾离隔板 2.4 cm；船头朝外，松开后 BODY2 可直接游出去。
- 起点：阶段 2 成功夹住时的真实状态（"上一阶段的终点 = 下一阶段的起点"）。由 collect_start_states() 生成并存成 .npz：
  先用脚本策略，阶段 2 的 SAC 训好后换成 SAC 的终点状态。
- 动作 (5,)：同阶段 2：(dx, dy, dz) × 1 cm、d_roll × 0.1 rad、d_grip × 0.1 rad。
  转手腕 = 转船（夹住时 1:1，10-22 实测；夹得不紧时桅杆会在两指间打滑）。
- 观测 (38,)：阶段 2 的 30 维 + 船到终点的位置差 (2) + 朝向误差 sin/cos (2) + 船的高度偏差 (1)
  + 指令 TCP - 实际 TCP (3)：手臂被桅杆顶着时的变形量；松手前把它归零（"卸力"），船就不会被弹开。真机上可测（指令已知，实际由舵机角度正运动学算）。
  高度偏差 = 船体中心 z - 自然漂浮时的 z。10-22 发现：夹着拖时手臂会把船压下或提起约 1.5 cm（竖直受力 ±4.6 N）、
  还带 1.5° 横倾，一松手船弹回原位，带着它以 2.5–7 cm/s 往侧面漂走。所以要"托着船走"，高度偏差进观测和奖励。
- 奖励：-2·位置误差(m) - 0.3·|朝向误差|(rad) - 10·|高度偏差|(m) - 0.01·|动作|² - 1（撞缸壁/隔板/平台/船体）
  + 0.2（没到终点时保持夹住）- 0.5（没到终点就松手）
  + 到终点后：0.5·夹爪张开程度 + 0.5（手指离开桅杆）；成功 +20 并结束。最多 200 步（10 s）。
- 成功：松手并抬起的那一刻，位置误差 < 2 cm、朝向误差 < 10°、夹爪张开 (> 0.25 rad) 且不碰桅杆、TCP 高出桅杆顶 1 cm 以上、
  船速 < 3 cm/s，连续 3 步。10-22：松手时船总会被带出 1.5–2 cm/s（试过放松夹紧力、慢慢张开、后退、同步指令，最好约 2 cm/s），
  这个水模型低速阻力很小，松手后还会漂 3–4 cm —— 作为已知局限单独记录（info 里的 speed），不算进成功。
"""
import os
import numpy as np
from gymnasium import spaces
import mujoco

from envs.grasp_env import GraspEnv
from envs.reset_env import MAST_TOP, ROLL_STEP, WS_LO, WS_HI, GRIP_OPEN

TARGET_XY = np.array([0.24, 0.0])
TARGET_YAW = 0.0
BOAT_Z_EQ = -0.04 - 0.0168 + 0.025       # 自然漂浮时船体中心的高度（水面 - 吃水 + 半高）
POS_TOL, YAW_TOL = 0.02, np.radians(10)
RELEASE_STEPS = 3
from envs.state_io import STATE_KEYS, snapshot, restore     # 环境之间传状态（10-22 移到 state_io.py，阶段 2 也要用）


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def collect_start_states(policy, n, seed=0, out=None):
    """用阶段 2 的策略跑 GraspEnv，收集 n 个"成功夹住"时的状态。policy(obs) -> action。"""
    env, states, ep = GraspEnv(), [], 0
    while len(states) < n:
        obs, _ = env.reset(seed=seed + ep); ep += 1
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                break
        if info["success"]:
            states.append(snapshot(env))
    data = {k: np.stack([s[k] for s in states]) for k in STATE_KEYS}
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, **data)
    print(f"收集到 {n} 个起点，用了 {ep} 个阶段 2 回合（成功率 {n/ep:.0%}）")
    return data


class DragEnv(GraspEnv):
    def __init__(self, start_states="results/stage3_drag/start_states_scripted.npz", max_steps=200, **kw):
        super().__init__(max_steps=max_steps, **kw)
        S = np.load(start_states)
        self.starts = [{k: S[k][i] for k in STATE_KEYS} for i in range(len(S["qpos"]))]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(38,), dtype=np.float32)

    def _errs(self):
        pos = TARGET_XY - self.d.xpos[self.bid][:2]
        yaw = wrap_pi(self._boat_yaw() - TARGET_YAW)
        return pos, float(yaw)

    def _obs(self):
        pos, yaw = self._errs()
        dz = self.d.xpos[self.bid][2] - BOAT_Z_EQ
        return np.concatenate([super()._obs(), pos, [np.sin(yaw), np.cos(yaw), dz],
                               self.tcp_cmd - self.ik.tcp(self.d)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super(GraspEnv, self).reset(seed=seed)                # 只用 ReachEnv.reset 设随机数，后面整体覆盖状态
        s = self.starts[int(self.np_random.integers(len(self.starts)))]
        restore(self, s)
        self.boat_xy0 = self.d.xpos[self.bid][:2].copy()
        self.hold = self.release = 0
        self.steps = 0
        self.prev_action = np.zeros(5)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.roll_cmd = float(np.clip(self.roll_cmd + a[3] * ROLL_STEP, self.roll_lo, self.roll_hi))
        self.grip_cmd = float(np.clip(self.grip_cmd + a[4] * 0.1, self.grip_lo, GRIP_OPEN))
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
        pos, yaw = self._errs()
        pos_err = float(np.linalg.norm(pos))
        f_touch, m_touch = self._finger_contacts()
        touching = f_touch or m_touch
        q_grip = float(self.d.qpos[self.grip_qa])
        grasped = f_touch and m_touch and (q_grip - self.grip_cmd > 0.03)
        at_target = pos_err < POS_TOL and abs(yaw) < YAW_TOL
        mast_top_z = self.d.xpos[self.bid][2] + MAST_TOP
        clear = (not touching) and q_grip > 0.25 and self.ik.tcp(self.d)[2] > mast_top_z + 0.01
        speed = float(np.linalg.norm(self.d.qvel[self.bva:self.bva + 2]))
        released_ok = at_target and clear and speed < 0.03
        self.release = self.release + 1 if released_ok else 0
        success = self.release >= RELEASE_STEPS
        dz = float(self.d.xpos[self.bid][2] - BOAT_Z_EQ)
        reward = -2.0 * pos_err - 0.3 * abs(yaw) - 10.0 * abs(dz) - 0.01 * float(a @ a) - (1.0 if collided else 0.0)
        if at_target:
            reward += 0.5 * np.clip(q_grip / GRIP_OPEN, 0, 1) + (0.5 if not touching else 0.0)
        else:
            reward += 0.2 if grasped else -0.5 * (not touching)
        reward += 20.0 if success else 0.0
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        info = dict(pos_err=pos_err, yaw_err_deg=float(np.degrees(yaw)), grasped=grasped, touching=touching,
                    at_target=at_target, clear=clear, speed=speed, success=bool(success), is_success=bool(success),
                    collided=collided, dist=pos_err, dz=dz)
        return self._obs(), reward, terminated, truncated, info


def scripted_drag(obs):
    """手写策略：夹着拖向终点 + 转腕调朝向；到了就张开夹爪、抬起离开。只看观测。"""
    pos, yaw, dz = obs[30:32], np.arctan2(obs[32], obs[33]), obs[34]
    a = np.zeros(5)
    settled = abs(dz) < 0.002 and np.linalg.norm(obs[20:22]) < 0.003    # 船回到自然高度、几乎不动
    at_target = np.linalg.norm(pos) < 0.012 and abs(yaw) < np.radians(6) and settled
    opening = obs[5] > 0.0                                   # 夹爪实际角度 > 0：已经开始松开了，就继续松
    if not at_target and not opening:
        a[0:2] = np.clip(pos / 0.03, -0.5, 0.5)              # TCP 带着桅杆往终点走。增益只用 1/3（误差 / 3 cm）：
                                                              # 用 1（误差 / 1 cm）时，舵机滞后让它在终点附近来回过冲（10-22）
        a[3] = np.clip(-yaw / (3 * ROLL_STEP), -0.5, 0.5)     # 转腕 = 转船（同方向）
        a[2] = np.clip(-dz / 0.03, -0.5, 0.5)                  # 托着船：把船带回自然漂浮高度（船低了就往上提）
        a[4] = -1.0                                           # 保持夹紧
    elif not opening and np.linalg.norm(obs[35:38]) > 0.0005:
        a[0:3] = np.clip(-obs[35:38] / 0.01, -1, 1)            # 先卸力：把指令 TCP 拉回实际 TCP，手臂不再顶着桅杆
        a[4] = -1.0
    else:
        a[4] = 1.0                                            # 张开
        if obs[5] > 0.25:
            a[2] = 1.0                                        # 张开后往上抬，离开桅杆
    return a
