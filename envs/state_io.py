"""在环境之间传递完整状态：上一阶段的终点 = 下一阶段的起点（技能串联时的"接力"）。

snapshot(env) 把环境此刻的仿真状态和指令变量存成 dict；restore(env, s) 装进另一个环境（同一个场景 XML，
所以 qpos 等数组的排列完全一样）。ReachEnv 没有夹爪指令变量 grip_cmd，存的时候按"张开"补上。
"""
import numpy as np
import mujoco

STATE_KEYS = ["qpos", "qvel", "ctrl", "q_cmd", "tcp_cmd", "roll_cmd", "grip_cmd"]


def snapshot(env, grip_open=0.35):
    return dict(qpos=env.d.qpos.copy(), qvel=env.d.qvel.copy(), ctrl=env.d.ctrl.copy(), q_cmd=env.q_cmd.copy(),
                tcp_cmd=env.tcp_cmd.copy(), roll_cmd=np.array(env.roll_cmd),
                grip_cmd=np.array(getattr(env, "grip_cmd", grip_open)))


def restore(env, s):
    mujoco.mj_resetData(env.m, env.d)
    env.d.qpos[:], env.d.qvel[:], env.d.ctrl[:] = s["qpos"], s["qvel"], s["ctrl"]
    mujoco.mj_forward(env.m, env.d)
    env.q_cmd, env.tcp_cmd = s["q_cmd"].copy(), s["tcp_cmd"].copy()
    env.roll_cmd, env.grip_cmd = float(s["roll_cmd"]), float(s["grip_cmd"])


def save_states(states, path):
    np.savez(path, **{k: np.stack([s[k] for s in states]) for k in STATE_KEYS})


def load_states(path):
    S = np.load(path)
    return [{k: S[k][i] for k in STATE_KEYS} for i in range(len(S["qpos"]))]
