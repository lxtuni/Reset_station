"""方案 F：SmolVLA 冻结。强化学习只学一件事——每隔 0.5 秒，是让 SmolVLA 继续，还是插入一个“脱身动作”。

失败分析的结论是：SmolVLA 第一次没夹住之后不会重来，而教师在那种状态下会上抬、绕开再重新对准。
让强化学习从零摸索出这一整套动作太难（前三次尝试都没成），所以把问题缩小：
  脱身动作由人写好几种（每种持续 10 步 = 0.5 秒，夹爪张开）：
    0 继续：这 10 步完全交给 SmolVLA
    1 上抬
    2 后退并上抬
    3 向左并上抬
    4 向右并上抬
  强化学习（DQN）学的是：看当前的状态，这 0.5 秒该选哪一个。
前 40 步（2 秒）不做选择，始终是 SmolVLA；大多数顺利的回合在这之后很快就结束了。
小网络看到的 29 个数全部来自关节角度和 SmolVLA 自己的输出，不用船的真实位置，也不知道“有没有夹住”。
奖励就是环境原来的奖励（成功 +20，撞墙和推船扣分，离桅杆越远扣得越多），按 10 步累加。
"""
import numpy as np, gymnasium as gym
from resid_env import ResidEnv

PRIMS = [None, (0.0, 0.0, 1.0), (-1.0, 0.0, 0.6), (0.0, 1.0, 0.4), (0.0, -1.0, 0.4)]
NAMES = ["继续", "上抬", "后退并上抬", "向左并上抬", "向右并上抬"]
START, DUR = 40, 10


class PrimEnv(gym.Env):
    def __init__(self, model_dir, engine, v_range=None, fast_bias=1.0):
        self.e = ResidEnv(model_dir, engine, alpha=1.0, fast_bias=fast_bias, v_range=v_range)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), np.float32)
        self.action_space = gym.spaces.Discrete(len(PRIMS))

    def _obs(self):
        e = self.e; h = np.array(self.hist[-10:]) if self.hist else np.zeros((1, 5), np.float32)
        last = np.zeros(5, np.float32); last[self.last_prim] = 1.0
        return np.concatenate([e.last_q, e.a_vla, h.mean(0), [self.t / 200.0, self.since / 200.0, self.n_prim / 5.0],
                               last, e.chunk10]).astype(np.float32)

    def _one(self, prim):
        e = self.e
        self.hist.append(e.a_vla.copy())
        delta = np.zeros(5, np.float32)
        if prim is not None:
            target = np.array([*prim, 0.0, 1.0], np.float32)             # 位移按脱身动作走，腕部不转，夹爪张开
            delta = target - e.a_vla
        _, r, term, trunc, info = e.step(delta)
        self.t += 1
        return r, term, trunc, info

    def reset(self, *, seed=None, options=None):
        self.e.reset(seed=seed)
        self.t, self.since, self.n_prim, self.last_prim, self.hist, self.early = 0, 0, 0, 0, [], None
        for _ in range(START):                                           # 前 2 秒不做选择
            r, term, trunc, info = self._one(None)
            if term or trunc:
                self.early = info; return np.zeros(29, np.float32), {}
        return self._obs(), {}

    def step(self, action):
        if self.early is not None:                                       # 回合在前 2 秒内就结束了：没有可做的选择
            info = dict(self.early); info["early"] = True; info["n_prim"] = 0
            return np.zeros(29, np.float32), 0.0, True, False, info
        action = int(action); R = 0.0
        if action != 0:
            self.n_prim += 1; self.since = 0
        self.last_prim = action
        for _ in range(DUR):
            r, term, trunc, info = self._one(PRIMS[action]); R += r; self.since += 1
            if term or trunc:
                info = dict(info); info["early"] = False; info["n_prim"] = self.n_prim
                return np.zeros(29, np.float32), R, bool(term), bool(trunc), info
        return self._obs(), R, False, False, {"n_prim": self.n_prim}
