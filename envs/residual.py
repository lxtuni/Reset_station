"""残差强化学习 (residual RL) 的包装器：最终动作 = 脚本策略的动作 + SCALE × RL 策略的动作。

为什么（10-22）：阶段 2 纯 SAC 训练 9.5 万步成功率 0 %；加 3 万步示范后 15 万步到 38 %（最高 55 %），远不如脚本策略的 100 %。
残差 RL 让 RL 从"脚本策略的水平"出发，只学一个小修正（每维最多 ±SCALE），用来减少推船、加快动作、应付脚本没想到的情况。
RL 输出 0 时就等于脚本策略，所以训练一开始就有很高的成功率，探索也安全。
观测、奖励都和原环境完全一样，只改动作。
"""
import numpy as np
import gymnasium as gym

SCALE = 0.3


class ResidualWrapper(gym.Wrapper):
    def __init__(self, env, base_policy, scale=SCALE):
        super().__init__(env)
        self.base, self.scale = base_policy, scale
        self._obs = None

    def reset(self, **kw):
        self._obs, info = self.env.reset(**kw)
        return self._obs, info

    def step(self, action):
        a = np.clip(self.base(self._obs) + self.scale * np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._obs, r, term, trunc, info = self.env.step(a)
        return self._obs, r, term, trunc, info

    # 训练脚本统计用：内层环境的步数
    @property
    def steps(self):
        return self.env.steps
