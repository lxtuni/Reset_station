"""训练残差策略：SmolVLA 冻结，只训练一个小的 SAC 网络输出修正量。

用法（仓库根目录，PYTHONPATH=act:vla）：python vla/train_resid.py MODEL_DIR ENGINE OUT_DIR [总步数=200000] [并行环境数=4] [alpha=0.3]
几个专门为“残差”做的设置：
  - 小网络最后一层初始化为 0、初始噪声很小：训练一开始修正量约等于 0，行为就是原来的 SmolVLA，不会先把它搞坏
  - 熵系数固定为一个小值：SAC 默认会主动加大随机性去探索，这里不需要大幅探索，只要在原动作附近小范围试
"""
import os, sys, time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

M, ENG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TOTAL = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
NENV = int(sys.argv[5]) if len(sys.argv) > 5 else 4
ALPHA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
PEN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.02      # 修正量的代价系数
ENT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.003     # 熵系数（越小越不鼓励随机探索）


def make(i):
    def _f():
        from resid_env import ResidEnv
        e = ResidEnv(M, ENG, alpha=ALPHA, pen=PEN); e.reset(seed=100000 + i)
        return e
    return _f


class Log(BaseCallback):
    def __init__(self):
        super().__init__(); self.s, self.d, self.t0, self.last = [], [], time.time(), 0
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.s.append(info["success"])
        self.d.append(float(np.abs(self.locals["actions"]).mean()))
        if self.num_timesteps - self.last >= 5000:
            self.last = self.num_timesteps
            print(f"步数 {self.num_timesteps}：最近 {len(self.s[-200:])} 回合成功率 {np.mean(self.s[-200:]) * 100:.1f}%，"
                  f"修正量平均幅度 {np.mean(self.d[-2000:]):.3f}，{self.num_timesteps / (time.time() - self.t0):.0f} 步/秒，"
                  f"已用 {(time.time() - self.t0) / 60:.0f} 分钟", flush=True)
        return True


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make(i) for i in range(NENV)], start_method="spawn"))
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300000, batch_size=256, learning_starts=1000,
                gamma=0.98, tau=0.01, train_freq=1, gradient_steps=NENV, ent_coef=ENT,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3.0), device="cuda", verbose=0)
    model.actor.mu.weight.data.zero_(); model.actor.mu.bias.data.zero_()      # 修正量从 0 开始
    # 第一次训练漏了这一步：SAC 的噪声大小由另一个输出层决定（上面的 log_std_init 参数对普通 SAC 不起作用），
    # 默认初始噪声的标准差约为 1，于是一开始修正量就在 ±0.5 左右乱跳。这里把噪声层也初始化成很小（标准差约 0.05）。
    model.actor.log_std.weight.data.zero_(); model.actor.log_std.bias.data.fill_(-3.0)
    print(f"alpha={ALPHA}，修正量代价 {PEN}，熵系数 {ENT}，{NENV} 个并行环境，共 {TOTAL} 步", flush=True)
    model.learn(TOTAL, callback=[Log(), CheckpointCallback(save_freq=max(50000 // NENV, 1), save_path=OUT, name_prefix="resid")])
    model.save(f"{OUT}/resid_final")
    print("RESID TRAIN DONE", flush=True)
