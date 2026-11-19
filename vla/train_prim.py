"""方案 F 的训练：用 DQN 学“什么时候插入哪一种脱身动作”。SmolVLA 冻结。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/train_prim.py MODEL_DIR ENGINE OUT_DIR [决策步数=20000] [并行环境数=6]
这里的“一步”是一次决策（0.5 秒，10 个控制周期）。训练只用快船（10–20 cm/s），因为失败和卡住主要发生在那里。
"""
import os, sys, time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

M, ENG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TOTAL = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
NENV = int(sys.argv[5]) if len(sys.argv) > 5 else 6


def make(i):
    def _f():
        from prim_env import PrimEnv
        e = PrimEnv(M, ENG, fast_bias=1.0); e.e.rng = np.random.default_rng(200000 + i)
        return e
    return _f


class Log(BaseCallback):
    def __init__(self):
        super().__init__(); self.ep, self.acts, self.t0, self.last = [], [], time.time(), 0
    def _on_step(self):
        self.acts.extend(int(a) for a in self.locals["actions"])
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.ep.append((info["success"], info.get("early", False), info.get("n_prim", 0)))
        if self.num_timesteps - self.last >= 1000:
            self.last = self.num_timesteps
            E = self.ep[-300:]; long = [e for e in E if not e[1]]
            a = np.bincount(self.acts[-3000:], minlength=5) / max(len(self.acts[-3000:]), 1)
            print(f"决策 {self.num_timesteps}：最近 {len(E)} 回合成功率 {np.mean([e[0] for e in E]) * 100:.1f}%（其中超过 2 秒的 {len(long)} 回合 "
                  f"{np.mean([e[0] for e in long]) * 100 if long else float('nan'):.1f}%）| 各选项的比例 继续 {a[0]:.2f} 上抬 {a[1]:.2f} 后退 {a[2]:.2f} 左 {a[3]:.2f} 右 {a[4]:.2f} | "
                  f"探索率 {self.model.exploration_rate:.2f} | 已用 {(time.time() - self.t0) / 60:.0f} 分钟", flush=True)
        return True


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = VecMonitor(SubprocVecEnv([make(i) for i in range(NENV)], start_method="spawn"))
    model = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=60000, batch_size=128, learning_starts=1500, gamma=0.97,
                train_freq=1, gradient_steps=NENV, target_update_interval=600, exploration_fraction=0.4,
                exploration_initial_eps=0.6, exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]), device="cpu", verbose=0)
    print(f"{NENV} 个并行环境，共 {TOTAL} 次决策", flush=True)
    model.learn(TOTAL, callback=[Log(), CheckpointCallback(save_freq=max(5000 // NENV, 1), save_path=OUT, name_prefix="prim")])
    model.save(f"{OUT}/prim_final")
    print("PRIM TRAIN DONE", flush=True)
