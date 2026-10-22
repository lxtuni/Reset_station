"""收集"上一阶段策略成功时的状态"，作为下一阶段训练的起点（技能接力，方案 A）。

    python eval/collect_states.py reach 500      # 阶段 1 的 SAC 成功时的状态 → results/stage2_grasp/start_states_reach_sac.npz
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from envs.reset_env import ReachEnv
from envs.state_io import snapshot, save_states

stage = sys.argv[1] if len(sys.argv) > 1 else "reach"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
assert stage == "reach", "目前只做了阶段 1 → 2 的接力（阶段 2 → 3 已在 envs/drag_env.py 的 collect_start_states 里）"
model = SAC.load("results/stage1_reach/sac_seed0/best_model.zip")
env, states, ep = ReachEnv(), [], 0
while len(states) < n:
    obs, _ = env.reset(seed=200_000 + ep); ep += 1             # 和评估、训练用的种子错开
    while True:
        obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
        if term or trunc:
            break
    if info["success"]:
        states.append(snapshot(env))
out = "results/stage2_grasp/start_states_reach_sac.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
save_states(states, out)
print(f"收集到 {n} 个阶段 1 终点状态（{ep} 回合，成功率 {n/ep:.0%}）→ {out}")
