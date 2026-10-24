"""拦截 → 拖动转向的接力：用拦截策略跑回合，把"夹住、船停稳"（拦截成功）那一刻的完整状态存下来，当拖动转向的起点。

    .venv/bin/python eval/collect_intercept_states.py MODEL_DIR N OUT.npz [seed0]
例：.venv/bin/python eval/collect_intercept_states.py results/stage5_intercept/kf/sac_seed0_demo30000_v4 300 \\
        results/stage6_intercept_drag/start_states_intercept.npz
船速在 0–20 cm/s 之间随机。打印这些起点离拖动终点有多远、船头朝向差多少、转腕够不够把船转到目标朝向。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv
from envs.drag_env import TARGET_XY, TARGET_YAW, wrap_pi
from envs.state_io import snapshot, save_states

rd, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 50_000           # 和评估用的种子（0–1059）错开
model = SAC.load(os.path.join(rd, "best_model.zip"))
env = InterceptEnv(predictor="history" if "history" in rd else "kf", v_range=(0.0, 0.20))
env.block_outward = "stage8" in rd or "stage9" in rd     # 只有 stage8 之后训练的模型用"边界处不往外推"
states, ep = [], 0
while len(states) < n:
    obs, _ = env.reset(seed=seed0 + ep); ep += 1
    while True:
        obs, r, t, tr, info = env.step(model.predict(obs, deterministic=True)[0])
        if t or tr:
            break
    if info["success"]:
        states.append(snapshot(env))
os.makedirs(os.path.dirname(out), exist_ok=True)
save_states(states, out)

# 起点的统计：位置误差、朝向误差、转腕需要转到哪里（夹住时转腕和船 1:1 同向转）
dist, yaw_err, roll_ok = [], [], []
for s in states:
    env.d.qpos[:] = s["qpos"]
    import mujoco; mujoco.mj_forward(env.m, env.d)
    dist.append(np.linalg.norm(TARGET_XY - env.d.xpos[env.bid][:2]))
    e = wrap_pi(env._boat_yaw() - TARGET_YAW); yaw_err.append(e)
    need = [float(s["roll_cmd"]) - e + k * 2 * np.pi for k in (-1, 0, 1)]   # 船转 -e，手腕也转 -e（或多转一圈）
    roll_ok.append(any(env.roll_lo <= r <= env.roll_hi for r in need))
dist, yaw_err = np.array(dist), np.degrees(np.abs(yaw_err))
print(f"{len(states)} 个起点，用了 {ep} 个拦截回合（成功率 {len(states)/ep:.0%}）→ {out}")
print(f"离终点距离：中位 {100*np.median(dist):.1f} cm，最大 {100*dist.max():.1f} cm")
print(f"船头朝向误差：中位 {np.median(yaw_err):.0f}°，> 90° 的 {np.mean(yaw_err > 90):.0%}，> 150° 的 {np.mean(yaw_err > 150):.0%}")
print(f"转腕在限位内能转到目标朝向的：{np.mean(roll_ok):.0%}")
