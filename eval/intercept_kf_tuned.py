"""方案 A 在噪声 ×3 下：KF 按真实噪声配置（测量噪声 3 mm / 1.5°，突变门限 18 mm）再测一次，同一批 60 个种子。
    .venv/bin/python eval/intercept_kf_tuned.py results/stage5_intercept/kf/sac_seed0_demo30000
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv
from envs.kalman import BoatKF

rd = sys.argv[1]
model = SAC.load(rd + "/best_model.zip")
env = InterceptEnv(predictor="kf", v_range=(0.0, 0.20))
env.sigma_pos, env.sigma_yaw = 0.003, np.radians(1.5)
env.kf = BoatKF(sigma_pos=0.003, sigma_yaw=np.radians(1.5), gate=0.018)
env.kf.reset(np.zeros(2), 0.0)                       # 先初始化一次（父类 reset 里会先读一次观测）
S, N, H = [], [], []
for ep in range(60):
    obs, _ = env.reset(seed=1000 + ep)
    while True:
        obs, r, t, tr, info = env.step(model.predict(obs, deterministic=True)[0])
        if t or tr:
            break
    S.append(info["success"]); N.append(env.steps); H.append(info["wall_hits"])
S = np.array(S)
print(rd, f"噪声×3 + KF 按 3 mm 配置：成功 {S.sum()}/60，步数中位 {np.median(np.array(N)[S]) if S.any() else None}，平均撞墙 {np.mean(H):.2f}")
