"""A（KF）/ B（原始测量历史）对比：每个运行用 best_model，6 个速度档 × 30 回合；再测 2 种干扰（全速度 0–20 cm/s，60 回合）。
    .venv/bin/python eval/intercept_ab.py RUN_DIR [RUN_DIR ...]（并行跑多个进程时会抢着写 json，结果以打印的行为准）   → 每个运行一行 JSON 结果，写进 results/stage5_intercept/eval_ab.json
"""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv

BINS = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]
COND = {"标准（1 mm、延迟 1 帧）": dict(sigma=0.001, lat=1), "噪声 ×3（3 mm、1.5°）": dict(sigma=0.003, lat=1),
        "延迟 3 帧（150 ms）": dict(sigma=0.001, lat=3)}


def run(model, env, n, seed0=0):
    S, N, H = [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            obs, r, t, tr, info = env.step(model.predict(obs, deterministic=True)[0])
            if t or tr:
                break
        S.append(info["success"]); N.append(env.steps); H.append(info["wall_hits"])
    S = np.array(S)
    return dict(succ=int(S.sum()), n=n, steps=float(np.median(np.array(N)[S])) if S.any() else None, walls=float(np.mean(H)))


out = {}
for rd in sys.argv[1:]:
    pred = "history" if "/history/" in rd else "kf"
    model = SAC.load(os.path.join(rd, "best_model.zip"))
    env = InterceptEnv(predictor=pred)
    res = {"bins": {}, "cond": {}}
    for vr in BINS:
        env.v_range = vr
        res["bins"][f"{100*vr[0]:.0f}–{100*vr[1]:.0f}"] = run(model, env, 30)
    env.v_range = (0.0, 0.20)
    for name, c in COND.items():
        env.sigma_pos, env.sigma_yaw, env.latency = c["sigma"], np.radians(0.5) * c["sigma"] / 0.001, c["lat"]
        res["cond"][name] = run(model, env, 60, seed0=1000)
    env.sigma_pos, env.sigma_yaw, env.latency = 0.001, np.radians(0.5), 1
    out[rd] = res
    print(rd, json.dumps(res, ensure_ascii=False), flush=True)
p = "results/stage5_intercept/eval_ab.json"
old = json.load(open(p)) if os.path.exists(p) else {}
old.update(out)
json.dump(old, open(p, "w"), ensure_ascii=False, indent=1)
