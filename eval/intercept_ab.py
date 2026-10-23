"""A（KF）/ B（原始测量历史）对比：每个运行用 best_model，6 个速度档 × 30 回合；再测 2 种干扰（全速度 0–20 cm/s，60 回合）。
    .venv/bin/python eval/intercept_ab.py RUN_DIR [RUN_DIR ...] [--kf-tuned]     # RUN_DIR 写 scripted = 只评估脚本拦截器
每个运行打印一行 JSON，并写进 results/stage5_intercept/eval/<方案>_<种子>.json（每个运行一个文件，多个进程并行也不会互相覆盖）。
- 目录名带 residual：动作 = 脚本拦截器（看后台 KF 观测）+ 0.3 × SAC，和训练时一样（envs/residual.py）。
- --kf-tuned：KF 按当前条件的噪声配置（测量噪声 = σ，突变门限 = 6 σ）。不加时 KF 固定按 1 mm 配置（v2 的评估方式）。
"""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv, scripted_intercept
from envs.kalman import BoatKF
from envs.residual import ResidualWrapper

BINS = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]
COND = {"标准（1 mm、延迟 1 帧）": dict(sigma=0.001, lat=1), "噪声 ×3（3 mm、1.5°）": dict(sigma=0.003, lat=1),
        "延迟 3 帧（150 ms）": dict(sigma=0.001, lat=3)}
TUNED = "--kf-tuned" in sys.argv


def set_cond(e, sigma, lat):
    e.sigma_pos, e.sigma_yaw, e.latency = sigma, np.radians(0.5) * sigma / 0.001, lat
    e.kf = BoatKF(sigma_pos=sigma, sigma_yaw=e.sigma_yaw, gate=6 * sigma) if TUNED else BoatKF()
    e.kf.reset(np.zeros(2), 0.0)                              # 先初始化一次（父类 reset 里会先读一次观测）


def run(model, env, n, seed0=0):
    S, N, H, J = [], [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            obs, r, t, tr, info = env.step(model.predict(obs, deterministic=True)[0])
            J.append(info["da2"])
            if t or tr:
                break
        S.append(info["success"]); N.append(env.steps); H.append(info["wall_hits"])
    S = np.array(S)
    return dict(succ=int(S.sum()), n=n, steps=float(np.median(np.array(N)[S])) if S.any() else None, walls=float(np.mean(H)),
                jitter=float(np.mean(J)))                     # 平滑度：每步 |Δa|² 的平均（越小越平滑）


class Zero:
    """"残差恒为 0" = 只用脚本拦截器（RUN_DIR 写 scripted 时用）。"""
    def predict(self, obs, deterministic=True):
        return np.zeros(5), None


for rd in [a for a in sys.argv[1:] if not a.startswith("--")]:
    rd = rd.rstrip("/")
    pred = "history" if "history" in rd else "kf"
    model = Zero() if rd == "scripted" else SAC.load(os.path.join(rd, "best_model.zip"))
    inner = InterceptEnv(predictor=pred)
    residual = "residual" in rd or rd == "scripted"
    env = ResidualWrapper(inner, lambda obs: scripted_intercept(inner.kf_obs())) if residual else inner
    res = {"kf_tuned": TUNED, "bins": {}, "cond": {}}
    set_cond(inner, 0.001, 1)
    for vr in BINS:
        inner.v_range = vr
        res["bins"][f"{100*vr[0]:.0f}–{100*vr[1]:.0f}"] = run(model, env, 30)
    inner.v_range = (0.0, 0.20)
    for name, c in COND.items():
        set_cond(inner, c["sigma"], c["lat"])
        res["cond"][name] = run(model, env, 60, seed0=1000)
    print(rd, json.dumps(res, ensure_ascii=False), flush=True)
    os.makedirs("results/stage5_intercept/eval", exist_ok=True)
    tag = rd.split("stage5_intercept/")[-1].replace("/", "_") + ("_kftuned" if TUNED else "")
    json.dump(res, open(f"results/stage5_intercept/eval/{tag}.json", "w"), ensure_ascii=False, indent=1)
