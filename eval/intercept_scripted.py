"""脚本拦截器（envs/intercept_env.py 的 scripted_intercept）按船速分档测试，每档 30 回合（种子 0–29）。
    .venv/bin/python eval/intercept_scripted.py [从第几档开始]
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from collections import Counter
from envs.intercept_env import InterceptEnv, scripted_intercept
bins = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]
if len(sys.argv) > 1:
    bins = bins[int(sys.argv[1]):]
env = InterceptEnv(predictor="kf")
for vr in bins:
    env.v_range = vr; S, N, H, KV, RA = [], [], [], [], []; why = Counter()
    for ep in range(30):
        obs, _ = env.reset(seed=ep); grasped = False; kv = []
        while True:
            obs, r, t, tr, info = env.step(scripted_intercept(obs)); grasped |= info["grasped"]; kv.append(info["kf_vel_err"])
            if t or tr:
                break
        S.append(info["success"]); N.append(env.steps); H.append(info["wall_hits"]); RA.append(info["reaims"]); KV.append(np.mean(kv))
        if not info["success"]:
            why["grasped-but-not-stopped" if grasped else "never-grasped"] += 1
    S = np.array(S)
    print(f"v0 {vr}: success {S.sum()}/30, steps median {np.median(np.array(N)[S]) if S.any() else None}, "
          f"wall hits mean {np.mean(H):.2f} (episodes with hits {sum(h > 0 for h in H)}), "
          f"re-aims mean {np.mean(RA):.2f}, KF vel err {100*np.mean(KV):.2f} cm/s, failures {dict(why)}")
