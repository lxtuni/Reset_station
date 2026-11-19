"""评测方案 F：SmolVLA + 学到的“脱身动作选择器”。6 个速度档 × 30 回合，和其他评测同一协议。
用法（仓库根目录，PYTHONPATH=act:vla）：
    python vla/eval_prim.py MODEL_DIR ENGINE none             # 永远选“继续”：就是 SmolVLA 自己
    python vla/eval_prim.py MODEL_DIR ENGINE outputs/prim/prim_final.zip [--seed0 1000]
"""
import argparse, json
import numpy as np
from common import BINS
from prim_env import PrimEnv, NAMES

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("engine"); ap.add_argument("dqn")
ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--json", default=None)
args = ap.parse_args()
env = PrimEnv(args.model, args.engine, v_range=(0.0, 0.0))
dqn = None
if args.dqn != "none":
    from stable_baselines3 import DQN
    dqn = DQN.load(args.dqn, device="cpu")
res, tot, used, cnt = {}, 0, 0, np.zeros(5, int)
for vr in BINS:
    env.e.fixed_v = vr; S, H, N = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed0 + ep)
        while True:
            a = int(dqn.predict(obs, deterministic=True)[0]) if dqn is not None else 0
            if env.early is None:
                cnt[a] += 1
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        S.append(info["success"]); H.append(info["wall_hits"]); N.append(info["n_prim"])
    name = f"{100 * vr[0]:.0f}-{100 * vr[1]:.0f} cm/s"; S = np.array(S)
    res[name] = dict(succ=int(S.sum()), n=len(S), walls=float(np.mean(H)), prims=float(np.mean(N)))
    tot += int(S.sum()); used += int(np.sum(np.array(N) > 0))
    print(f"{name}: 成功 {int(S.sum())}/{len(S)}，撞墙 {np.mean(H):.2f} 次/回合，用过脱身动作的回合 {int(np.sum(np.array(N) > 0))} 个", flush=True)
n = args.episodes * len(BINS)
print(f"合计成功率 {tot}/{n} = {tot / n * 100:.1f}%  选择器={args.dqn} 种子起点={args.seed0} | 用过脱身动作的回合 {used} 个 | "
      f"各选项被选的次数 " + "，".join(f"{NAMES[i]} {cnt[i]}" for i in range(5)))
if args.json:
    json.dump(dict(args=vars(args), bins=res), open(args.json, "w"), ensure_ascii=False, indent=1)
