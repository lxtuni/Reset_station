"""评测“SmolVLA + 残差修正”。协议和 eval_vla.py 相同：6 个速度档 × 30 回合，每步都重新推理。

用法（仓库根目录，PYTHONPATH=act:vla）：
    python vla/eval_resid.py MODEL_DIR ENGINE none            # 修正量恒为 0：应该复现 SmolVLA 自己的成绩
    python vla/eval_resid.py MODEL_DIR ENGINE outputs/resid/resid_final.zip [--seed0 1000] [--alpha 0.3]
"""
import argparse, json, time
import numpy as np
from common import BINS
from resid_env import ResidEnv

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("engine"); ap.add_argument("sac")
ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--bins", type=int, default=6)
ap.add_argument("--json", default=None)
args = ap.parse_args()

env = ResidEnv(args.model, args.engine, alpha=args.alpha, v_range=(0.0, 0.0))
sac = None
if args.sac != "none":
    from stable_baselines3 import SAC
    sac = SAC.load(args.sac, device="cpu")
res, tot, D, t0, nstep = {}, 0, [], time.time(), 0
for vr in BINS[:args.bins]:
    env.fixed_v = vr; S, N, H = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed0 + ep)
        while True:
            d = sac.predict(obs, deterministic=True)[0] if sac is not None else np.zeros(5, np.float32)
            D.append(float(np.abs(d).mean()))
            obs, r, term, trunc, info = env.step(d); nstep += 1
            if term or trunc:
                break
        S.append(info["success"]); N.append(info["steps"]); H.append(info["wall_hits"])
    S = np.array(S); name = f"{100 * vr[0]:.0f}-{100 * vr[1]:.0f} cm/s"
    res[name] = dict(succ=int(S.sum()), n=len(S), walls=float(np.mean(H)))
    tot += int(S.sum())
    print(f"{name}: 成功 {int(S.sum())}/{len(S)}，用时 {np.median(np.array(N)[S]) * 0.05 if S.any() else float('nan'):.1f} s，撞墙 {np.mean(H):.2f} 次/回合", flush=True)
n = args.episodes * args.bins
print(f"合计成功率 {tot}/{n} = {tot / n * 100:.1f}%  残差={args.sac} alpha={args.alpha} 种子起点={args.seed0} "
      f"修正量平均幅度={np.mean(D):.3f}  每步 {1000 * (time.time() - t0) / nstep:.1f} ms")
if args.json:
    json.dump(dict(args=vars(args), bins=res), open(args.json, "w"), ensure_ascii=False, indent=1)
