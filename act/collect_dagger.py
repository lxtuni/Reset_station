"""DAgger 数据采集：让 ACT（学生）自己开，走到它自己会走到的状态；每一步都问老师“这里你会怎么做”，存老师的答案。

纯模仿学习只见过老师走过的状态。学生一旦偏离，就进入没见过的画面，越偏越远。DAgger 把这些偏离后的画面
配上老师的纠正动作加进训练集。
为了让回合不至于全是失败轨迹，每 5 步随机决定这一段由学生还是老师控制（学生占 beta）。
标签永远是老师的动作；图像和关节角度是实际走到的状态。

用法（仓库根目录）：python act/collect_dagger.py MODEL_DIR --episodes 50 --seed0 40000 --out data/raw_dagger_0
"""
import argparse, os, time
import numpy as np, torch
from stable_baselines3 import SAC
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints, TEACHER

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--v_min", type=float, default=0.0)
ap.add_argument("--v_max", type=float, default=0.20)
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed0", type=int, default=40000)        # 和示教（10000–50000 内的分片）、评测（0–29）错开
ap.add_argument("--beta", type=float, default=0.7)         # 学生控制的比例
ap.add_argument("--out", default="data/raw_dagger")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

policy = ACTPolicy.from_pretrained(args.model).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
env = make_env(v_range=(args.v_min, args.v_max)); cams = Cams(env); teacher = SAC.load(TEACHER)
rng = np.random.default_rng(args.seed0)
succ, frames, t0 = 0, 0, time.time()
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed0 + ep)
    tops, wrists, states, actions, queue = [], [], [], [], []
    while True:
        top, wrist = cams(); q = joints(env)
        a_t = np.asarray(teacher.predict(obs, deterministic=True)[0], np.float32)
        if not queue:                                      # 每 5 步决定下一段谁来开
            if rng.random() < args.beta:
                b = pre({"observation.images.top": f(top), "observation.images.wrist": f(wrist),
                         "observation.state": torch.from_numpy(q).unsqueeze(0), "task": [""]})
                with torch.inference_mode():
                    queue = list(post(policy.predict_action_chunk(b))[0, :5].cpu().numpy())
            else:
                queue = [None] * 5                         # None = 这一步执行老师的动作
        a_s = queue.pop(0)
        tops.append(top); wrists.append(wrist); states.append(q); actions.append(a_t.copy())
        obs, r, term, trunc, info = env.step(np.array(a_t if a_s is None else a_s, dtype=np.float32))
        if term or trunc:
            break
    succ += int(info["success"]); frames += len(actions)
    np.savez_compressed(f"{args.out}/ep_{ep:04d}.npz", top=np.array(tops), wrist=np.array(wrists),
                        state=np.array(states), action=np.array(actions), seed=args.seed0 + ep, v0=info["v0"])
    if (ep + 1) % 10 == 0:
        print(f"{ep + 1}/{args.episodes} 回合，混合控制成功率 {succ / (ep + 1) * 100:.0f}%，{frames} 帧，"
              f"已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{args.episodes} 回合，{frames} 帧，存在 {args.out}")
