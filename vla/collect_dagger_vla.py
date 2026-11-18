"""SmolVLA 的 DAgger 数据采集。思路和 act/collect_dagger.py 相同：
学生（微调后的 SmolVLA）和教师（SAC）轮流控制，每一步都记录教师的动作作为标签，图像和关节角度是实际走到的状态。

每 5 步随机决定这一段由谁控制（学生占 beta）。学生控制时每一步都重新推理、只执行第一步动作，
去噪 3 次——这是前面实验里成功率最高、也真正能在 50 ms 周期内跑的设置。
用法（仓库根目录，PYTHONPATH=act）：python vla/collect_dagger_vla.py MODEL_DIR --episodes 40 --seed0 40000 --out data/raw_vd1_0
"""
import argparse, os, time
import numpy as np, torch
from stable_baselines3 import SAC
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints, TEACHER

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--v_min", type=float, default=0.0)
ap.add_argument("--v_max", type=float, default=0.20)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--seed0", type=int, default=40000)        # 和示教（10000–50000 的分片）、评测（0–29）错开
ap.add_argument("--beta", type=float, default=0.7)
ap.add_argument("--num_steps", type=int, default=3)
ap.add_argument("--out", default="data/raw_vd1")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

policy = SmolVLAPolicy.from_pretrained(args.model).to("cuda").eval()
policy.config.num_steps = args.num_steps
if hasattr(policy.model, "config"):
    policy.model.config.num_steps = args.num_steps
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
env = make_env(v_range=(args.v_min, args.v_max)); cams = Cams(env); teacher = SAC.load(TEACHER, device="cpu")
rng = np.random.default_rng(args.seed0)
succ, frames, t0 = 0, 0, time.time()
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed0 + ep); policy.reset()
    tops, wrists, states, actions, left, student = [], [], [], [], 0, False
    while True:
        top, wrist = cams(); q = joints(env)
        a_t = np.asarray(teacher.predict(obs, deterministic=True)[0], np.float32)
        if left == 0:                                      # 每 5 步决定下一段谁来开
            student = rng.random() < args.beta; left = 5
        left -= 1
        a = a_t
        if student:
            b = pre({"observation.images.camera1": f(top), "observation.images.camera2": f(wrist),
                     "observation.state": torch.from_numpy(q).unsqueeze(0),
                     "task": ["intercept the drifting boat and grasp its mast"]})
            with torch.inference_mode():
                a = post(policy.predict_action_chunk(b))[0, 0].cpu().numpy()
        tops.append(top); wrists.append(wrist); states.append(q); actions.append(a_t.copy())
        obs, r, term, trunc, info = env.step(np.array(a, dtype=np.float32))
        if term or trunc:
            break
    succ += int(info["success"]); frames += len(actions)
    np.savez_compressed(f"{args.out}/ep_{ep:04d}.npz", top=np.array(tops), wrist=np.array(wrists),
                        state=np.array(states), action=np.array(actions), seed=args.seed0 + ep, v0=info["v0"])
    if (ep + 1) % 10 == 0:
        print(f"{ep + 1}/{args.episodes} 回合，混合控制成功率 {succ / (ep + 1) * 100:.0f}%，{frames} 帧，"
              f"已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
print(f"完成：{args.episodes} 回合，{frames} 帧，存在 {args.out}")
