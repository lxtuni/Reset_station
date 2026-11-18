"""录一段微调后的 SmolVLA 抓快船的动图（左：顶部相机，右：腕部相机，就是模型看到的画面）。
去噪 3 次、每步都重新推理。用法（仓库根目录，PYTHONPATH=act）：python vla/make_gif_vla.py MODEL_DIR OUT.gif [船速下限 上限]
"""
import sys
import numpy as np, torch, imageio
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

M, OUT = sys.argv[1], sys.argv[2]
vr = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.15, 0.20)
policy = SmolVLAPolicy.from_pretrained(M).to("cuda").eval()
policy.config.num_steps = 3
if hasattr(policy.model, "config"):
    policy.model.config.num_steps = 3
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
env = make_env(vr); cams = Cams(env)
for seed in range(20):
    env.reset(seed=seed); policy.reset(); frames = []
    while True:
        top, wrist = cams(); frames.append(np.concatenate([top, wrist], axis=1))
        b = pre({"observation.images.camera1": f(top), "observation.images.camera2": f(wrist),
                 "observation.state": torch.from_numpy(joints(env)).unsqueeze(0),
                 "task": ["intercept the drifting boat and grasp its mast"]})
        with torch.inference_mode():
            a = post(policy.predict_action_chunk(b))[0, 0].cpu().numpy()
        obs, r, term, trunc, info = env.step(np.array(a, dtype=np.float32))
        if term or trunc:
            break
    print(f"seed {seed}: {'成功' if info['success'] else '失败'}，{len(frames)} 步，船速 {float(info['v0']):.3f}")
    if info["success"] and 50 <= len(frames) <= 110:       # 选一个长度适中的成功回合
        frames += [frames[-1]] * 20                         # 结尾停 1 秒
        imageio.mimsave(OUT, frames, duration=0.05, loop=0)
        print("已保存", OUT); break
