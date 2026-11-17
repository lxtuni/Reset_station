"""生成三样给人看的东西，存到 outputs/visuals：
  loss_curve.png   训练损失曲线（纯模仿 6 万步 + DAgger 接着训练 3 万步）
  samples.png      数据集里的几帧（上排顶部相机，下排腕部相机）
  act_success_*.gif / act_fail_*.gif   ACT 控制机械臂抓快船的录像（左：顶部相机，右：腕部相机）

用法（仓库根目录）：python act/make_visuals.py MODEL_DIR
"""
import os, re, sys
import numpy as np, torch, imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

OUT = "outputs/visuals"; os.makedirs(OUT, exist_ok=True)

# 1. 损失曲线。日志每 200 步打印一行
def losses(path):
    return np.array([float(x) for x in re.findall(r"loss:([0-9.]+)", open(path, errors="ignore").read())])
a, b = losses("outputs_train_full.log"), losses("outputs_train_d1.log")
xa = (np.arange(len(a)) + 1) * 200; xb = 30000 + (np.arange(len(b)) + 1) * 200
plt.figure(figsize=(9, 4))
plt.plot(xa, a, label="imitation only (400 demos)")
plt.plot(xb, b, label="DAgger fine-tune from step 30k (560 episodes)")
plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("loss (L1 + 10 x KL), log scale"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/loss_curve.png", dpi=130); plt.close()

# 2. 数据样例：一个示教回合里均匀取 5 帧
d = np.load("data/raw_full_3/ep_0000.npz")
T = len(d["action"]); idx = np.linspace(0, T - 1, 5).astype(int)
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
for j, t in enumerate(idx):
    ax[0, j].imshow(d["top"][t]); ax[0, j].set_title(f"top cam, t={t * 0.05:.2f}s", fontsize=10)
    ax[1, j].imshow(d["wrist"][t]); ax[1, j].set_title("action=" + np.array2string(d["action"][t], precision=2), fontsize=8)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/samples.png", dpi=110); plt.close()
print("loss 点数:", len(a), len(b), "| 样例回合帧数:", T, "船速 v0 =", d["v0"])

# 3. 录像：最终模型，快船，每步都重新推理
M = sys.argv[1]
policy = ACTPolicy.from_pretrained(M).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
env = make_env((0.10, 0.20)); cams = Cams(env)
got = {"success": 0, "fail": 0}
for seed in range(40):
    if min(got.values()) >= 2:
        break
    env.reset(seed=seed); frames = []
    while True:
        top, wrist = cams(); frames.append(np.concatenate([top, wrist], axis=1))
        bt = pre({"observation.images.top": f(top), "observation.images.wrist": f(wrist),
                  "observation.state": torch.from_numpy(joints(env)).unsqueeze(0), "task": [""]})
        with torch.inference_mode():
            act = post(policy.predict_action_chunk(bt))[0, 0].cpu().numpy()
        obs, r, term, trunc, info = env.step(np.array(act, dtype=np.float32))
        if term or trunc:
            break
    kind = "success" if info["success"] else "fail"
    if got[kind] < 2:
        frames += [frames[-1]] * 20                        # 结尾停 1 秒
        imageio.mimsave(f"{OUT}/act_{kind}_{got[kind]}.gif", frames, duration=0.05, loop=0)
        print(f"seed {seed}: {kind}，{len(frames) - 20} 步，船速 {info['v0']}")
        got[kind] += 1
