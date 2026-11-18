"""SmolVLA 的推理耗时，以及它由哪两部分组成。显卡空闲时运行。

SmolVLA 一次推理分两段：
  1. 前缀：视觉语言模型读两张图、任务描述、关节角度，只做一次
  2. 去噪：动作专家从随机噪声出发，迭代 num_steps 次，逐步修成 50 步动作
所以 耗时 ≈ 前缀耗时 + num_steps × 每次迭代耗时。改变 num_steps 测几次，用直线拟合就能把两部分分开。

用法（仓库根目录，PYTHONPATH=act）：python vla/bench_vla.py MODEL_DIR
"""
import sys, time
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

M = sys.argv[1]
policy = SmolVLAPolicy.from_pretrained(M).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
env = make_env((0.06, 0.10)); cams = Cams(env); env.reset(seed=0)
top, wrist = cams(); q = joints(env)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
obs = lambda: {"observation.images.camera1": f(top), "observation.images.camera2": f(wrist),
               "observation.state": torch.from_numpy(q).unsqueeze(0), "task": ["intercept the drifting boat and grasp its mast"]}
n_par = sum(p.numel() for p in policy.parameters())
print(f"参数量 {n_par / 1e6:.0f} M，显卡 {torch.cuda.get_device_name(0)}")


def bench(n_steps, amp=False, runs=40):
    policy.config.num_steps = n_steps
    if hasattr(policy.model, "config"):
        policy.model.config.num_steps = n_steps
    tp, ti, to = [], [], []
    for i in range(runs + 5):
        policy.reset()
        t0 = time.perf_counter(); b = pre(obs()); torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            c = policy.predict_action_chunk(b)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        a = post(c.float())[0].cpu().numpy(); t3 = time.perf_counter()
        if i >= 5:
            tp.append((t1 - t0) * 1e3); ti.append((t2 - t1) * 1e3); to.append((t3 - t2) * 1e3)
    ti = np.array(ti)
    return np.mean(tp), ti.mean(), np.percentile(ti, 95), np.mean(to), a


res = {}
for n in [10, 5, 3, 2, 1]:
    p, m, p95, o, _ = bench(n)
    res[n] = m
    print(f"去噪 {n:2d} 次：前处理 {p:.1f} ms | 网络 {m:.1f} ms (p95 {p95:.1f}) | 后处理 {o:.1f} ms | 合计 {p + m + o:.1f} ms")
xs = np.array(list(res)); ys = np.array(list(res.values()))
b_, a_ = np.polyfit(xs, ys, 1)
print(f"拟合：网络耗时 ≈ {a_:.1f} ms（前缀，只做一次） + {b_:.1f} ms × 去噪次数")
print(f"去噪 10 次时，前缀占 {a_ / res[10] * 100:.0f}%，去噪占 {b_ * 10 / res[10] * 100:.0f}%")
try:                                                        # 半精度（4090 有专用加速单元）
    p, m, p95, o, a16 = bench(10, amp=True)
    _, _, _, _, a32 = bench(10)
    print(f"半精度（autocast），去噪 10 次：网络 {m:.1f} ms (p95 {p95:.1f})")
except Exception as e:
    print("半精度运行失败：", type(e).__name__, str(e)[:150])
print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
