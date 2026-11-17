"""性能剖析：一次推理的时间花在网络的哪一部分。

用法（仓库根目录）：python act/profile_torch.py MODEL_DIR DEPLOY_DIR
两种看法：
  1. 按模块：ResNet 图像骨干 / Transformer 编码器 / Transformer 解码器（用 hook + cuda 同步计时）
  2. 按算子：torch.profiler 列出最耗时的 12 种算子
"""
import sys, time
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity
from lerobot.policies.act.modeling_act import ACTPolicy, OBS_IMAGES, OBS_STATE

policy = ACTPolicy.from_pretrained(sys.argv[1]).to("cuda").eval()
d = np.load(f"{sys.argv[2]}/sample.npz")
names = [k.split(".")[-1] for k in policy.config.image_features]
batch = {OBS_STATE: torch.from_numpy(d["state"]).cuda(), OBS_IMAGES: [torch.from_numpy(d[n]).cuda() for n in names]}
m = policy.model
run = lambda: m(batch)[0]

with torch.inference_mode():
    for _ in range(30):
        run()                                             # 预热
    torch.cuda.synchronize()
    ms = []
    for _ in range(200):
        t = time.perf_counter(); run(); torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
    ms = np.array(ms)
    print(f"整网：mean={ms.mean():.1f} ms  p50={np.percentile(ms, 50):.1f}  p95={np.percentile(ms, 95):.1f}")

    # 1. 按模块。每个模块前后做 cuda 同步，所以各段加起来会比整网略多
    acc = {}
    def hook(name):
        def pre(mod, inp):
            torch.cuda.synchronize(); mod._t = time.perf_counter()
        def post(mod, inp, out):
            torch.cuda.synchronize(); acc[name] = acc.get(name, 0) + (time.perf_counter() - mod._t) * 1000
        return pre, post
    hs = []
    for name in ["backbone", "encoder", "decoder"]:
        p, q = hook(name); mod = getattr(m, name)
        hs += [mod.register_forward_pre_hook(p), mod.register_forward_hook(q)]
    N = 100
    t = time.perf_counter()
    for _ in range(N):
        run()
    torch.cuda.synchronize(); total = (time.perf_counter() - t) * 1000 / N
    for h in hs:
        h.remove()
    print(f"按模块（带同步，每次推理合计 {total:.1f} ms）：")
    for name, v in acc.items():
        print(f"  {name:9s} {v / N:6.1f} ms  ({v / N / total * 100:.0f}%)")
    print(f"  其余      {total - sum(acc.values()) / N:6.1f} ms")

    # 2. 按算子
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            run()
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=40))
