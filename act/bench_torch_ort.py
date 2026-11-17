"""同一帧输入上测 PyTorch 和 ONNX Runtime（GPU）的网络推理延迟。和 trt_bench.py 的方法一致：预热 20 次，计时 200 次。

用法：python bench_torch_ort.py MODEL_DIR DEPLOY_DIR ONNX文件
DEPLOY_DIR 里要有 sample.npz（官方 ACT 的键是 image/state，复位台 ACT 是 top/wrist/state）。
"""
import sys, time
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy, OBS_IMAGES, OBS_STATE

policy = ACTPolicy.from_pretrained(sys.argv[1]).to("cuda").eval()
d = np.load(f"{sys.argv[2]}/sample.npz")
img_keys = [k for k in d.files if k != "state"]
batch = {OBS_STATE: torch.from_numpy(d["state"]).cuda(), OBS_IMAGES: [torch.from_numpy(d[k]).cuda() for k in img_keys]}


def stat(name, ms):
    ms = np.array(ms)
    print(f"{name}: mean={ms.mean():.1f}ms p50={np.percentile(ms, 50):.1f} p95={np.percentile(ms, 95):.1f}")


with torch.inference_mode():
    for _ in range(20):
        policy.model(batch)
    torch.cuda.synchronize(); ms = []
    for _ in range(200):
        t = time.perf_counter(); policy.model(batch); torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
stat("PyTorch        ", ms)

import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
sess = ort.InferenceSession(sys.argv[3], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
feed = {k: d[k] for k in d.files}
for _ in range(20):
    sess.run(None, feed)
ms = []
for _ in range(200):
    t = time.perf_counter(); sess.run(None, feed); ms.append((time.perf_counter() - t) * 1000)
stat("ONNX Runtime GPU", ms)
