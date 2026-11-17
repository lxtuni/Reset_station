"""测 TensorRT 引擎的延迟，并和 PyTorch 的参考输出对比。

用法（和引擎、sample.npz、ref_actions.npy 放在同一个文件夹）：
    python trt_bench.py act_trt86_fp32.engine [act_trt86_fp16.engine ...]
"""
import sys
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

d = np.load("sample.npz"); ref = np.load("ref_actions.npy")
feed = {k: d[k] for k in d.files}                     # 输入名和 npz 里的键一致

for path in sys.argv[1:] or ["act_fp16.engine"]:
    with TrtRunner(EngineFromBytes(BytesFromPath(path))) as runner:
        for _ in range(20):                           # 预热
            out = runner.infer(feed)
        ms = []
        for _ in range(200):
            out = runner.infer(feed)
            ms.append(runner.last_inference_time() * 1000)
    ms = np.array(ms)
    diff = np.abs(out["actions"] - ref).max()
    print(f"{path}: mean={ms.mean():.1f}ms p50={np.percentile(ms, 50):.1f} p95={np.percentile(ms, 95):.1f} | "
          f"和 PyTorch FP32 的最大误差 {diff:.2e}")
