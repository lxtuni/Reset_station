"""香橙派上：比较同一个 ACT 的几个 ONNX 版本（FP32 和几种 INT8）。4 个大核，4 线程。
用法：python bench_models.py <目录> <onnx 文件名...>
"""
import os, sys, time
import numpy as np, onnxruntime as ort

D = sys.argv[1]
d = np.load(f"{D}/sample.npz"); ref = np.load(f"{D}/ref_actions.npy"); feed = {k: d[k] for k in d.files}
os.sched_setaffinity(0, [4, 5, 6, 7])
for f in sys.argv[2:]:
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    try:
        s = ort.InferenceSession(f"{D}/{f}", so, providers=["CPUExecutionProvider"])
        for _ in range(3):
            out = s.run(None, feed)[0]
        ms = []
        for _ in range(20):
            t = time.perf_counter(); out = s.run(None, feed)[0]; ms.append((time.perf_counter() - t) * 1e3)
        ms = np.array(ms)
        sz = sum(os.path.getsize(f"{D}/{x}") for x in os.listdir(D) if x.startswith(f)) / 2**20
        print(f"{f:22s} mean={ms.mean():6.1f} ms  p95={np.percentile(ms, 95):6.1f} | 文件 {sz:4.0f} MB | 和 PyTorch 的最大误差 {np.abs(out - ref).max():.2e}", flush=True)
    except Exception as e:
        print(f"{f}: 失败 —— {type(e).__name__}: {str(e)[:200]}")
print(f"芯片温度 {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.0f} °C")
print("MODELS BENCH DONE")
