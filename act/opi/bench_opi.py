"""在香橙派（RK3588S，没有 NVIDIA 显卡）上用 ONNX Runtime 的处理器后端运行 ACT。

做三件事：
  1. 核对：同一帧输入，输出和 PyTorch 的参考结果差多少
  2. 测速：不同的线程数、不同的核心（RK3588S 有 4 个大核 A76 和 4 个小核 A55，谁在算差别很大）
  3. 剖析：时间花在哪类运算上（ONNX Runtime 自带的逐节点计时）
用法：python bench_opi.py <目录，里面有 act_rs.onnx、sample.npz、ref_actions.npy>
"""
import glob, json, os, sys, time
import numpy as np, onnxruntime as ort

D = sys.argv[1]
d = np.load(f"{D}/sample.npz"); ref = np.load(f"{D}/ref_actions.npy")
feed = {k: d[k] for k in d.files}
print("ONNX Runtime", ort.__version__, "| 可用后端:", ort.get_available_providers(), "| 核心数:", os.cpu_count())
print("输入:", {k: v.shape for k, v in feed.items()})


def run(threads, cpus=None, n=20, profile=False):
    if cpus is not None:
        os.sched_setaffinity(0, cpus)                       # 把进程绑到指定的核心上
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.enable_profiling = profile
    s = ort.InferenceSession(f"{D}/act_rs.onnx", so, providers=["CPUExecutionProvider"])
    for _ in range(3):
        out = s.run(None, feed)[0]
    ms = []
    for _ in range(n):
        t = time.perf_counter(); out = s.run(None, feed)[0]; ms.append((time.perf_counter() - t) * 1e3)
    prof = s.end_profiling() if profile else None
    os.sched_setaffinity(0, range(os.cpu_count()))
    return np.array(ms), out, prof


ms, out, _ = run(4, cpus=[4, 5, 6, 7])
print(f"核对：和 PyTorch 的最大误差 {np.abs(out - ref).max():.2e}（动作范围 ±{np.abs(ref).max():.1f}）")
for name, th, cpus in [("4 个大核 A76，4 线程", 4, [4, 5, 6, 7]), ("4 个大核 A76，2 线程", 2, [4, 5, 6, 7]), ("1 个大核 A76，1 线程", 1, [4]),
                       ("4 个小核 A55，4 线程", 4, [0, 1, 2, 3]), ("全部 8 核，8 线程", 8, None), ("全部 8 核，4 线程（不绑核）", 4, None)]:
    ms, _, _ = run(th, cpus)
    print(f"  {name:28s} mean={ms.mean():7.1f} ms  p50={np.percentile(ms, 50):7.1f}  p95={np.percentile(ms, 95):7.1f}", flush=True)

ms, _, prof = run(4, cpus=[4, 5, 6, 7], n=10, profile=True)
ev = [e for e in json.load(open(prof)) if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
tot, by = {}, {}
for e in ev:
    op = e["args"].get("op_name", "?"); by[op] = by.get(op, 0) + e["dur"]
runs = 13
total = sum(by.values()) / runs / 1000
print(f"按运算类型（4 个大核，每次推理合计 {total:.1f} ms）：")
for op, v in sorted(by.items(), key=lambda x: -x[1])[:8]:
    print(f"  {op:22s} {v / runs / 1000:7.1f} ms  ({v / runs / 1000 / total * 100:4.1f}%)")
os.remove(prof)
t = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
print(f"芯片温度 {int(t) / 1000:.0f} °C")
print("OPI BENCH DONE")
