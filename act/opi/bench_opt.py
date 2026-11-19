"""香橙派上：两项优化的效果。都从相机原图（uint8）算到最终动作，和电脑上 PyTorch 的结果核对。

  方案 0（之前）：Python 做归一化 → 骨干在 NPU 上一张接一张 → Transformer 在 NPU
  方案 1：归一化写进 NPU 模型，直接喂原图
  方案 2：方案 1 + 两路相机的骨干放在 NPU 的两个核上同时算（NPU 有 3 个核）
用法：python bench_opt.py <模型目录> <librknnrt.so> <原始样本目录>
"""
import os, sys, time, threading
import numpy as np
from rknn_c import NPU, NCHW

D, LIB, RAW = sys.argv[1:4]
os.sched_setaffinity(0, [4, 5, 6, 7])
H, W = 240, 320
top = np.fromfile(f"{RAW}/top.u8", np.uint8).reshape(H, W, 3); wrist = np.fromfile(f"{RAW}/wrist.u8", np.uint8).reshape(H, W, 3)
q = np.fromfile(f"{RAW}/state.f32", np.float32); ref = np.fromfile(f"{RAW}/ref_actions.f32", np.float32).reshape(20, 5)
n = np.fromfile(f"{D}/norm.f32", np.float32)
m_img, s_img, m_q, s_q, m_a, s_a = n[0:3], n[3:6], n[12:18], n[18:24], n[24:29], n[29:34]
FS = (1, 512, 8, 10)
qn = ((q - m_q) / s_q)[None].astype(np.float32)
head = NPU(LIB, f"{D}/head_fp16.rknn")
finish = lambda f0, f1: head.run_multi([f0, f1, qn], [NCHW, NCHW, 3]).reshape(20, 5) * s_a + m_a


def stat(name, fn, n_run=60):
    for _ in range(8):
        a = fn()
    ms = []
    for _ in range(n_run):
        t = time.perf_counter(); a = fn(); ms.append((time.perf_counter() - t) * 1e3)
    ms = np.array(ms); e = np.abs(a - ref)
    print(f"{name}: mean={ms.mean():.1f} ms  p50={np.percentile(ms, 50):.1f}  p95={np.percentile(ms, 95):.1f} | "
          f"和 PyTorch 的动作误差：最大 {e.max():.2e}，平均 {e.mean():.2e}", flush=True)


# 方案 0
bb0 = NPU(LIB, f"{D}/backbone_fp16.rknn")
prep = lambda im: np.ascontiguousarray((((im.astype(np.float32) / 255.0) - m_img) / s_img).transpose(2, 0, 1)[None])
stat("方案 0  Python 归一化 + 骨干依次算     ", lambda: finish(bb0.run(prep(top), NCHW).reshape(FS), bb0.run(prep(wrist), NCHW).reshape(FS)))
# 方案 1
bb1 = NPU(LIB, f"{D}/backbone_u8_fp16.rknn")
stat("方案 1  归一化交给 NPU + 骨干依次算     ", lambda: finish(bb1.run(top[None], u8=True).reshape(FS), bb1.run(wrist[None], u8=True).reshape(FS)))
# 方案 2：两个骨干各占 NPU 的一个核，两个线程同时发起（调用 C 库时 Python 不会互相卡住）
bbA = NPU(LIB, f"{D}/backbone_u8_fp16.rknn", core_mask=1); bbB = NPU(LIB, f"{D}/backbone_u8_fp16.rknn", core_mask=2)
out = [None, None]


def par():
    def w(i, net, im):
        out[i] = net.run(im[None], u8=True).reshape(FS)
    t1 = threading.Thread(target=w, args=(0, bbA, top)); t2 = threading.Thread(target=w, args=(1, bbB, wrist))
    t1.start(); t2.start(); t1.join(); t2.join()
    return finish(out[0], out[1])


stat("方案 2  归一化交给 NPU + 两个骨干并行   ", par)
# 分段：只测骨干
for name, fn in [("    其中骨干，依次算", lambda: (bb1.run(top[None], u8=True), bb1.run(wrist[None], u8=True))),
                 ("    其中骨干，并行  ", lambda: (par.__globals__["out"], None))]:
    if "并行" in name:
        ts = []
        for _ in range(40):
            t = time.perf_counter()
            t1 = threading.Thread(target=lambda: bbA.run(top[None], u8=True)); t2 = threading.Thread(target=lambda: bbB.run(wrist[None], u8=True))
            t1.start(); t2.start(); t1.join(); t2.join(); ts.append((time.perf_counter() - t) * 1e3)
    else:
        ts = []
        for _ in range(40):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    print(f"{name}: {np.mean(ts):.1f} ms")
ts = []
f0, f1 = bb1.run(top[None], u8=True).reshape(FS), bb1.run(wrist[None], u8=True).reshape(FS)
for _ in range(40):
    t = time.perf_counter(); finish(f0, f1); ts.append((time.perf_counter() - t) * 1e3)
print(f"    其中 Transformer: {np.mean(ts):.1f} ms")
print(f"芯片温度 {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.0f} °C")
print("OPT BENCH DONE")
