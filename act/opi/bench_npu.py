"""香橙派上：骨干网络在 NPU 上运行，其余部分（Transformer）在处理器的 4 个大核上运行。

每次推理：
  两路相机的图像 → NPU（.rknn）各跑一次骨干 → 两张特征图 → 处理器（head.onnx，ONNX Runtime）→ 20 步动作
核对：NPU 算出的特征图、最终动作，分别和电脑上 PyTorch 的结果比。然后分段计时。
用法：python bench_npu.py <目录> <librknnrt.so 的路径> <rknn 文件名...>
"""
import os, sys, time
import numpy as np, onnxruntime as ort
from rknn_c import NPU, NCHW, NHWC

D, LIB = sys.argv[1], sys.argv[2]
r = np.load(f"{D}/split_ref.npz")
names = ["top", "wrist"]; SHAPE = r["feat_top"].shape       # (1, 512, 8, 10)
os.sched_setaffinity(0, [4, 5, 6, 7])
so = ort.SessionOptions(); so.intra_op_num_threads = 4
head = ort.InferenceSession(f"{D}/head.onnx", so, providers=["CPUExecutionProvider"])
head_run = lambda f: head.run(None, {"feat_top": f[0], "feat_wrist": f[1], "state": r["state"]})[0]

bb = ort.InferenceSession(f"{D}/backbone.onnx", so, providers=["CPUExecutionProvider"])
for _ in range(3):
    f = [bb.run(None, {"image": r[n]})[0] for n in names]; o = head_run(f)
tb, th = [], []
for _ in range(15):
    t0 = time.perf_counter(); f = [bb.run(None, {"image": r[n]})[0] for n in names]; t1 = time.perf_counter(); o = head_run(f); t2 = time.perf_counter()
    tb.append((t1 - t0) * 1e3); th.append((t2 - t1) * 1e3)
print(f"全在处理器上：骨干 {np.mean(tb):.1f} ms + 其余 {np.mean(th):.1f} ms = {np.mean(tb) + np.mean(th):.1f} ms | 动作误差 {np.abs(o - r['actions']).max():.2e}", flush=True)


def layouts(y):                                             # NPU 输出的排列方式不确定，两种都还原出来
    n, c, h, w = SHAPE
    return {"通道在前": y.reshape(n, c, h, w), "通道在后": y.reshape(n, h, w, c).transpose(0, 3, 1, 2)}


for f_rknn in sys.argv[3:]:
    try:
        npu = NPU(LIB, f"{D}/{f_rknn}")
        # 输入和输出各有两种可能的排列，用参考特征图找出误差最小的组合
        best = None
        for fin, fname in [(NCHW, "输入通道在前"), (NHWC, "输入通道在后")]:
            x = r["top"] if fin == NCHW else np.ascontiguousarray(r["top"].transpose(0, 2, 3, 1))
            try:
                y = npu.run(x, fin)
            except Exception as e:
                continue
            for lname, yy in layouts(y).items():
                e = float(np.abs(yy - r["feat_top"]).mean())
                if best is None or e < best[0]:
                    best = (e, fin, fname, lname)
        _, fin, fname, lname = best
        prep = (lambda a: a) if fin == NCHW else (lambda a: np.ascontiguousarray(a.transpose(0, 2, 3, 1)))
        xs = [prep(r[n]) for n in names]
        run = lambda x: layouts(npu.run(x, fin))[lname]
        for _ in range(5):
            f = [run(x) for x in xs]
        tb, th = [], []
        for _ in range(40):
            t0 = time.perf_counter(); f = [run(x) for x in xs]; t1 = time.perf_counter(); o = head_run(f); t2 = time.perf_counter()
            tb.append((t1 - t0) * 1e3); th.append((t2 - t1) * 1e3)
        ef = [np.abs(f[i] - r[f"feat_{n}"]) for i, n in enumerate(names)]
        tot = np.array(tb) + np.array(th)
        print(f"{f_rknn}（{fname}，输出{lname}）: 骨干在 NPU {np.mean(tb):.1f} ms（两路相机）+ 其余在处理器 {np.mean(th):.1f} ms = {tot.mean():.1f} ms (p95 {np.percentile(tot, 95):.1f})\n"
              f"    特征图误差：最大 {max(e.max() for e in ef):.2e}，平均 {np.mean([e.mean() for e in ef]):.2e}（特征图范围 0 到 {r['feat_top'].max():.0f}）| "
              f"最终动作误差：最大 {np.abs(o - r['actions']).max():.2e}，平均 {np.abs(o - r['actions']).mean():.2e}（动作范围 ±{np.abs(r['actions']).max():.1f}）", flush=True)
        npu.close()
    except Exception as e:
        print(f"{f_rknn}：失败 —— {type(e).__name__}: {str(e)[:300]}", flush=True)
print(f"芯片温度 {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.0f} °C")
print("NPU BENCH DONE")
