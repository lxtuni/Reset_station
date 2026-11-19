"""香橙派上：骨干和 Transformer 都放在 NPU 上（两个 .rknn 文件），看能不能跑、准不准、多快。
用法：python bench_full_npu.py <目录> <librknnrt.so> <骨干 rknn> <head rknn>
"""
import os, sys, time
import numpy as np
from rknn_c import NPU, NCHW

D, LIB, BB, HD = sys.argv[1:5]
r = np.load(f"{D}/split_ref.npz"); names = ["top", "wrist"]; SHAPE = r["feat_top"].shape
os.sched_setaffinity(0, [4, 5, 6, 7])
bb = NPU(LIB, f"{D}/{BB}"); hd = NPU(LIB, f"{D}/{HD}")
feat = lambda x: bb.run(x, NCHW).reshape(SHAPE)
UNDEF = 3                                                   # 关节角度不是图像，没有“通道在前/在后”之分


def infer():
    f = [feat(r[n]) for n in names]
    return hd.run_multi([f[0], f[1], r["state"]], [NCHW, NCHW, UNDEF]).reshape(r["actions"].shape)


for _ in range(5):
    o = infer()
ms = []
for _ in range(40):
    t = time.perf_counter(); o = infer(); ms.append((time.perf_counter() - t) * 1e3)
ms = np.array(ms); e = np.abs(o - r["actions"])
print(f"骨干 {BB} + Transformer {HD}，全部在 NPU：{ms.mean():.1f} ms (p95 {np.percentile(ms, 95):.1f}) | "
      f"最终动作误差：最大 {e.max():.2e}，平均 {e.mean():.2e}（动作范围 ±{np.abs(r['actions']).max():.1f}）")
# 单独测 Transformer 那一段在 NPU 上的时间（用参考特征图当输入）
ms2 = []
for _ in range(30):
    t = time.perf_counter(); hd.run_multi([r["feat_top"], r["feat_wrist"], r["state"]], [NCHW, NCHW, UNDEF]); ms2.append((time.perf_counter() - t) * 1e3)
print(f"其中 Transformer 那一段：{np.mean(ms2):.1f} ms")
print("FULL NPU DONE")
