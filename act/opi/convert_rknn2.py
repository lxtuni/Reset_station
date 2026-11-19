"""优化一：把图像的归一化写进 NPU 的模型里。

之前板子上从原图到动作要 89 ms，其中约 25 ms 花在用 Python 做归一化上（除以 255、减均值、除标准差、调整排列）。
转换时把均值和标准差告诉转换工具，生成的模型就能直接接收相机原图（0–255 的整数），归一化由 NPU 顺手完成。
均值和标准差要换算到 0–255 的尺度：(x/255 - m)/s = (x - 255m)/(255s)。
用法：python convert_rknn2.py <部署目录，里面有 backbone.onnx> <norm.f32>
"""
import os, sys
import numpy as np
from rknn.api import RKNN

D, NORM = sys.argv[1], sys.argv[2]
n = np.fromfile(NORM, np.float32)
mean, std = (n[0:3] * 255).tolist(), (n[3:6] * 255).tolist()          # 两路相机的均值和标准差相同
print("均值（0–255 尺度）", [round(x, 2) for x in mean], "标准差", [round(x, 2) for x in std])
rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", mean_values=[mean], std_values=[std])
assert rknn.load_onnx(model=f"{D}/backbone.onnx") == 0
assert rknn.build(do_quantization=False) == 0
out = f"{D}/backbone_u8_fp16.rknn"; rknn.export_rknn(out)
print(f"转换成功，{os.path.getsize(out) / 2**20:.0f} MB")
print("CONVERT2 DONE")
