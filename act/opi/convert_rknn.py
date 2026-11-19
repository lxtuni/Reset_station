"""把 ACT 的骨干网络（backbone.onnx）转换成瑞芯微 NPU 用的 .rknn 文件。在电脑上运行（转换工具只有 x86 版）。

生成两个版本：
  backbone_fp16.rknn   不量化。NPU 上用 16 位小数算
  backbone_int8.rknn   量化成 8 位整数。需要校准图像（这里用示教里抽的 100 帧，两路相机各 50 帧）
输入已经是归一化后的小数图像（和 ONNX 一样），所以不在转换时设置均值和标准差。
用法：python convert_rknn.py <部署目录，里面有 backbone.onnx> <校准 npz>
"""
import os, sys
import numpy as np
from rknn.api import RKNN

D, CAL = sys.argv[1], sys.argv[2]
cal = np.load(CAL)
os.makedirs(f"{D}/rknn_calib", exist_ok=True)
lines = []
for i in range(50):                                          # 校准数据要以“一个输入一个 npy 文件”的形式给转换工具
    for k in ("top", "wrist"):
        p = f"{D}/rknn_calib/{k}_{i:03d}.npy"; np.save(p, cal[k][i]); lines.append(os.path.abspath(p))
open(f"{D}/rknn_calib/dataset.txt", "w").write("\n".join(lines) + "\n")

for tag, quant in [("fp16", False), ("int8", True)]:
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588")
    ret = rknn.load_onnx(model=f"{D}/backbone.onnx")
    if ret != 0:
        print(f"{tag}：读 ONNX 失败"); continue
    ret = rknn.build(do_quantization=quant, dataset=f"{D}/rknn_calib/dataset.txt" if quant else None)
    if ret != 0:
        print(f"{tag}：转换失败"); continue
    out = f"{D}/backbone_{tag}.rknn"
    rknn.export_rknn(out)
    print(f"{tag}：转换成功，{os.path.getsize(out) / 2**20:.0f} MB")
    rknn.release()
print("CONVERT DONE")
