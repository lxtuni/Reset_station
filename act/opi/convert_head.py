"""尝试把 ACT 除骨干以外的部分（Transformer 编码器和解码器，head.onnx）也转换成 NPU 格式。不量化，16 位小数。
Transformer 里的注意力运算 NPU 不一定都支持，转换工具会报告哪些运算被留在处理器上。
用法：python convert_head.py <部署目录>
"""
import os, sys
from rknn.api import RKNN

D = sys.argv[1]
rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588")
if rknn.load_onnx(model=f"{D}/head.onnx") != 0:
    print("head：读 ONNX 失败"); sys.exit(0)
if rknn.build(do_quantization=False) != 0:
    print("head：转换失败"); sys.exit(0)
out = f"{D}/head_fp16.rknn"; rknn.export_rknn(out)
print(f"head：转换成功，{os.path.getsize(out) / 2**20:.0f} MB")
print("CONVERT HEAD DONE")
