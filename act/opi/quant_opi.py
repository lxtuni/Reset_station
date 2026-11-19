"""给香橙派准备 INT8 版本的 ACT。在本机（有校准数据的地方）运行，生成的 ONNX 文件再传到板子上测。

香橙派上的剖析结果：卷积占 65%，全连接（Gemm）占 28%。ARM 处理器对 8 位整数运算有专门的指令，
所以把这两类运算量化成 INT8 是在这块硬件上最直接的加速办法（在 4090 上量化卷积没有收益，因为那里卷积不是瓶颈）。
生成三个版本：
  int8_conv.onnx      只量化卷积（每个通道一个刻度，校准时去掉极端值）
  int8_conv_gemm.onnx 卷积和全连接都量化
  int8_dyn.onnx       只量化全连接和矩阵乘法的权重，不需要校准数据（动态量化）
用法：python quant_opi.py <目录，里面有 act_rs.onnx> <校准 npz>
"""
import sys, time
import numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)

D, CAL = sys.argv[1], sys.argv[2]
SRC = f"{D}/act_rs.onnx"
cal = np.load(CAL); N = len(cal["state"]); HELD = list(range(N - 20, N))
names = [i.name for i in onnx.load(SRC, load_external_data=False).graph.input]
feed = lambda i: {k: cal[k][i] for k in names}


class Reader(CalibrationDataReader):
    def __init__(self, n):
        self.i, self.n = 0, n
    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return feed(self.i - 1)


def outs(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.stack([s.run(None, feed(i))[0] for i in HELD])


ref = outs(SRC)
print(f"FP32 模型在 20 帧上的输出范围 ±{np.abs(ref).max():.1f}", flush=True)
for tag, ops in [("int8_conv", ["Conv"]), ("int8_conv_gemm", ["Conv", "Gemm"])]:
    t = time.time(); p = f"{D}/{tag}.onnx"
    quantize_static(SRC, p, Reader(120), quant_format=QuantFormat.QDQ, per_channel=True, op_types_to_quantize=ops,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile)
    d = np.abs(outs(p) - ref)
    print(f"{tag}: 量化 {'、'.join(ops)}，用时 {time.time() - t:.0f} 秒 | 和 FP32 比：最大误差 {d.max():.2e}，平均 {d.mean():.2e}", flush=True)
p = f"{D}/int8_dyn.onnx"
quantize_dynamic(SRC, p, op_types_to_quantize=["Gemm", "MatMul"], weight_type=QuantType.QInt8)
d = np.abs(outs(p) - ref)
print(f"int8_dyn: 动态量化 Gemm、MatMul | 和 FP32 比：最大误差 {d.max():.2e}，平均 {d.mean():.2e}")
print("QUANT OPI DONE")
