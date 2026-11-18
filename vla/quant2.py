"""ACT 的量化第二轮：弄清楚 FP16 为什么坏了，并换两种做法再试。

第一轮的结果：整体转成 FP16 的引擎闭环只有 4/180；INT8 引擎编译失败。这里做三件事：
  A. 诊断：直接用 ONNX Runtime 跑 FP16 模型，和 FP32 比。如果 ONNX Runtime 上也坏，问题在 16 位本身（数值溢出），不在 TensorRT
  B. 混合精度：用工具自动找出“必须留在 32 位”的节点，其余转 16 位（它会反复试，每次用真实输入验证输出没变）
  C. INT8 改为“每层一个量化刻度”（第一轮是每个通道一个刻度，TensorRT 没法把它和后面的乘法合并），只量化卷积
用法：python quant2.py <onnx> <校准 npz> <输出目录>
"""
import os, sys, time, traceback
import numpy as np, onnx, onnxruntime as ort

SRC, CAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
cal = np.load(CAL); N = len(cal["state"])
in_names = [i.name for i in onnx.load(SRC, load_external_data=False).graph.input]
feed = lambda i: {k: cal[k][i] for k in in_names}
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
HELD = list(range(N - 20, N))


def ort_out(path):
    s = ort.InferenceSession(path, providers=PROV)
    return np.stack([s.run(None, feed(i))[0] for i in HELD])


ref = ort_out(SRC)
print(f"输出范围 ±{np.abs(ref).max():.1f}", flush=True)
paths = {}

# ---------- A. 诊断 ----------
naive = OUT.rstrip("/").rsplit("_", 1)[0] + "_Conv/fp16.onnx"       # 第一轮生成的“全部转 16 位”的模型
try:
    o = ort_out(naive); d = np.abs(o.astype(np.float32) - ref)
    print(f"A. 全部转 16 位的模型，用 ONNX Runtime 跑：最大误差 {np.nanmax(d):.2e}，平均 {np.nanmean(d):.2e}，"
          f"输出里有 {int(np.isnan(o).sum())} 个 NaN、{int(np.isinf(o).sum())} 个无穷大", flush=True)
except Exception as e:
    print(f"A. 诊断失败 —— {type(e).__name__}: {str(e)[:200]}", flush=True)

# ---------- B. 混合精度 ----------
try:
    from onnxconverter_common import auto_mixed_precision
    m = onnx.load(SRC); t = time.time()
    f0 = feed(0)
    mm = auto_mixed_precision.auto_convert_mixed_precision(m, f0, rtol=0.02, atol=0.002, keep_io_types=True)
    p = f"{OUT}/fp16mix.onnx"; onnx.save(mm, p, save_as_external_data=True, all_tensors_to_one_file=True, location="fp16mix.onnx.data")
    n16 = sum(1 for n in mm.graph.node if n.op_type == "Cast")
    o = ort_out(p); d = np.abs(o - ref)
    print(f"B. 混合精度模型已生成，用时 {time.time() - t:.0f} 秒；ONNX Runtime 上最大误差 {d.max():.2e}，平均 {d.mean():.2e}；"
          f"图里有 {n16} 个类型转换节点（32 位和 16 位区域的边界）", flush=True)
    paths["fp16mix"] = p
except Exception as e:
    print(f"B. 混合精度失败 —— {type(e).__name__}: {str(e)[:300]}")
    print("   位置：", " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}" for x in traceback.extract_tb(e.__traceback__)[-3:]), flush=True)

# ---------- C. INT8，每层一个刻度 ----------
try:
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, n):
            self.i, self.n = 0, n
        def get_next(self):
            if self.i >= self.n:
                return None
            self.i += 1
            return feed(self.i - 1)

    pp = f"{OUT}/pre.onnx"
    try:
        quant_pre_process(SRC, pp, skip_symbolic_shape=True)
    except Exception as e:
        print(f"（量化前的图整理失败，用原图：{type(e).__name__}）"); pp = SRC
    for tag, method in [("int8", CalibrationMethod.MinMax), ("int8pct", CalibrationMethod.Percentile)]:
        p = f"{OUT}/{tag}.onnx"
        quantize_static(pp, p, Reader(min(N - 20, 150)), quant_format=QuantFormat.QDQ, per_channel=False,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"],
                        calibrate_method=method,
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
        o = ort_out(p); d = np.abs(o - ref)
        print(f"C. INT8（{'最大最小值' if tag == 'int8' else '去掉极端值的百分位'}校准）模型：ONNX Runtime 上最大误差 {d.max():.2e}，平均 {d.mean():.2e}", flush=True)
        paths[tag] = p
except Exception as e:
    print(f"C. INT8 失败 —— {type(e).__name__}: {str(e)[:300]}")
    print("   位置：", " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}" for x in traceback.extract_tb(e.__traceback__)[-3:]), flush=True)

# ---------- 编译引擎、比精度、测速 ----------
import tensorrt as trt
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
logger = trt.Logger(trt.Logger.ERROR)
for tag, p in [("fp32", SRC)] + list(paths.items()):
    try:
        eng = f"{OUT}/{tag}.engine"
        builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
        if not parser.parse_from_file(p):
            print(f"{tag}：TensorRT 读不了 ——", [str(parser.get_error(i))[:250] for i in range(min(parser.num_errors, 2))]); continue
        blob = builder.build_serialized_network(net, builder.create_builder_config())
        if blob is None:
            print(f"{tag}：引擎编译失败"); continue
        open(eng, "wb").write(blob)
        with TrtRunner(EngineFromBytes(BytesFromPath(eng))) as r:
            want = list(r.get_input_metadata().keys()); fd = lambda i: {k: cal[k][i] for k in want}
            for _ in range(50):
                r.infer(fd(0))
            ms = []
            for _ in range(300):
                t = time.perf_counter(); r.infer(fd(0)); ms.append((time.perf_counter() - t) * 1e3)
            o = np.stack([r.infer(fd(i))["actions"].copy() for i in HELD])
        d = np.abs(o - ref)
        print(f"{tag} 引擎: 推理 {np.mean(ms):.2f} ms (p95 {np.percentile(ms, 95):.2f}) | {os.path.getsize(eng) / 2**20:.0f} MB | "
              f"和 FP32（ONNX Runtime）比：最大误差 {d.max():.2e}，平均 {d.mean():.2e}", flush=True)
    except Exception as e:
        print(f"{tag}：失败 —— {type(e).__name__}: {str(e)[:300]}", flush=True)
print("QUANT2 DONE")
