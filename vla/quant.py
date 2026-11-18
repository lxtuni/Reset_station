"""在 RTX 4090 上给一个导出的 ONNX 模型做三种精度的 TensorRT 引擎并比较：FP32、FP16、INT8。

TensorRT 11 取消了旧的 FP16 / INT8 开关和校准器，引擎的精度完全由 ONNX 模型自己决定，所以要先改 ONNX：
  FP16：把模型里的 32 位小数权重和运算改成 16 位（输入输出保持 32 位）
  INT8：用校准数据统计每层的数值范围，在卷积 / 全连接前后插入“量化 / 反量化”节点（QDQ），TensorRT 看到它们就用 8 位整数算
然后对每个版本：编译引擎 → 测速 → 和 FP32 引擎在 20 帧没参与校准的输入上比输出。
只需要 numpy、onnx、onnxruntime、tensorrt，不需要 PyTorch。
用法：python quant.py <onnx 文件> <校准数据 npz> <输出目录> [INT8 量化哪些运算，默认 Conv；可写 Conv,MatMul,Gemm]
"""
import os, sys, time, traceback
import numpy as np, onnx

SRC, CAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OPS = (sys.argv[4] if len(sys.argv) > 4 else "Conv").split(",")
os.makedirs(OUT, exist_ok=True)
cal = np.load(CAL); N = len(cal["state"])
in_names = [i.name for i in onnx.load(SRC, load_external_data=False).graph.input]
feed = lambda i: {k: cal[k][i] for k in in_names}
print("模型输入：", in_names, "| 校准样本", N, "| INT8 量化的运算：", OPS, flush=True)
paths = {"fp32": SRC}

# ---------- FP16 ----------
try:
    from onnxconverter_common import float16
    m = onnx.load(SRC)
    # op_block_list=[]：所有运算都转成 16 位。转换工具默认会让 Resize、Range、Min、Max 等留在 32 位并在两边插类型转换，
    # 但它插得不完整，会出现“一个除法的两个输入分别是 32 位和 16 位”的情况，TensorRT 11 类型检查严格，直接拒绝。
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=[])
    # 转换工具只改“以张量形式存的常数”。导出器还会把一些常数直接写成节点属性里的小数（value_float），
    # 或写在 ConstantOfShape 的填充值里，这些它不碰，于是留下了 32 位的常数。这里补上。
    from onnx import TensorProto, helper, numpy_helper
    nfix = 0
    for n in m16.graph.node:
        if n.op_type == "Constant":
            for a in list(n.attribute):
                if a.name in ("value_float", "value_floats"):
                    v = np.array(a.f if a.name == "value_float" else list(a.floats), dtype=np.float16)
                    n.attribute.remove(a); n.attribute.append(helper.make_attribute("value", numpy_helper.from_array(v))); nfix += 1
                elif a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16))); nfix += 1
        elif n.op_type == "ConstantOfShape":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16))); nfix += 1
        elif n.op_type == "Cast" and n.output[0] not in {o.name for o in m16.graph.output}:
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT and not n.name.startswith("graph_"):
                    a.i = TensorProto.FLOAT16; nfix += 1
    print(f"FP16：另外修正了 {nfix} 处留在 32 位的常数或类型转换")
    p = f"{OUT}/fp16.onnx"
    onnx.save(m16, p, save_as_external_data=True, all_tensors_to_one_file=True, location="fp16.onnx.data")
    paths["fp16"] = p; del m, m16
    print("FP16 模型已生成", flush=True)
except Exception as e:
    print(f"FP16 转换失败 —— {type(e).__name__}: {str(e)[:300]}", flush=True)

# ---------- INT8（QDQ） ----------
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
        quant_pre_process(SRC, pp, skip_symbolic_shape=True, save_as_external_data=True, all_tensors_to_one_file=True,
                          external_data_location="pre.onnx.data")
    except Exception as e:
        print(f"（量化前的图整理失败，直接用原图：{type(e).__name__}）"); pp = SRC
    p = f"{OUT}/int8.onnx"; t = time.time()
    quantize_static(pp, p, Reader(min(N - 20, 100)), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, op_types_to_quantize=OPS,
                    calibrate_method=CalibrationMethod.MinMax, use_external_data_format=True,
                    # TensorRT 要求对称量化（零点为 0）；偏置不量化——量化工具会把偏置存成 32 位整数，TensorRT 不接受
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
    g = onnx.load(p, load_external_data=False).graph
    nq = sum(n.op_type == "QuantizeLinear" for n in g.node)
    paths["int8"] = p
    print(f"INT8（QDQ）模型已生成，用时 {time.time() - t:.0f} 秒，插入了 {nq} 个量化节点", flush=True)
except Exception as e:
    print(f"INT8 量化失败 —— {type(e).__name__}: {str(e)[:300]}")
    print("   位置：", " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}" for x in traceback.extract_tb(e.__traceback__)[-3:]), flush=True)

# ---------- 编译引擎、测速、比精度 ----------
import tensorrt as trt
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
logger = trt.Logger(trt.Logger.ERROR)
HELD = list(range(N - 20, N))                              # 最后 20 帧不参与校准，只用来比精度
ref = None
for prec in ["fp32", "fp16", "int8"]:
    if prec not in paths:
        continue
    try:
        eng = f"{OUT}/{prec}.engine"
        if not os.path.exists(eng):
            builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(paths[prec]):
                print(f"{prec}：TensorRT 读不了这个 ONNX ——", [str(parser.get_error(i))[:250] for i in range(min(parser.num_errors, 2))]); continue
            t = time.time(); blob = builder.build_serialized_network(net, builder.create_builder_config())
            if blob is None:
                print(f"{prec}：引擎编译失败"); continue
            open(eng, "wb").write(blob); bt = time.time() - t
        else:
            bt = 0
        with TrtRunner(EngineFromBytes(BytesFromPath(eng))) as r:
            want = list(r.get_input_metadata().keys())
            fd = lambda i: {k: cal[k][i] for k in want}
            for _ in range(20):
                r.infer(fd(0))
            ms = []
            for _ in range(100):
                t = time.perf_counter(); r.infer(fd(0)); ms.append((time.perf_counter() - t) * 1e3)
            outs = np.stack([r.infer(fd(i))["actions"].copy() for i in HELD])
        if ref is None:
            ref = outs
        d = np.abs(outs - ref)
        print(f"{prec}: 推理 {np.mean(ms):.2f} ms (p95 {np.percentile(ms, 95):.2f}) | 引擎 {os.path.getsize(eng) / 2**20:.0f} MB | "
              f"编译 {bt:.0f} 秒 | 和 FP32 引擎比：最大误差 {d.max():.2e}，平均误差 {d.mean():.2e}（输出范围 ±{np.abs(ref).max():.1f}）", flush=True)
    except Exception as e:
        print(f"{prec}：失败 —— {type(e).__name__}: {str(e)[:300]}", flush=True)
print("QUANT DONE")
