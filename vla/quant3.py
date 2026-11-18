"""找出 ACT 的 FP16 引擎在 TensorRT 里到底坏在哪一类运算上。

已知：同一个 16 位 ONNX 模型，ONNX Runtime 跑出来是对的（误差 6e-3），TensorRT 11.5 编译出的引擎是错的（误差 0.45）。
做法：每次只让一类运算（比如 Softmax）留在 32 位，其余都转 16 位，编译引擎，看输出误差。
哪一类留在 32 位之后误差恢复正常，问题就出在那一类上。
用法：python quant3.py <onnx> <校准 npz> <输出目录>
"""
import os, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16
import tensorrt as trt
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

SRC, CAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
cal = np.load(CAL); N = len(cal["state"]); HELD = list(range(N - 20, N))
base = onnx.load(SRC)
in_names = [i.name for i in base.graph.input]
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
s = ort.InferenceSession(SRC, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
ref = np.stack([s.run(None, {k: cal[k][i] for k in in_names})[0] for i in HELD]); del s
ops = {}
for n in base.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("模型里的运算：", sorted(ops.items(), key=lambda x: -x[1]), flush=True)
logger = trt.Logger(trt.Logger.INTERNAL_ERROR)


def make_fp16(block):
    m = onnx.ModelProto(); m.CopyFrom(base)
    m = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=list(block))
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n.op_type)
    for n in m.graph.node:                                   # 补上转换工具漏掉的常数（喂给 32 位运算的不动）
        if n.op_type not in ("Constant", "ConstantOfShape") or any(c in block for c in cons.get(n.output[0], [])):
            continue
        for a in list(n.attribute):
            if a.name in ("value_float", "value_floats"):
                v = np.array(a.f if a.name == "value_float" else list(a.floats), dtype=np.float16)
                n.attribute.remove(a); n.attribute.append(helper.make_attribute("value", numpy_helper.from_array(v)))
            elif a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16)))
    return m


def trt_err(m, tag):
    p = f"{OUT}/{tag}.onnx"; onnx.save(m, p)
    builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(p):
        return None, None, "读不了：" + str(parser.get_error(0))[:90]
    blob = builder.build_serialized_network(net, builder.create_builder_config())
    if blob is None:
        return None, None, "编译失败"
    eng = f"{OUT}/{tag}.engine"; open(eng, "wb").write(blob)
    with TrtRunner(EngineFromBytes(BytesFromPath(eng))) as r:
        want = list(r.get_input_metadata().keys()); fd = lambda i: {k: cal[k][i] for k in want}
        for _ in range(30):
            r.infer(fd(0))
        ms = []
        for _ in range(200):
            t = time.perf_counter(); r.infer(fd(0)); ms.append((time.perf_counter() - t) * 1e3)
        o = np.stack([r.infer(fd(i))["actions"].copy() for i in HELD])
    return float(np.abs(o - ref).mean()), float(np.mean(ms)), ""


e, ms, msg = trt_err(make_fp16([]), "all16")
print(f"全部 16 位：平均误差 {e if e is None else round(e, 4)}，推理 {ms and round(ms, 2)} ms {msg}", flush=True)
good = []
for op in sorted(ops, key=lambda x: -ops[x]):
    if op in ("Constant", "ConstantOfShape", "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Reshape", "Transpose", "Slice", "Expand", "Cast"):
        continue                                             # 这些只搬运数据或处理形状，不做小数运算
    try:
        e, ms, msg = trt_err(make_fp16([op]), "blk_" + op)
    except Exception as ex:
        e, ms, msg = None, None, f"{type(ex).__name__}: {str(ex)[:80]}"
    print(f"  {op:22s}（{ops[op]:4d} 个）留在 32 位 → 平均误差 {'—' if e is None else f'{e:.4f}'}，推理 {'—' if ms is None else f'{ms:.2f}'} ms {msg}", flush=True)
    if e is not None and e < 0.01:
        good.append((op, e, ms))
print("误差恢复正常（< 0.01）的：", good)
if good:
    best = min(good, key=lambda x: x[2])
    os.replace(f"{OUT}/blk_{best[0]}.engine", f"{OUT}/fp16fix.engine")
    print(f"选用：只把 {best[0]} 留在 32 位，引擎存为 fp16fix.engine")
print("QUANT3 DONE")
