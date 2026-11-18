"""在单独的环境（ortgpu）里：对导出的 SmolVLA ONNX 做显卡上的 ONNX Runtime 测速，并尝试编译成 TensorRT 引擎。
只需要 numpy、onnxruntime-gpu、tensorrt，不需要 PyTorch 和 LeRobot。
用法：python ort_trt_bench.py <导出目录> <onnx 文件名>
"""
import os, sys, time
import numpy as np

D, F = sys.argv[1], sys.argv[2]
d = np.load(f"{D}/sample.npz"); ref = np.load(f"{D}/ref.npy")
path = f"{D}/{F}"


def stat(name, fn, n=60):
    for _ in range(10):
        out = fn()
    ms = []
    for _ in range(n):
        t = time.perf_counter(); out = fn(); ms.append((time.perf_counter() - t) * 1e3)
    ms = np.array(ms)
    print(f"{name}: mean={ms.mean():.1f} ms  p50={np.percentile(ms, 50):.1f}  p95={np.percentile(ms, 95):.1f} | "
          f"和 PyTorch（32 位）的最大误差 {np.abs(out - ref).max():.2e}", flush=True)


# ---------- ONNX Runtime（显卡） ----------
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("ONNX Runtime", ort.__version__, "providers:", sess.get_providers())
    feed = {i.name: d[i.name] for i in sess.get_inputs()}
    stat("ONNX Runtime", lambda: sess.run(None, feed)[0])
    del sess
except Exception as e:
    print(f"ONNX Runtime 失败 —— {type(e).__name__}: {str(e)[:400]}")

# ---------- TensorRT ----------
try:
    import tensorrt as trt
    print("TensorRT", trt.__version__, flush=True)
    logger = trt.Logger(trt.Logger.ERROR)
    for prec in ["fp32", "fp16"]:
        eng_path = f"{D}/{F}.{prec}.engine"
        if not os.path.exists(eng_path):
            builder = trt.Builder(logger); net = builder.create_network(0)
            parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(path):
                print("TensorRT 读 ONNX 失败：", [str(parser.get_error(i))[:300] for i in range(min(parser.num_errors, 3))]); break
            cfg = builder.create_builder_config()
            if prec == "fp16":
                cfg.set_flag(trt.BuilderFlag.FP16)
            t = time.time(); blob = builder.build_serialized_network(net, cfg)
            if blob is None:
                print(f"TensorRT {prec} 编译失败"); continue
            open(eng_path, "wb").write(blob)
            print(f"TensorRT {prec} 引擎编译成功，用时 {time.time() - t:.0f} 秒，{os.path.getsize(eng_path) / 2**20:.0f} MB", flush=True)
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        with TrtRunner(EngineFromBytes(BytesFromPath(eng_path))) as r:
            names = [k for k in d.files if k in r.get_input_metadata()]
            feed = {k: d[k] for k in names}
            stat(f"TensorRT {prec}", lambda: r.infer(feed)["actions"])
except Exception as e:
    print(f"TensorRT 失败 —— {type(e).__name__}: {str(e)[:400]}")
print("BENCH DONE")
