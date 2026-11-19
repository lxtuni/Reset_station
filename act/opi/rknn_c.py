"""直接调用瑞芯微 NPU 运行库（librknnrt.so）的 C 接口。

瑞芯微提供的 Python 包（rknn-toolkit-lite2）把运行库的路径写死成 /usr/lib/librknnrt.so，而系统里那个是 1.4.0 的旧版，
读不了新工具转换的模型。这里用 ctypes 自己加载放在用户目录里的新版，不改动系统文件。
C 接口只用到 6 个函数：初始化、设置用哪几个 NPU 核、设置输入、运行、取输出、释放。
"""
import ctypes as C
import numpy as np


class RknnInput(C.Structure):
    _fields_ = [("index", C.c_uint32), ("buf", C.c_void_p), ("size", C.c_uint32), ("pass_through", C.c_uint8),
                ("type", C.c_int), ("fmt", C.c_int)]


class RknnOutput(C.Structure):
    _fields_ = [("want_float", C.c_uint8), ("is_prealloc", C.c_uint8), ("index", C.c_uint32), ("buf", C.c_void_p), ("size", C.c_uint32)]


FLOAT32, UINT8, NCHW, NHWC = 0, 3, 0, 1                     # rknn_tensor_type / rknn_tensor_format 里的编号


class NPU:
    def __init__(self, lib_path, model_path, core_mask=7):  # 7 = 同时用 NPU 的 3 个核
        self.lib = C.CDLL(lib_path)
        self.lib.rknn_init.argtypes = [C.POINTER(C.c_uint64), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
        self.lib.rknn_set_core_mask.argtypes = [C.c_uint64, C.c_int]
        self.lib.rknn_inputs_set.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnInput)]
        self.lib.rknn_run.argtypes = [C.c_uint64, C.c_void_p]
        self.lib.rknn_outputs_get.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnOutput), C.c_void_p]
        self.lib.rknn_outputs_release.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnOutput)]
        self.lib.rknn_destroy.argtypes = [C.c_uint64]
        self.model = open(model_path, "rb").read()
        self.ctx = C.c_uint64(0)
        ret = self.lib.rknn_init(C.byref(self.ctx), self.model, len(self.model), 0, None)
        if ret != 0:
            raise RuntimeError(f"rknn_init 失败，错误码 {ret}")
        ret = self.lib.rknn_set_core_mask(self.ctx, core_mask)
        if ret != 0:
            print(f"（设置 NPU 核心失败，错误码 {ret}，用默认设置继续）")

    def run(self, x, fmt=NCHW, u8=False):
        # u8=True：直接喂相机原图（0–255 的整数，通道在后）。归一化已经写在模型里，由 NPU 自己做
        x = np.ascontiguousarray(x, np.uint8 if u8 else np.float32)
        inp = RknnInput(0, x.ctypes.data, x.nbytes, 0, UINT8 if u8 else FLOAT32, NHWC if u8 else fmt)
        if self.lib.rknn_inputs_set(self.ctx, 1, C.byref(inp)) != 0:
            raise RuntimeError("rknn_inputs_set 失败")
        if self.lib.rknn_run(self.ctx, None) != 0:
            raise RuntimeError("rknn_run 失败")
        out = RknnOutput(1, 0, 0, None, 0)                  # want_float=1：不管内部是 8 位还是 16 位，输出都还原成 32 位小数
        if self.lib.rknn_outputs_get(self.ctx, 1, C.byref(out), None) != 0:
            raise RuntimeError("rknn_outputs_get 失败")
        y = np.frombuffer(C.string_at(out.buf, out.size), np.float32).copy()
        self.lib.rknn_outputs_release(self.ctx, 1, C.byref(out))
        return y

    def run_multi(self, xs, fmts):                          # 多个输入的模型（Transformer 那一段有 3 个输入）
        xs = [np.ascontiguousarray(x, np.float32) for x in xs]
        arr = (RknnInput * len(xs))(*[RknnInput(i, x.ctypes.data, x.nbytes, 0, FLOAT32, f) for i, (x, f) in enumerate(zip(xs, fmts))])
        if self.lib.rknn_inputs_set(self.ctx, len(xs), arr) != 0:
            raise RuntimeError("rknn_inputs_set 失败")
        if self.lib.rknn_run(self.ctx, None) != 0:
            raise RuntimeError("rknn_run 失败")
        out = RknnOutput(1, 0, 0, None, 0)
        if self.lib.rknn_outputs_get(self.ctx, 1, C.byref(out), None) != 0:
            raise RuntimeError("rknn_outputs_get 失败")
        y = np.frombuffer(C.string_at(out.buf, out.size), np.float32).copy()
        self.lib.rknn_outputs_release(self.ctx, 1, C.byref(out))
        return y

    def close(self):
        self.lib.rknn_destroy(self.ctx)
