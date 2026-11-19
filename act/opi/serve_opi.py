"""香橙派上的推理服务：收到两张相机原图和关节角度，返回 20 步动作。整条链路都在板子上完成：
  原图（uint8）→ 归一化 → 骨干（NPU）→ Transformer（NPU 或处理器）→ 还原成真实动作
电脑上的仿真器每一步把画面发过来，拿到动作再走一步，这样就能在不搬动仿真器的情况下做闭环验证。
只监听本机地址（127.0.0.1），电脑通过 SSH 隧道连进来，不对外开放端口。
用法：python serve_opi.py <模型目录> <librknnrt.so> <方案：npu16opt | npu16 | npu8 | cpu> [端口=5555]
  npu16opt  npu16 的优化版：归一化交给 NPU，两路骨干在 NPU 的两个核上并行
  npu16  骨干和 Transformer 都在 NPU 上，16 位小数
  npu8   骨干在 NPU 上用 INT8，Transformer 在 NPU 上用 16 位
  cpu    原始的 ONNX 模型，处理器 4 个大核
"""
import os, socket, struct, sys, time
import numpy as np

D, LIB, MODE = sys.argv[1], sys.argv[2], sys.argv[3]; PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 5555
os.sched_setaffinity(0, [4, 5, 6, 7])
H, W = 240, 320
n = np.fromfile(f"{D}/norm.f32", np.float32)                # 归一化常数，顺序见 dump_raw_sample.py
m_top, s_top, m_wr, s_wr, m_q, s_q, m_a, s_a = n[0:3], n[3:6], n[6:9], n[9:12], n[12:18], n[18:24], n[24:29], n[29:34]

RAW_IN = MODE == "npu16opt"                                 # 这个方案直接喂原图，不在处理器上做归一化
if MODE == "npu16opt":
    # 优化后的方案：归一化写在 NPU 模型里；两路相机的骨干各占 NPU 的一个核，两个线程同时发起；Transformer 在 NPU 上
    import threading
    from rknn_c import NPU, NCHW
    bbA = NPU(LIB, f"{D}/backbone_u8_fp16.rknn", core_mask=1); bbB = NPU(LIB, f"{D}/backbone_u8_fp16.rknn", core_mask=2)
    hd = NPU(LIB, f"{D}/head_fp16.rknn")
    _o = [None, None]

    def net(a, b, q):
        def w(i, m, im):
            _o[i] = m.run(im, u8=True).reshape(1, 512, 8, 10)
        t1 = threading.Thread(target=w, args=(0, bbA, a)); t2 = threading.Thread(target=w, args=(1, bbB, b))
        t1.start(); t2.start(); t1.join(); t2.join()
        return hd.run_multi([_o[0], _o[1], q], [NCHW, NCHW, 3])
elif MODE == "cpu":
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(f"{D}/act_rs.onnx", so, providers=["CPUExecutionProvider"])
    net = lambda a, b, q: sess.run(None, {"top": a, "wrist": b, "state": q})[0]
else:
    from rknn_c import NPU, NCHW
    bb = NPU(LIB, f"{D}/backbone_{'fp16' if MODE == 'npu16' else 'int8'}.rknn"); hd = NPU(LIB, f"{D}/head_fp16.rknn")
    net = lambda a, b, q: hd.run_multi([bb.run(a, NCHW).reshape(1, 512, 8, 10), bb.run(b, NCHW).reshape(1, 512, 8, 10), q], [NCHW, NCHW, 3])


def prep(buf, m, s):                                        # (240,320,3) uint8 → (1,3,240,320) 归一化后的小数
    x = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) * (1 / 255.0)
    return np.ascontiguousarray(((x - m) / s).transpose(2, 0, 1)[None])


def recvall(c, k):
    b = bytearray()
    while len(b) < k:
        p = c.recv(k - len(b))
        if not p:
            return None
        b += p
    return bytes(b)


NB = H * W * 3
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen(1)
print(f"推理服务就绪：方案 {MODE}，端口 {PORT}", flush=True)
while True:
    c, _ = srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ms = []
    while True:
        buf = recvall(c, 2 * NB + 24)
        if buf is None:
            break
        t = time.perf_counter()
        if RAW_IN:
            a = np.frombuffer(buf[:NB], np.uint8).reshape(1, H, W, 3); b = np.frombuffer(buf[NB:2 * NB], np.uint8).reshape(1, H, W, 3)
        else:
            a = prep(buf[:NB], m_top, s_top); b = prep(buf[NB:2 * NB], m_wr, s_wr)
        q = ((np.frombuffer(buf[2 * NB:], np.float32) - m_q) / s_q)[None].astype(np.float32)
        act = np.asarray(net(a, b, q), np.float32).reshape(20, 5) * s_a + m_a
        dt = (time.perf_counter() - t) * 1e3; ms.append(dt)
        c.sendall(act.astype(np.float32).tobytes() + struct.pack("<f", dt))
    c.close()
    if ms:
        ms = np.array(ms[1:] or ms)
        print(f"一个连接结束：{len(ms)} 次推理，板上耗时 mean={ms.mean():.1f} ms p50={np.percentile(ms, 50):.1f} p95={np.percentile(ms, 95):.1f} max={ms.max():.1f}", flush=True)
