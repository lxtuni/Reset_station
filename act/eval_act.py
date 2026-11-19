"""在拦截环境里评测 ACT（只看两路图像 + 关节角度）。协议和 eval/intercept_ab.py 一样：
6 个速度档 × 30 回合（种子 0–29），指标是成功率、成功用时、撞墙次数、抖动。

用法（仓库根目录）：
    python act/eval_act.py MODEL_DIR --bins                         # 6 个速度档
    python act/eval_act.py MODEL_DIR --v_min 0.10 --v_max 0.20 --episodes 60 --delay_ms 100 [--async_exec]
    python act/eval_act.py MODEL_DIR --bins --backend onnx --engine act_intercept.onnx
MODEL_DIR 例如 outputs/act_full/checkpoints/last/pretrained_model
"""
import argparse, json, time
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints, BINS

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--bins", action="store_true")
ap.add_argument("--v_min", type=float, default=0.0)
ap.add_argument("--v_max", type=float, default=0.20)
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--exec_steps", type=int, default=5)      # 每次推理后执行几步再重新推理
ap.add_argument("--delay_ms", type=float, default=0)       # 注入的推理延迟（控制周期 50 ms）
ap.add_argument("--async_exec", action="store_true")
ap.add_argument("--backend", default="torch")              # torch / onnx
ap.add_argument("--engine", default=None)
ap.add_argument("--csv", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

STEP_MS = 50
k = round(args.delay_ms / STEP_MS)
assert args.exec_steps > k or k == 0, "exec_steps 必须大于延迟对应的步数"

policy = ACTPolicy.from_pretrained(args.model).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model)
IMG_KEYS = list(policy.config.image_features)              # 训练时的相机顺序


def to_img(x):                                             # (H,W,3) uint8 → (1,3,H,W) float [0,1]
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)


if args.backend == "onnx":
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(args.engine, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("ONNX Runtime providers:", sess.get_providers())

    def net(b):
        feed = {key.split(".")[-1]: b[key].float().cpu().numpy() for key in IMG_KEYS}
        feed["state"] = b["observation.state"].float().cpu().numpy()
        return torch.from_numpy(sess.run(None, feed)[0]).cuda()
elif args.backend == "trt":
    # TensorRT 引擎（FP32 / FP16 / INT8 都走这里）。输入输出直接用显卡上 PyTorch 张量的地址，不经过 CPU。
    import tensorrt as trt
    _rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    _eng = _rt.deserialize_cuda_engine(open(args.engine, "rb").read())
    _ctx = _eng.create_execution_context()
    _out = torch.empty(1, policy.config.chunk_size, 5, device="cuda")
    print("TensorRT", trt.__version__, "引擎：", args.engine)

    def net(b):
        keep = []                                          # 推理期间这些张量不能被释放
        for key in IMG_KEYS + ["observation.state"]:
            t = b[key].to("cuda", torch.float32).contiguous(); keep.append(t)
            _ctx.set_tensor_address(key.split(".")[-1], t.data_ptr())
        _ctx.set_tensor_address("actions", _out.data_ptr())
        _ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return _out.clone()
else:
    def net(b):
        return policy.predict_action_chunk(b)

T = {"pre": [], "infer": [], "post": []}


if args.backend == "remote":
    # 推理在另一台设备上（香橙派）：把两张原图和关节角度发过去，收回 20 步真实动作。归一化和还原都在那边做。
    # --engine 写 “地址:端口”，例如 127.0.0.1:5555（通过 SSH 隧道）
    import socket, struct
    _h, _p = args.engine.split(":")
    _sock = socket.create_connection((_h, int(_p))); _sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _recv(k):
        b = bytearray()
        while len(b) < k:
            p = _sock.recv(k - len(b))
            if not p:
                raise RuntimeError("推理服务断开了连接")
            b += p
        return bytes(b)


def infer(top, wrist, q):
    if args.backend == "remote":
        t0 = time.perf_counter()
        _sock.sendall(np.ascontiguousarray(top).tobytes() + np.ascontiguousarray(wrist).tobytes() + q.astype(np.float32).tobytes())
        buf = _recv(404)
        dev_ms = struct.unpack("<f", buf[400:])[0]            # 板子上从收到原图到算出动作的时间
        T["pre"].append((time.perf_counter() - t0) * 1000 - dev_ms)   # 这一项在远程模式下是“网络传输”的时间
        T["infer"].append(dev_ms); T["post"].append(0.0)
        return np.frombuffer(buf[:400], np.float32).reshape(20, 5).copy()
    t0 = time.perf_counter()
    # 相机的名字取自模型自己的配置（第一个是顶部相机，第二个是腕部相机），这样换了数据集里的相机名也能用
    b = pre({IMG_KEYS[0]: to_img(top), IMG_KEYS[1]: to_img(wrist),
             "observation.state": torch.from_numpy(q).unsqueeze(0), "task": [""]})
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.inference_mode():
        chunk = net(b)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    acts = post(chunk)[0].cpu().numpy()                     # (20, 5)
    t3 = time.perf_counter()
    T["pre"].append((t1 - t0) * 1000); T["infer"].append((t2 - t1) * 1000); T["post"].append((t3 - t2) * 1000)
    return acts


env = make_env(); cams = Cams(env)
HOLD = np.zeros(5, np.float32)                              # 等推理时：动作增量为 0，机械臂停在原地


def run(v_range, n, seed0):
    env.v_range = v_range
    S, N, Hh, J = [], [], [], []
    for ep in range(n):
        env.reset(seed=seed0 + ep); queue, started = [], False
        while True:
            early = args.async_exec and k > 0 and started and len(queue) == k
            if not queue or early:
                acts = infer(*cams(), joints(env))[:args.exec_steps]
                if early:
                    queue = queue + list(acts[k:])          # 旧块剩下的 k 步先执行，新块跳过已过时的前 k 步
                else:
                    queue = ([HOLD] * k if started else []) + list(acts)
                started = True
            obs, r, term, trunc, info = env.step(np.array(queue.pop(0), dtype=np.float32))
            J.append(info["da2"])
            if term or trunc:
                break
        S.append(info["success"]); N.append(env.steps); Hh.append(info["wall_hits"])
    S = np.array(S)
    return dict(succ=int(S.sum()), n=n, steps=float(np.median(np.array(N)[S])) if S.any() else None,
                walls=float(np.mean(Hh)), jitter=float(np.mean(J)))


res = {}
ranges = BINS if args.bins else [(args.v_min, args.v_max)]
for vr in ranges:
    name = f"{100 * vr[0]:.0f}-{100 * vr[1]:.0f} cm/s"
    res[name] = run(vr, args.episodes, args.seed0)
    r_ = res[name]
    print(f"{name}: 成功 {r_['succ']}/{r_['n']}，用时 {(r_['steps'] or float('nan')) * 0.05:.1f} s，"
          f"撞墙 {r_['walls']:.2f} 次/回合，抖动 {r_['jitter']:.3f}", flush=True)

tot_s = sum(r_["succ"] for r_ in res.values()); tot_n = sum(r_["n"] for r_ in res.values())
m = {key: float(np.mean(v[1:] or v)) for key, v in T.items()}
lat = np.array(T["infer"][1:] or T["infer"])
print(f"合计成功率 {tot_s}/{tot_n} = {tot_s / tot_n * 100:.1f}%  backend={args.backend} exec_steps={args.exec_steps} "
      f"delay={args.delay_ms:.0f}ms async={int(args.async_exec)}")
print(f"时间分解（每次推理）：前处理 {m['pre']:.1f} ms | 网络推理 {m['infer']:.1f} ms (p95 {np.percentile(lat, 95):.1f}) | "
      f"后处理 {m['post']:.1f} ms | 合计 {sum(m.values()):.1f} ms，控制周期 {STEP_MS} ms")
if args.csv:
    walls = float(np.mean([r_["walls"] for r_ in res.values()]))
    with open(args.csv, "a") as fh:
        fh.write(f"{args.backend},{args.delay_ms:.0f},{args.exec_steps},{int(args.async_exec)},"
                 f"{tot_s / tot_n * 100:.1f},{walls:.3f},{m['infer']:.2f},{tot_n}\n")
if args.json:
    json.dump(dict(args=vars(args), bins=res, timing=m), open(args.json, "w"), ensure_ascii=False, indent=1)
