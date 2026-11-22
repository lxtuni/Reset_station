"""在拦截环境里评测微调后的 SmolVLA。协议和 act/eval_act.py 完全一样：
6 个速度档 × 30 回合（种子 0–29），指标是成功率、成功用时、撞墙次数、抖动。

和 ACT 评测的区别：
  - 输入多了一句任务描述；相机叫 camera1（顶部）、camera2（腕部）
  - 动作块是 50 步（ACT 是 20 步）；动作要迭代去噪 num_steps 次才生成
  - --delay auto：用“实测的推理耗时”当延迟（耗时不足一个 50 ms 周期算 0，和 ACT 的评测口径一致），
    而不是人为注入一个数。SmolVLA 推理慢，这才是它真实的工作状态。

用法（仓库根目录，PYTHONPATH=act）：
    python vla/eval_vla.py MODEL_DIR --bins --exec_steps 5
    python vla/eval_vla.py MODEL_DIR --bins --exec_steps 5 --delay auto --async_exec
    python vla/eval_vla.py MODEL_DIR --v_min 0.10 --v_max 0.20 --episodes 60 --num_steps 4
"""
import argparse, json, math, time
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints, BINS

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--bins", action="store_true")
ap.add_argument("--v_min", type=float, default=0.0)
ap.add_argument("--v_max", type=float, default=0.20)
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--exec_steps", type=int, default=5)       # 每次推理后执行几步再重新推理
ap.add_argument("--delay", default="0")                    # 毫秒数，或 auto = 用实测推理耗时
ap.add_argument("--async_exec", action="store_true")
ap.add_argument("--num_steps", type=int, default=None)     # 去噪迭代次数（默认用模型自带的 10）
ap.add_argument("--fp32", action="store_true")            # 把模型转成 32 位小数再跑（导出 ONNX 时必须这样做，这里验证成功率不受影响）
ap.add_argument("--backend", default="torch")              # torch / trt
ap.add_argument("--engine", default=None)                  # TensorRT 引擎文件（去噪次数在导出时已固定）
ap.add_argument("--commit", type=int, default=0)           # 自适应执行：预测的动作块开头明显上抬时，连续执行这么多步再重新推理（0 = 关）
ap.add_argument("--commit_z", type=float, default=0.4)     # “明显上抬”的门槛：动作块前 4 步的上下分量的平均值
ap.add_argument("--csv", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

STEP_MS = 50
TASK = "intercept the drifting boat and grasp its mast"
policy = SmolVLAPolicy.from_pretrained(args.model).to("cuda").eval()
if args.num_steps:
    policy.config.num_steps = args.num_steps
    if hasattr(policy.model, "config"):
        policy.model.config.num_steps = args.num_steps
if args.fp32:
    policy.float()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model)

if args.backend == "trt":
    # TensorRT 引擎里已经包含：图像缩放填充、视觉语言模型、固定次数的去噪。这里只负责把显卡上的张量地址交给它。
    import tensorrt as trt
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    _rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    _eng = _rt.deserialize_cuda_engine(open(args.engine, "rb").read())
    _ctx = _eng.create_execution_context()
    _DT = {trt.DataType.FLOAT: torch.float32, trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32,
           trt.DataType.BOOL: torch.bool}
    _names = [_eng.get_tensor_name(i) for i in range(_eng.num_io_tensors)]
    _out = torch.empty(1, policy.config.chunk_size, 5, device="cuda")
    print("TensorRT", trt.__version__, "引擎的输入输出：", {n: str(_eng.get_tensor_dtype(n)).split(".")[-1] for n in _names})

    def net(b):
        src = {"cam1": b["observation.images.camera1"], "cam2": b["observation.images.camera2"],
               "state": b["observation.state"], "tokens": b[OBS_LANGUAGE_TOKENS], "mask": b[OBS_LANGUAGE_ATTENTION_MASK],
               "noise": torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")}
        keep = []                                          # 推理期间这些张量不能被释放
        for n in _names:
            if n == "actions":
                _ctx.set_tensor_address(n, _out.data_ptr()); continue
            t = src[n].to("cuda", _DT[_eng.get_tensor_dtype(n)]).contiguous(); keep.append(t)
            _ctx.set_tensor_address(n, t.data_ptr())
        _ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return _out.clone()
else:
    def net(b):
        return policy.predict_action_chunk(b)


def to_img(x):                                             # (H,W,3) uint8 → (1,3,H,W) float [0,1]
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)


T = {"pre": [], "infer": [], "post": []}


def infer(top, wrist, q):
    t0 = time.perf_counter()
    b = pre({"observation.images.camera1": to_img(top), "observation.images.camera2": to_img(wrist),
             "observation.state": torch.from_numpy(q).unsqueeze(0), "task": [TASK]})
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.inference_mode():
        chunk = net(b)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    acts = post(chunk)[0].cpu().numpy()                     # (50, 5)
    t3 = time.perf_counter()
    T["pre"].append((t1 - t0) * 1000); T["infer"].append((t2 - t1) * 1000); T["post"].append((t3 - t2) * 1000)
    return acts, (t3 - t0) * 1000


env = make_env(); cams = Cams(env)
HOLD = np.zeros(5, np.float32)                              # 等推理时：动作增量为 0，机械臂停在原地
for _ in range(3):                                          # 预热（第一次推理包含初始化，很慢）
    env.reset(seed=0); infer(*cams(), joints(env))
lat_est = float(np.mean([a + b_ + c for a, b_, c in zip(T["pre"][1:], T["infer"][1:], T["post"][1:])]))
for v in T.values():
    v.clear()
K = []                                                      # 每次推理对应的延迟步数
CM = []                                                     # 每次推理是否触发了“连续执行”


def delay_steps():
    ms = lat_est if args.delay == "auto" else float(args.delay)
    return min(int(math.floor(ms / STEP_MS + 1e-6)) if args.delay == "auto" else round(ms / STEP_MS), args.exec_steps - 1)


def run(v_range, n, seed0):
    global lat_est
    env.v_range = v_range
    S, N, Hh, J = [], [], [], []
    for ep in range(n):
        env.reset(seed=seed0 + ep); policy.reset(); queue, started = [], False
        while True:
            k = delay_steps()
            early = args.async_exec and k > 0 and started and len(queue) == k
            if not queue or early:
                acts, ms = infer(*cams(), joints(env))
                lat_est = 0.9 * lat_est + 0.1 * ms          # 推理耗时的滑动平均
                go = bool(args.commit and float(np.mean(acts[:4, 2])) > args.commit_z)
                CM.append(go)
                acts = acts[:args.commit if go else args.exec_steps]; K.append(k)
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
for vr in (BINS if args.bins else [(args.v_min, args.v_max)]):
    name = f"{100 * vr[0]:.0f}-{100 * vr[1]:.0f} cm/s"
    res[name] = r_ = run(vr, args.episodes, args.seed0)
    print(f"{name}: 成功 {r_['succ']}/{r_['n']}，用时 {(r_['steps'] or float('nan')) * 0.05:.1f} s，"
          f"撞墙 {r_['walls']:.2f} 次/回合，抖动 {r_['jitter']:.3f}", flush=True)

tot_s = sum(r_["succ"] for r_ in res.values()); tot_n = sum(r_["n"] for r_ in res.values())
m = {key: float(np.mean(v)) for key, v in T.items()}
lat = np.array(T["infer"])
print(f"合计成功率 {tot_s}/{tot_n} = {tot_s / tot_n * 100:.1f}%  exec_steps={args.exec_steps} delay={args.delay} "
      f"async={int(args.async_exec)} num_steps={policy.config.num_steps} backend={args.backend} fp32={int(args.fp32)} "
      f"平均延迟步数={np.mean(K):.2f}")
if args.commit:
    print(f"自适应执行：门槛 {args.commit_z}，连续 {args.commit} 步；{len(CM)} 次推理里触发 {int(np.sum(CM))} 次（{np.mean(CM) * 100:.1f}%）")
print(f"时间分解（每次推理）：前处理 {m['pre']:.1f} ms | 网络推理 {m['infer']:.1f} ms (p95 {np.percentile(lat, 95):.1f}) | "
      f"后处理 {m['post']:.1f} ms | 合计 {sum(m.values()):.1f} ms，控制周期 {STEP_MS} ms")
if args.csv:
    walls = float(np.mean([r_["walls"] for r_ in res.values()]))
    with open(args.csv, "a") as fh:
        fh.write(f"smolvla,{args.delay},{args.exec_steps},{int(args.async_exec)},{policy.config.num_steps},"
                 f"{tot_s / tot_n * 100:.1f},{walls:.3f},{m['infer']:.2f},{tot_n}\n")
if args.json:
    json.dump(dict(args=vars(args), bins=res, timing=m, mean_delay_steps=float(np.mean(K))), open(args.json, "w"),
              ensure_ascii=False, indent=1)
