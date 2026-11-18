"""尝试把微调后的 SmolVLA 整个推理过程导出成 ONNX，并用 ONNX Runtime 核对和测速。

和 ACT 的导出相比，难点有三个：
  1. 多了语言输入（整数的词编号 + 掩码）
  2. 图像的缩放和填充（240×320 → 512×512）也在网络里
  3. 去噪要迭代。这里把去噪次数固定（默认 3 次），循环会被“展开”成 3 段相同的计算，写死在图里
随机噪声改成网络的一个输入，这样 PyTorch 和 ONNX 可以喂同一份噪声来比较输出。
成功与否都如实打印。用法（仓库根目录，PYTHONPATH=act）：python vla/export_vla.py MODEL_DIR OUT_DIR [去噪次数]
"""
import os, sys, time, traceback
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from common import make_env, Cams, joints

M, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(OUT, exist_ok=True)
policy = SmolVLAPolicy.from_pretrained(M).to("cuda").eval()
policy.config.num_steps = N
if hasattr(policy.model, "config"):
    policy.model.config.num_steps = N
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
K1, K2 = "observation.images.camera1", "observation.images.camera2"

env = make_env((0.06, 0.10)); cams = Cams(env); env.reset(seed=0)
top, wrist = cams(); q = joints(env)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
b = pre({K1: f(top), K2: f(wrist), OBS_STATE: torch.from_numpy(q).unsqueeze(0),
         "task": ["intercept the drifting boat and grasp its mast"]})
print("前处理后的输入：", {k: (tuple(v.shape), str(v.dtype)) for k, v in b.items() if torch.is_tensor(v)})


class Core(torch.nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p
    def forward(self, cam1, cam2, state, tokens, mask, noise):
        batch = {K1: cam1, K2: cam2, OBS_STATE: state}
        images, img_masks = self.p.prepare_images(batch)          # 缩放、填充到 512×512，数值变到 [-1, 1]
        st = self.p.prepare_state(batch)                          # 关节角度补零到 32 维
        a = self.p.model.sample_actions(images, img_masks, tokens, mask, st, noise=noise)
        return a[:, :, :5]                                        # 去掉补零的动作维度 → (1, 50, 5)


core = Core(policy).eval()
torch.manual_seed(0)
noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")
inputs = (b[K1].float(), b[K2].float(), b[OBS_STATE].float(), b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], noise)
names = ["cam1", "cam2", "state", "tokens", "mask", "noise"]
with torch.no_grad():
    ref_bf16 = core(*inputs).cpu().numpy()
# 视觉语言模型的权重是以 bfloat16（一种 16 位小数）加载的。ONNX Runtime 不接受这种类型，
# 所以导出前把整个模型转成普通的 32 位小数，并看看这样做输出变了多少。
policy.float()
with torch.no_grad():
    ref = core(*inputs).cpu().numpy()
    ref2 = core(*inputs).cpu().numpy()
print("PyTorch 输出形状", ref.shape, "| 同一份噪声两次运行的差别", float(np.abs(ref - ref2).max()),
      f"| 转成 32 位后和原模型（bfloat16）的最大差别 {np.abs(ref - ref_bf16).max():.2e}")
np.savez(f"{OUT}/sample.npz", **{n: x.cpu().numpy() for n, x in zip(names, inputs)}); np.save(f"{OUT}/ref.npy", ref)

path = f"{OUT}/smolvla_n{N}.onnx"; ok = False
for label, kw in [("新导出器（dynamo）", dict(dynamo=True)), ("旧导出器（逐行跟踪）", dict(dynamo=False))]:
    try:
        t = time.time()
        with torch.no_grad():
            torch.onnx.export(core, inputs, path, input_names=names, output_names=["actions"], opset_version=18, **kw)
        sz = sum(os.path.getsize(f"{OUT}/{x}") for x in os.listdir(OUT) if x.startswith(f"smolvla_n{N}.onnx")) / 2**20
        print(f"{label}：导出成功，用时 {time.time() - t:.0f} 秒，文件共 {sz:.0f} MB"); ok = True; break
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f"{label}：导出失败 —— {type(e).__name__}: {' | '.join(msg[:3])[:500]}")
        tb = traceback.extract_tb(e.__traceback__)
        print("   出错位置：", " <- ".join(f"{os.path.basename(x.filename)}:{x.lineno}" for x in tb[-4:]))
if not ok:
    print("EXPORT FAILED"); sys.exit(0)

import onnx, onnxruntime as ort
g = onnx.load(path, load_external_data=False).graph
ops = {}
for n in g.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"图里有 {len(g.node)} 个节点；最多的几种：", sorted(ops.items(), key=lambda x: -x[1])[:10])
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
try:
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("ONNX Runtime providers:", sess.get_providers())
    want = {i.name for i in sess.get_inputs()}
    feed = {n: x.cpu().numpy() for n, x in zip(names, inputs) if n in want}
    print("引擎实际需要的输入：", sorted(want))
    out = sess.run(None, feed)[0]
    print(f"ONNX Runtime 和 PyTorch 的最大误差：{np.abs(out - ref).max():.2e}（动作的数值范围约 ±{np.abs(ref).max():.1f}）")
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        print("这个环境里 ONNX Runtime 用不了显卡（它要的 CUDA 12 库和 PyTorch 自带的 CUDA 13 不一致），"
              "这里只做数值核对，显卡上的测速放到单独的环境里做（ort_trt_bench.py）")
        print("EXPORT SCRIPT DONE"); sys.exit(0)
    for _ in range(10):
        sess.run(None, feed)
    ms = []
    for _ in range(50):
        t = time.perf_counter(); sess.run(None, feed); ms.append((time.perf_counter() - t) * 1e3)
    with torch.inference_mode():
        for _ in range(10):
            core(*inputs)
        torch.cuda.synchronize(); mt = []
        for _ in range(50):
            t = time.perf_counter(); core(*inputs); torch.cuda.synchronize(); mt.append((time.perf_counter() - t) * 1e3)
    print(f"去噪 {N} 次：PyTorch {np.mean(mt):.1f} ms (p95 {np.percentile(mt, 95):.1f}) | "
          f"ONNX Runtime {np.mean(ms):.1f} ms (p95 {np.percentile(ms, 95):.1f})   注：显卡此时可能被其他任务占用")
except Exception as e:
    print(f"ONNX Runtime 运行失败 —— {type(e).__name__}: {str(e)[:500]}")
print("EXPORT SCRIPT DONE")
