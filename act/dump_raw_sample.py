"""给 C++ 程序准备一帧“原始”输入和 Python 的参考答案，全部存成不带格式的二进制文件（C++ 用 fread 就能读）。

用法（仓库根目录）：python act/dump_raw_sample.py MODEL_DIR OUT_DIR
存下的文件：
  top.u8 / wrist.u8      相机原图，240×320×3，uint8，按 高×宽×通道 排列（相机给出的就是这种排列）
  state.f32              6 个关节角度（未归一化）
  norm.f32               归一化用的常数，共 34 个数，顺序是：
                         top 均值 3、top 标准差 3、wrist 均值 3、wrist 标准差 3、关节均值 6、关节标准差 6、动作均值 5、动作标准差 5
  ref_top.f32 / ref_wrist.f32   Python 前处理的结果（1×3×240×320），用来核对 CUDA kernel
  ref_actions.f32        Python 整条链路的最终输出（20×5，已还原成真实动作），用来核对 C++ 整条链路
归一化常数不是从文件里读的，而是“量”出来的：前处理对每个通道都是 y = (x - 均值) / 标准差，
喂全 0 和全 1 两个输入就能解出均值和标准差。这样不依赖 LeRobot 内部怎么存这些数。
"""
import os, sys
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

M, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
policy = ACTPolicy.from_pretrained(M).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
K_TOP, K_WR, K_ST = "observation.images.top", "observation.images.wrist", "observation.state"


def run_pre(top01, wrist01, q):          # 图像已是 0–1 的 (1,3,H,W)
    return pre({K_TOP: top01, K_WR: wrist01, K_ST: q, "task": [""]})


# 量出归一化常数
z, o = torch.zeros(1, 3, 240, 320), torch.ones(1, 3, 240, 320)
b0, b1 = run_pre(z, z, torch.zeros(1, 6)), run_pre(o, o, torch.ones(1, 6))
def solve(y0, y1):                        # y0 = -m/s, y1 = (1-m)/s
    s = 1.0 / (y1 - y0); return (-y0 * s).cpu().numpy().astype(np.float32), s.cpu().numpy().astype(np.float32)
mt, st_ = solve(b0[K_TOP][0, :, 0, 0], b1[K_TOP][0, :, 0, 0])
mw, sw = solve(b0[K_WR][0, :, 0, 0], b1[K_WR][0, :, 0, 0])
mq, sq = solve(b0[K_ST][0], b1[K_ST][0])
# 后处理是 a = y * 标准差 + 均值
a0 = post(torch.zeros(1, 20, 5, device="cuda"))[0, 0].cpu().numpy().astype(np.float32)
a1 = post(torch.ones(1, 20, 5, device="cuda"))[0, 0].cpu().numpy().astype(np.float32)
ma, sa = a0, a1 - a0
np.concatenate([mt, st_, mw, sw, mq, sq, ma, sa]).astype(np.float32).tofile(f"{OUT}/norm.f32")
print("top 均值/标准差:", mt, st_, "\nwrist 均值/标准差:", mw, sw, "\n关节 均值/标准差:", mq, sq, "\n动作 均值/标准差:", ma, sa)

# 一帧真实输入
env = make_env((0.06, 0.10)); cams = Cams(env); env.reset(seed=0)
for _ in range(10):
    env.step(np.zeros(5, np.float32))
top, wrist = cams(); q = joints(env)
np.ascontiguousarray(top).tofile(f"{OUT}/top.u8"); np.ascontiguousarray(wrist).tofile(f"{OUT}/wrist.u8")
q.astype(np.float32).tofile(f"{OUT}/state.f32")

f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
b = run_pre(f(top), f(wrist), torch.from_numpy(q).unsqueeze(0))
b[K_TOP].float().cpu().numpy().tofile(f"{OUT}/ref_top.f32"); b[K_WR].float().cpu().numpy().tofile(f"{OUT}/ref_wrist.f32")
with torch.inference_mode():
    acts = post(policy.predict_action_chunk(b))[0].cpu().numpy().astype(np.float32)
acts.tofile(f"{OUT}/ref_actions.f32")
print("参考动作第 1 步:", acts[0])
