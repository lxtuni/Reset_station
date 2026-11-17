"""把复位台 ACT 的网络部分导出成 ONNX（三个输入：top、wrist、state），并和 PyTorch 逐数值核对。

用法（仓库根目录）：python act/export_onnx.py MODEL_DIR OUT_DIR
生成 OUT_DIR/act_rs.onnx(+.data)、sample.npz（一帧真实输入）、ref_actions.npy（PyTorch 输出）
"""
import os, sys
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy, OBS_IMAGES, OBS_STATE
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

M, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
policy = ACTPolicy.from_pretrained(M).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
IMG_KEYS = list(policy.config.image_features)            # 训练时的相机顺序，导出必须保持一致
NAMES = [k.split(".")[-1] for k in IMG_KEYS]             # ["top", "wrist"]


class ACTCore(torch.nn.Module):                          # 只包含网络，不含归一化
    def __init__(self, p):
        super().__init__(); self.model = p.model
    def forward(self, cam0, cam1, state):                # 图像 (1,3,240,320)，state (1,6)，都已归一化
        return self.model({OBS_STATE: state, OBS_IMAGES: [cam0, cam1]})[0]   # (1,20,5)


core = ACTCore(policy).eval()

# 从仿真里取一帧真实输入（船在动的一局，先空走 10 步让画面不是初始帧）
env = make_env((0.06, 0.10)); cams = Cams(env)
env.reset(seed=0)
for _ in range(10):
    env.step(np.zeros(5, np.float32))
top, wrist = cams()
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
b = pre({"observation.images.top": f(top), "observation.images.wrist": f(wrist),
         "observation.state": torch.from_numpy(joints(env)).unsqueeze(0), "task": [""]})
imgs = [b[k].float().clone() for k in IMG_KEYS]; st = b[OBS_STATE].float().clone()
np.savez(f"{OUT}/sample.npz", **{n: x.cpu().numpy() for n, x in zip(NAMES, imgs)}, state=st.cpu().numpy())

with torch.no_grad():
    ref = core(*imgs, st).cpu().numpy()
np.save(f"{OUT}/ref_actions.npy", ref)
print("相机顺序:", NAMES, "| PyTorch 输出形状:", ref.shape)

with torch.no_grad():                                    # opset 18：见 act_deploy/export_onnx.py 里的说明
    torch.onnx.export(core, (*imgs, st), f"{OUT}/act_rs.onnx", input_names=NAMES + ["state"],
                      output_names=["actions"], opset_version=18)

import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
sess = ort.InferenceSession(f"{OUT}/act_rs.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
d = np.load(f"{OUT}/sample.npz")
out = sess.run(None, {k: d[k] for k in d.files})[0]
print("ONNX Runtime providers:", sess.get_providers())
print(f"ONNX Runtime 和 PyTorch 的最大误差: {np.abs(out - ref).max():.2e}")
