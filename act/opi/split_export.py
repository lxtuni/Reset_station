"""把 ACT 拆成两段导出，为“骨干放 NPU、其余放处理器”做准备。

  backbone.onnx   图像骨干 ResNet-18：一张归一化后的图 (1,3,240,320) → 特征图 (1,512,8,10)。两路相机各跑一次
  head.onnx       其余全部：两张特征图 + 关节角度 → 20 步动作（1×1 卷积、位置编码、Transformer 编码器和解码器）
拆的位置就是 ACT 代码里 self.backbone(img)["feature_map"] 这一行的前后。
做法：导出 head 时，把模型里的骨干临时换成一个“什么都不做、直接把输入当特征返回”的模块，这样不用重写 ACT 的前向代码。
同时存下参考数据，供板子上核对：两路特征图、最终动作。
用法（仓库根目录，PYTHONPATH=act）：python act/split_export.py MODEL_DIR DEPLOY_DIR（里面要已有 sample.npz、ref_actions.npy）
"""
import sys
import numpy as np, torch
from lerobot.policies.act.modeling_act import ACTPolicy, OBS_IMAGES, OBS_STATE

M, D = sys.argv[1], sys.argv[2]
policy = ACTPolicy.from_pretrained(M).to("cpu").eval()     # 在处理器上导出即可（模型默认会被放到显卡上）
model = policy.model
d = np.load(f"{D}/sample.npz"); ref = np.load(f"{D}/ref_actions.npy")
names = [k for k in d.files if k != "state"]                # ["top", "wrist"]
imgs = [torch.from_numpy(d[k]) for k in names]; st = torch.from_numpy(d["state"])


class Backbone(torch.nn.Module):
    def __init__(self, b):
        super().__init__(); self.b = b
    def forward(self, image):
        return self.b(image)["feature_map"]


class PassThrough(torch.nn.Module):                         # 顶替骨干：输入已经是特征图
    def forward(self, x):
        return {"feature_map": x}


bb = Backbone(model.backbone).eval()
with torch.no_grad():
    feats = [bb(x) for x in imgs]
print("特征图形状:", tuple(feats[0].shape))
np.savez(f"{D}/split_ref.npz", **{f"feat_{n}": f.numpy() for n, f in zip(names, feats)}, state=d["state"], actions=ref,
         **{n: d[n] for n in names})

with torch.no_grad():
    try:                                                     # 骨干是标准的卷积网络，用旧导出器、较低的算子版本，转换工具更容易接受
        torch.onnx.export(bb, (imgs[0],), f"{D}/backbone.onnx", input_names=["image"], output_names=["feat"], opset_version=13, dynamo=False)
        print("backbone.onnx：旧导出器，算子版本 13")
    except Exception as e:
        torch.onnx.export(bb, (imgs[0],), f"{D}/backbone.onnx", input_names=["image"], output_names=["feat"], opset_version=18)
        print("backbone.onnx：新导出器，算子版本 18（旧导出器失败：", type(e).__name__, "）")


class Head(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, feat0, feat1, state):
        return self.m({OBS_STATE: state, OBS_IMAGES: [feat0, feat1]})[0]


real = model.backbone; model.backbone = PassThrough()
head = Head(model).eval()
with torch.no_grad():
    out = head(*feats, st).numpy()
    print(f"拆开后在 PyTorch 里重新接上，和原模型的最大误差 {np.abs(out - ref).max():.2e}")
    torch.onnx.export(head, (*feats, st), f"{D}/head.onnx", input_names=[f"feat_{n}" for n in names] + ["state"],
                      output_names=["actions"], opset_version=18)
model.backbone = real

import onnxruntime as ort
sb = ort.InferenceSession(f"{D}/backbone.onnx", providers=["CPUExecutionProvider"])
sh = ort.InferenceSession(f"{D}/head.onnx", providers=["CPUExecutionProvider"])
f2 = [sb.run(None, {"image": d[n]})[0] for n in names]
o2 = sh.run(None, {**{f"feat_{n}": f for n, f in zip(names, f2)}, "state": d["state"]})[0]
print(f"两个 ONNX 接起来（ONNX Runtime），和 PyTorch 的最大误差 {np.abs(o2 - ref).max():.2e}；特征图的数值范围 {f2[0].min():.2f} 到 {f2[0].max():.2f}")
print("SPLIT DONE")
