"""给量化准备“校准数据”：从教师示教里均匀抽 N 帧，过一遍前处理，存成网络输入的样子。

INT8 量化要把小数映射到 256 个整数上，必须先知道每一层数值的大致范围；这个范围就是拿真实输入跑一遍统计出来的。
用法（仓库根目录，PYTHONPATH=act）：python vla/calib_dump.py act|vla MODEL_DIR OUT.npz [N=200]
存下的每个数组第一维是样本编号，例如 top 是 (N, 1, 3, 240, 320)。
"""
import glob, sys
import numpy as np, torch
from lerobot.policies.factory import make_pre_post_processors

KIND, M, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
if KIND == "act":
    from lerobot.policies.act.modeling_act import ACTPolicy as P
    K1, K2, names = "observation.images.top", "observation.images.wrist", ("top", "wrist")
else:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    K1, K2, names = "observation.images.camera1", "observation.images.camera2", ("cam1", "cam2")
policy = P.from_pretrained(M).to("cuda").eval()
if KIND == "act":                                           # 相机名取自模型配置，和导出 ONNX 时的输入名保持一致
    K1, K2 = list(policy.config.image_features)[:2]; names = (K1.split(".")[-1], K2.split(".")[-1])
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)

files = sorted(glob.glob("data/raw_full_*/ep_*.npz")) + sorted(glob.glob("data/raw_static/ep_*.npz"))
rng = np.random.default_rng(0); out = {}
for i in range(N):
    d = np.load(files[rng.integers(len(files))]); t = rng.integers(len(d["action"]))
    b = pre({K1: f(d["top"][t]), K2: f(d["wrist"][t]), "observation.state": torch.from_numpy(d["state"][t].astype(np.float32)).unsqueeze(0),
             "task": ["intercept the drifting boat and grasp its mast"]})
    s = {names[0]: b[K1].float(), names[1]: b[K2].float(), "state": b["observation.state"].float()}
    if KIND == "vla":
        s["tokens"] = b[OBS_LANGUAGE_TOKENS]; s["mask"] = b[OBS_LANGUAGE_ATTENTION_MASK]
        s["noise"] = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim)
    for k, v in s.items():
        out.setdefault(k, []).append(v.cpu().numpy())
np.savez(OUT, **{k: np.stack(v) for k, v in out.items()})
print("校准数据：", {k: np.stack(v).shape for k, v in out.items()})
