"""把教师示教（npz）转成给 SmolVLA 微调用的 LeRobot 数据集。

和 ACT 用的 convert_to_lerobot.py 的区别只有两点：
  1. 相机改名为 camera1（顶部）和 camera2（腕部）：SmolVLA 基础模型预训练时相机就叫这两个名字
  2. 每一帧带一句任务描述（VLA 的语言输入）。ACT 的数据集里也存了，只是 ACT 不看它
用法（仓库根目录）：python vla/convert_vla.py data/raw_static data/raw_full_* data/vla_all
"""
import glob, shutil, sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "intercept the drifting boat and grasp its mast"
*raw_dirs, out = sys.argv[1:]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
img = {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
features = {
    "observation.images.camera1": dict(img),
    "observation.images.camera2": dict(img),
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    "action": {"dtype": "float32", "shape": (5,), "names": ["tcp_dx", "tcp_dy", "tcp_dz", "wrist_roll", "gripper"]},
}
shutil.rmtree(out, ignore_errors=True)
ds = LeRobotDataset.create(repo_id="local/" + out.rstrip("/").split("/")[-1], fps=20, features=features, root=out,
                           robot_type="so101_sim", use_videos=False, image_writer_threads=16)
n_ep = n_fr = 0
for rd in raw_dirs:
    for f in sorted(glob.glob(f"{rd}/ep_*.npz")):
        d = np.load(f)
        for t in range(len(d["action"])):
            ds.add_frame({"observation.images.camera1": d["top"][t], "observation.images.camera2": d["wrist"][t],
                          "observation.state": d["state"][t].astype(np.float32), "action": d["action"][t].astype(np.float32),
                          "task": TASK})
        ds.save_episode()
        n_ep += 1; n_fr += len(d["action"])
        if n_ep % 50 == 0:
            print(f"{n_ep} 个回合，{n_fr} 帧", flush=True)
ds.finalize()
print(f"完成：{n_ep} 个回合，{n_fr} 帧，数据集在 {out}")
