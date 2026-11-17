"""把 collect_demos.py 存的原始回合（npz）转成 LeRobot 数据集，给 lerobot-train 用。

用法（仓库根目录）：python act/convert_to_lerobot.py data/raw_full data/lerobot_full [更多原始文件夹...]
    第一个参数之后、最后一个参数之前可以有多个原始文件夹，会合并进同一个数据集：
    python act/convert_to_lerobot.py data/raw_static data/raw_full data/lerobot_all
"""
import glob, shutil, sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

*raw_dirs, out = sys.argv[1:]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
img = {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}
features = {
    "observation.images.top": dict(img),
    "observation.images.wrist": dict(img),
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    "action": {"dtype": "float32", "shape": (5,), "names": ["tcp_dx", "tcp_dy", "tcp_dz", "wrist_roll", "gripper"]},
}
shutil.rmtree(out, ignore_errors=True)
ds = LeRobotDataset.create(repo_id="local/" + out.rstrip("/").split("/")[-1], fps=20, features=features, root=out,
                           robot_type="so101_sim", use_videos=False, image_writer_threads=8)
n_ep = n_fr = 0
for rd in raw_dirs:
    for f in sorted(glob.glob(f"{rd}/ep_*.npz")):
        d = np.load(f)
        for t in range(len(d["action"])):
            ds.add_frame({"observation.images.top": d["top"][t], "observation.images.wrist": d["wrist"][t],
                          "observation.state": d["state"][t].astype(np.float32), "action": d["action"][t].astype(np.float32),
                          "task": "intercept the drifting boat and grasp its mast"})
        ds.save_episode()
        n_ep += 1; n_fr += len(d["action"])
        if n_ep % 25 == 0:
            print(f"{n_ep} 个回合，{n_fr} 帧", flush=True)
ds.finalize()
print(f"完成：{n_ep} 个回合，{n_fr} 帧，数据集在 {out}")
