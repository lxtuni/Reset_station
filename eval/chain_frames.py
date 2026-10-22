"""把完整链路的一个回合渲染成 5 张关键帧（2 × 3 排，最后一格空白），存成 docs/chain_episode.png（不开窗口）。

    MUJOCO_GL=glfw python eval/chain_frames.py [scripted|sac] [swim] [种子]
关键帧：开始 → 阶段 1 结束（夹爪在桅杆上方）→ 阶段 2 结束（夹住）→ 阶段 3 途中（拖着转向）→ 松手抬起后 3 s。
"""
import os, sys, zlib, struct
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.chain import Chain

mode = "scripted" if "scripted" in sys.argv else "sac"
swim = "swim" in sys.argv
seed = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 3

ch = Chain(mode)
r = mujoco.Renderer(ch.m, 360, 480)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.25, 0.0, 0.0]; cam.distance = 0.75; cam.azimuth = 150; cam.elevation = -35
frames, labels, state = [], [], dict(stage=None, n=0)


def grab(env, label):
    r.update_scene(env.d, cam)
    frames.append(r.render().copy()); labels.append(label)


def on_step(env):
    name = {id(ch.reach): "reach", id(ch.grasp): "grasp", id(ch.drag): "drag"}[id(env)]
    if state["stage"] != name:
        if state["stage"] is None and name == "reach":
            grab(env, "开始")
        state["stage"], state["n"] = name, 0
    state["n"] += 1
    if name == "drag" and state["n"] == 15:
        grab(env, "阶段 3 途中")


res = ch.run(seed=seed, swim=swim, on_step=on_step)
print(res)
# 阶段结束时的画面：run() 结束后各环境的数据还在，补拍
for env, lab in [(ch.reach, "阶段 1 结束：预抓取"), (ch.grasp, "阶段 2 结束：夹住")]:
    grab(env, lab)
grab(ch.drag, "松手后 3 s")
order = ["开始", "阶段 1 结束：预抓取", "阶段 2 结束：夹住", "阶段 3 途中", "松手后 3 s"]
imgs = [frames[labels.index(l)] for l in order if l in labels]
while len(imgs) < 6:
    imgs.append(np.full_like(imgs[0], 255))
img = np.concatenate([np.concatenate(imgs[:3], 1), np.concatenate(imgs[3:6], 1)], 0)
h, w, _ = img.shape
raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))
ch_ = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
out = "docs/chain_episode.png"
open(out, "wb").write(b"\x89PNG\r\n\x1a\n" + ch_(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                      + ch_(b"IDAT", zlib.compress(raw, 9)) + ch_(b"IEND", b""))
print("已保存", out, "顺序：", [l for l in order if l in labels])
