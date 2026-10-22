"""把完整链路的一个回合做成 GIF：docs/reset_chain.gif（每帧左上角写当前阶段，绿球 = 终点）。

分两步（.venv 里没装 Pillow；~/venvs/mjx 里有，避免往项目环境里多装包）：
    MUJOCO_GL=glfw python eval/chain_gif.py render reach=sac grasp=sac drag=sac swim 2   # .venv：渲染每帧 → /tmp/chain_frames.npz
    ~/venvs/mjx/bin/python eval/chain_gif.py gif                                         # Pillow：加文字、合成 GIF
最后一个数字参数是随机种子。控制频率 20 Hz，GIF 每 2 帧取 1 帧、10 fps，所以是真实速度。
"""
import os, sys
import numpy as np

FRAMES = "/tmp/chain_frames.npz"
OUT = "docs/reset_chain.gif"
LABELS = {"swim": "阶段 0 · BODY2 游进可抓区 / swim into reach",
          "reach": "阶段 1 · 预抓取 / pre-grasp (SAC)",
          "grasp": "阶段 2 · 夹住桅杆 / grasp (SAC)",
          "drag": "阶段 3 · 拖到终点并转向 / drag + turn (SAC)",
          "release": "松手后 / after release"}

if sys.argv[1] == "render":
    import mujoco
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.chain import Chain
    from envs.drag_env import TARGET_XY
    seed = int([a for a in sys.argv[2:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[2:]) else 2
    swim = "swim" in sys.argv
    ch = Chain("sac")
    r = mujoco.Renderer(ch.m, 360, 480)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.30, 0.0, 0.0]; cam.distance = 0.85; cam.azimuth = 150; cam.elevation = -35
    frames, stages, st = [], [], dict(phase="swim" if swim else "reach")
    names = {id(ch.reach): "reach", id(ch.grasp): "grasp", id(ch.drag): "drag"}

    def grab(env, phase):
        r.update_scene(env.d, cam)
        s = r.scene                                           # 终点标记：绿色小球（只画，不参与物理）
        g = s.geoms[s.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0, 0]),
                            np.array([TARGET_XY[0], TARGET_XY[1], 0.03]), np.eye(3).ravel(),
                            np.array([0.1, 0.85, 0.25, 0.9], np.float32))
        s.ngeom += 1
        frames.append(r.render().copy()); stages.append(phase)

    def on_step(env):
        name = names[id(env)]
        if name == "reach" and st["phase"] == "swim" and getattr(env, "steps", 0) > 0:   # 游进来时 steps 还没定义
            st["phase"] = "reach"
        if name != "reach":
            st["phase"] = "release" if ch.released else name
        grab(env, st["phase"])

    res = ch.run(seed=seed, swim=swim, on_step=on_step)
    print(res)
    np.savez_compressed(FRAMES, frames=np.array(frames), stages=np.array(stages), res=str(res))
    print(f"{len(frames)} 帧 → {FRAMES}")

elif sys.argv[1] == "gif":
    from PIL import Image, ImageDraw, ImageFont
    D = np.load(FRAMES)
    frames, stages = D["frames"], D["stages"]
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 18)
    imgs = []
    for i in range(0, len(frames), 2):
        im = Image.fromarray(frames[i])
        dr = ImageDraw.Draw(im)
        text = LABELS[str(stages[i])]
        dr.rectangle([0, 0, 480, 30], fill=(20, 20, 20))
        dr.text((8, 4), text, font=font, fill=(255, 255, 255))
        dr.text((400, 4), f"{i * 0.05:4.1f} s", font=font, fill=(200, 200, 200))
        imgs.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
    print(f"{len(imgs)} 帧，{os.path.getsize(OUT) / 1e6:.1f} MB → {OUT}")
