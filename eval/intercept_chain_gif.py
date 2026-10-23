"""完整链路 GIF（拦截 → 拖动转向 → 松手后 3 s），左斜视、右俯视，左上角写阶段和时间。

    MUJOCO_GL=glfw .venv/bin/python eval/intercept_chain_gif.py render V_LO V_HI SEED /tmp/chain.npz [icpt=…] [drag=…]
    ~/venvs/mjx/bin/python eval/intercept_chain_gif.py gif /tmp/chain.npz docs/intercept_chain.gif
"""
import os, sys
import numpy as np

STAGE = {"intercept": "1 · intercept: predict, catch up, grasp, stop",
         "drag": "2 · drag to target + turn the bow, unload, release",
         "after": "3 · released (arm still)"}

if sys.argv[1] == "render":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mujoco
    vlo, vhi, seed, out = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[6:]]      # icpt= / drag= 交给 intercept_chain 解析
    import eval.intercept_chain as C
    from envs.drag_env import TARGET_XY
    r = mujoco.Renderer(C.ienv.m, 360, 420)
    side = mujoco.MjvCamera(); side.distance = 0.6; side.azimuth = 150; side.elevation = -30
    top = mujoco.MjvCamera(); top.lookat[:] = [0.44, 0, 0]; top.distance = 1.15; top.azimuth = 90; top.elevation = -89.9
    frames, stages = [], []

    def grab(env, phase):
        side.lookat[:] = [env.d.xpos[env.bid][0] - 0.05, env.d.xpos[env.bid][1], 0.02]
        imgs = []
        for cam in (side, top):
            r.update_scene(env.d, cam)
            s = r.scene; g = s.geoms[s.ngeom]                 # 拖动终点：绿色小球
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.01, 0, 0]), np.array([TARGET_XY[0], TARGET_XY[1], 0.03]),
                                np.eye(3).ravel(), np.array([0.1, 0.85, 0.25, 0.9], np.float32)); s.ngeom += 1
            imgs.append(r.render().copy())
        frames.append(np.concatenate(imgs, 1)); stages.append(phase)

    res = C.episode(seed, (vlo, vhi), on_step=grab)
    print(res)
    np.savez_compressed(out, frames=np.array(frames), stages=np.array(stages))
else:
    from PIL import Image, ImageDraw, ImageFont
    D = np.load(sys.argv[2])
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyhbd.ttc", 15)
    imgs = []
    for k, (f, st) in enumerate(zip(D["frames"], D["stages"])):
        if k % 2:                                             # 每 2 帧取 1 帧、10 fps：仍是真实速度，文件小一半
            continue
        fr = Image.fromarray(f).resize((int(f.shape[1] * 0.8), int(f.shape[0] * 0.8)), Image.LANCZOS)
        im = Image.new("RGB", (fr.width, fr.height + 28), (255, 255, 255))
        im.paste(fr, (0, 28))
        ImageDraw.Draw(im).text((8, 4), f"{STAGE[str(st)]}    t = {0.05 * (k + 1):.1f} s", fill=(30, 30, 30), font=font)
        imgs.append(im)
    imgs[0].save(sys.argv[3], save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
    print("saved", sys.argv[3], len(imgs), "frames")
