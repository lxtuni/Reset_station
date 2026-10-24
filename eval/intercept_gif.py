"""拦截任务调试 GIF：左 = 斜视角，右 = 俯视；接触点和接触力显示出来；每帧写 步数 / 船速 / 夹持点距离 / 接触。
接触按物理子步记录（只看控制步末尾会漏掉几毫秒的撞击）。
    render: MUJOCO_GL=glfw .venv/bin/python eval/intercept_gif.py render V_LO V_HI SEED OUT_NPZ [model=…/best_model.zip] [idle]
            不给 model 用脚本拦截器；idle = 手臂不动（只看船的行为）
    gif:    ~/venvs/mjx/bin/python eval/intercept_gif.py gif OUT_NPZ OUT_GIF "标题" [clean]
"""
import sys
import numpy as np

if sys.argv[1] == "render":
    sys.path.insert(0, ".")
    import mujoco
    from envs.intercept_env import InterceptEnv, scripted_intercept
    vlo, vhi, seed, out = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    mp = [x[6:] for x in sys.argv if x.startswith("model=")]
    from stable_baselines3 import SAC
    model = SAC.load(mp[0]) if mp else None
    occl = "occl" in sys.argv or (bool(mp) and "occl" in mp[0])        # 模型目录带 occl / wrist 就打开对应的传感器
    wrist = "wrist" in sys.argv or (bool(mp) and "occl_wrist" in mp[0])   # 实验目录叫 stage7_wrist，要看 occl_wrist
    s8 = bool(mp) and "stage8" in mp[0]                      # stage8：两个标签 + 动作低通滤波 0.6；stage7：一个标签、不滤波
    env = InterceptEnv(predictor="history" if mp and "history" in mp[0] else "kf", v_range=(vlo, vhi), occlusion=occl, wrist=wrist,
                       two_tags=not (bool(mp) and "stage7" in mp[0]), action_filter=0.8 if (bool(mp) and "stage9" in mp[0]) else (0.6 if s8 or "filter" in sys.argv else 0.0))
    env.block_outward = s8 or (bool(mp) and "stage9" in mp[0]) or not mp   # 只有 stage8 之后训练的模型（和脚本）用"边界处不往外推"
    from envs.residual import SCALE
    from envs.intercept_env import TAG_LOCALS, TAG_HALF
    # 遮挡打开时，AprilTag 画成绿（看得到）/ 红（被挡）；腕部相机有读数时文字里显示
    residual = bool(mp) and "residual" in mp[0]               # 残差模型：动作 = 脚本拦截器 + SCALE × SAC（和训练时一样）
    m, d = env.m, env.d
    nm = lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name
    r = mujoco.Renderer(m, 360, 420)
    opt = mujoco.MjvOption()
    show_contacts = "clean" not in sys.argv                   # clean（README 用）：不画接触点和接触力
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contacts
    m.vis.map.force = 0.02
    side = mujoco.MjvCamera(); side.distance = 0.55; side.azimuth = 150; side.elevation = -30
    top = mujoco.MjvCamera(); top.lookat[:] = [0.44, 0, 0]; top.distance = 1.15; top.azimuth = 90; top.elevation = -89.9
    seen = set(); orig = env.water.apply
    def hook(dd):                                             # 记录每个物理子步的接触（只看控制步末尾会漏掉短暂撞击）
        for c in dd.contact[:dd.ncon]:
            seen.add(tuple(sorted((nm(c.geom1), nm(c.geom2)))))
        orig(dd)
    env.water.apply = hook
    obs, _ = env.reset(seed=seed)
    frames, texts = [], []
    while True:
        if "idle" in sys.argv:
            a = np.zeros(5)
        elif model is None:
            a = scripted_intercept(env.kf_obs())
        elif residual:
            a = np.clip(scripted_intercept(env.kf_obs()) + SCALE * model.predict(obs, deterministic=True)[0], -1, 1)
        else:
            a = model.predict(obs, deterministic=True)[0]
        seen.clear()
        obs, rew, t, tr, info = env.step(a)
        side.lookat[:] = [d.xpos[env.bid][0] - 0.05, d.xpos[env.bid][1], 0.02]
        imgs = []
        for cam in (side, top):
            r.update_scene(d, cam, opt)
            if env.occlusion:                                 # 每个 AprilTag：绿 = 顶部相机看得到，红 = 被机械臂挡住
                sc = r.scene; R = d.xmat[env.bid].reshape(3, 3)
                for loc, vis in zip(env.tag_locals, env.tags_visible()):
                    g = sc.geoms[sc.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([TAG_HALF, TAG_HALF, 0.0015]),
                                        d.xpos[env.bid] + R @ loc, R.ravel(),
                                        np.array([0.1, 0.8, 0.2, 1] if vis else [0.9, 0.1, 0.1, 1], np.float32))
                    sc.ngeom += 1
            imgs.append(r.render().copy())
        frames.append(np.concatenate(imgs, 1))
        bodies = lambda g: g.split("_sph")[0].split("_box")[0]  # 合并同一个零件的多个碰撞体，避免一行太长
        cs = sorted({f"{bodies(x)}-{bodies(y)}" for x, y in seen if "floor" not in x + y and "water" not in x + y})
        texts.append((f"顶部相机：{'被挡住（丢帧）' if info['occluded'] else '看得到标签'}  累计丢帧 {info['n_occluded']}/{env.steps}  " if env.occlusion else "")
                     + (f"腕部相机：{'看到桅杆' if env.wrist_age == 0 else '没看到'}  " if env.wrist else "") + f"step {env.steps:3d}  船速 {100*info['boat_speed']:4.1f} cm/s  夹持点距离 {1000*info['dist']:3.0f} mm"
                     f"\n夹住 {'是' if info['grasped'] else '否'}  连续静止 {env.hold}/10  撞墙 {info.get('wall_hits', 0)} 次  游回 {info.get('reaims', 0)} 次  推力 {'开' if env.swim_target is not None else '关'}"
                     f"  KF 速度误差 {100*info['kf_vel_err']:.1f} cm/s\n接触: {', '.join(cs) if cs else '无'}")
        if t or tr:
            end = "成功" if info["success"] else ("船撞墙" if info["wall"] else "超时")
            texts[-1] += f"\n结束: {end}"
            break
    np.savez_compressed(out, frames=np.array(frames), texts=np.array(texts))
    print(len(frames), "frames,", texts[-1])

else:
    from PIL import Image, ImageDraw, ImageFont
    D = np.load(sys.argv[2]); title = sys.argv[4]
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 16)
    big = ImageFont.truetype("/mnt/c/Windows/Fonts/msyhbd.ttc", 18)
    clean = "clean" in sys.argv                                # clean：只写标题和时间（README 用），不写调试信息
    head = 30 if clean else 112
    imgs = []
    for k, (f, txt) in enumerate(zip(D["frames"], D["texts"])):
        im = Image.new("RGB", (f.shape[1], f.shape[0] + head), (255, 255, 255))
        im.paste(Image.fromarray(f), (0, head))
        dr = ImageDraw.Draw(im)
        dr.text((8, 4), title + (f"    t = {0.05 * (k + 1):.1f} s" if clean else ""), fill=(30, 30, 30) if clean else (160, 20, 20), font=big)
        if not clean:
            dr.text((8, 28), str(txt), fill=(0, 0, 0), font=font)
        imgs.append(im)
    imgs += [imgs[-1]] * 20                                   # 结尾停 1 s
    imgs[0].save(sys.argv[3], save_all=True, append_images=imgs[1:], duration=50, loop=0, optimize=True)
    print("saved", sys.argv[3], len(imgs), "frames")
