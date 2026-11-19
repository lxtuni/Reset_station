"""录一段带标注的完整流程：左边是场景的旁观视角，右边是模型看到的两路相机，下面逐步写出这一步调用了什么、得到了什么。
用的是 ACT（每步都重新推理）。用法（仓库根目录，PYTHONPATH=act）：python vla/make_demo.py MODEL_DIR OUT.gif [船速下限 上限] [字体文件]
"""
import sys, time
import numpy as np, torch, imageio, mujoco
from PIL import Image, ImageDraw, ImageFont
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from common import make_env, Cams, joints

M, OUT = sys.argv[1], sys.argv[2]
vr = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.10, 0.15)
FONT = sys.argv[5] if len(sys.argv) > 5 else None
font = ImageFont.truetype(FONT, 17) if FONT else ImageFont.load_default()
small = ImageFont.truetype(FONT, 14) if FONT else ImageFont.load_default()
policy = ACTPolicy.from_pretrained(M).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=M)
KEYS = list(policy.config.image_features)
f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
env = make_env(vr); cams = Cams(env)
scene = mujoco.Renderer(env.m, 360, 480)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.40, 0.0, 0.08]; cam.distance = 1.15; cam.azimuth = 150; cam.elevation = -32


def frame(top, wrist, lines):
    scene.update_scene(env.d, cam); big = scene.render()
    W, H = 800, 480 + 26 * len(lines) + 14
    img = Image.new("RGB", (W, H), (250, 250, 250))
    img.paste(Image.fromarray(big), (0, 60)); img.paste(Image.fromarray(top), (480, 0)); img.paste(Image.fromarray(wrist), (480, 240))
    d = ImageDraw.Draw(img)
    d.text((8, 8), "旁观视角（模型看不到这个画面）", font=small, fill=(60, 60, 60))
    d.text((486, 4), "顶部相机 → 模型输入", font=small, fill=(255, 255, 255)); d.text((486, 244), "腕部相机 → 模型输入", font=small, fill=(255, 255, 255))
    for i, s in enumerate(lines):
        d.text((10, 488 + 26 * i), s, font=font, fill=(20, 20, 20))
    return np.asarray(img)


for seed in range(30):
    env.reset(seed=seed); frames = []; t = 0
    while True:
        t0 = time.perf_counter(); top, wrist = cams(); q = joints(env); t1 = time.perf_counter()
        b = pre({KEYS[0]: f(top), KEYS[1]: f(wrist), "observation.state": torch.from_numpy(q).unsqueeze(0), "task": [""]})
        torch.cuda.synchronize(); t2 = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(b)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        a = post(chunk)[0, 0].cpu().numpy()
        obs, r, term, trunc, info = env.step(np.array(a, dtype=np.float32)); t += 1
        state = f"已夹住，稳住 {env.hold}/10 步" if info["grasped"] else f"未夹住，离桅杆 {info['dist'] * 100:.1f} cm"
        lines = [f"第 {t:3d} 步（{t * 0.05:.2f} 秒）   船速 {info['boat_speed'] * 100:4.1f} cm/s   {state}",
                 f"① cams() 渲染两路相机 {1000 * (t1 - t0):.1f} ms   ② pre() 归一化 {1000 * (t2 - t1):.1f} ms",
                 f"③ policy.predict_action_chunk()：ACT 推理 {1000 * (t3 - t2):.1f} ms，输出未来 20 步动作",
                 f"④ 取第 1 步：末端位移 x {a[0]:+.2f}  y {a[1]:+.2f}  z {a[2]:+.2f}   腕部旋转 {a[3]:+.2f}   夹爪 {a[4]:+.2f}",
                 "⑤ env.step()：动作平滑 → 逆运动学算关节角 → MuJoCo 向前走 50 ms → 回到 ①"]
        frames.append(frame(top, wrist, lines))
        if term or trunc:
            break
    print(f"seed {seed}: {'成功' if info['success'] else '失败'}，{t} 步，船速 {float(info['v0']):.3f}")
    if info["success"] and 55 <= t <= 100:
        frames += [frames[-1]] * 25
        imageio.mimsave(OUT, frames, duration=0.07, loop=0)
        print("已保存", OUT, f"{len(frames)} 帧"); break
