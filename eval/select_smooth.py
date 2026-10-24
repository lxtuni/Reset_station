"""按"成功率 + 平滑度"挑模型（10-24）：best_model.zip 是按评估成功率挑的，不看平滑；训练中途的检查点常常更平滑、成功率一样。

    .venv/bin/python eval/select_smooth.py RUN_DIR [最少步数=100000] [最少成功=29]
对 RUN_DIR 里每个 ckpt_*_steps.zip（≥ 最少步数）跑 30 回合（船速 0、6–10、15–20 cm/s 各 10 个，种子 1000–1009），
在成功 ≥ 最少成功的检查点里挑"关节速度每步变号"最小的，复制成 RUN_DIR/smooth_model.zip，并把每个检查点的结果写进 smooth_select.json。
环境设置（遮挡、标签、滤波、平滑惩罚）按目录名决定，和 eval/intercept_ab.py 一样（stage9：两个标签 + 遮挡 + 滤波 0.8）。
"""
import os, sys, glob, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv

rd = sys.argv[1].rstrip("/")
min_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
min_succ = int(sys.argv[3]) if len(sys.argv) > 3 else 29
kw = dict(predictor="history" if "history" in rd else "kf", occlusion="occl" in rd, wrist="occl_wrist" in rd,
          two_tags="stage7" not in rd,
          action_filter=0.8 if "stage9" in rd else (0.6 if "stage8" in rd else 0.0))


BLOCK = "stage8" in rd or "stage9" in rd                 # "边界处不往外推"只对 stage8 之后训练的模型打开


def score(path):
    m = SAC.load(path)
    S, flips, acc = 0, [], []
    for vr in [(0.0, 0.0), (0.06, 0.10), (0.15, 0.20)]:
        env = InterceptEnv(v_range=vr, **kw)
        env.block_outward = BLOCK
        for s in range(10):
            o, _ = env.reset(seed=1000 + s); QD = []
            while True:
                o, r, t, tr, info = env.step(m.predict(o, deterministic=True)[0])
                QD.append(env.d.qvel[env.arm_va][:5].copy())
                if t or tr:
                    break
            QD = np.array(QD); S += info["success"]
            flips.append(np.mean((np.sign(QD[1:]) != np.sign(QD[:-1])) & (np.abs(QD[1:]) > 0.3) & (np.abs(QD[:-1]) > 0.3)))
            acc.append(np.sqrt(np.mean((np.diff(QD, axis=0) / 0.05) ** 2)))
    return dict(success=int(S), flips=float(np.mean(flips)), acc=float(np.mean(acc)))


cands = sorted(glob.glob(os.path.join(rd, "ckpt_*_steps.zip")), key=lambda p: int(p.split("_")[-2]))
cands = [p for p in cands if int(p.split("_")[-2]) >= min_steps] + [os.path.join(rd, "best_model.zip")]
res = {}
for p in cands:
    res[os.path.basename(p)] = score(p)
    print(os.path.basename(p), res[os.path.basename(p)], flush=True)
ok = {k: v for k, v in res.items() if v["success"] >= min_succ}
pick = min(ok, key=lambda k: ok[k]["flips"]) if ok else "best_model.zip"
shutil.copy(os.path.join(rd, pick), os.path.join(rd, "smooth_model.zip"))
json.dump(dict(picked=pick, min_succ=min_succ, results=res), open(os.path.join(rd, "smooth_select.json"), "w"), indent=1)
print(f"挑中 {pick}：{res[pick]}（best_model：{res['best_model.zip']}）")
