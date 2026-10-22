"""画训练曲线：python train/plot_curves.py results/stage1_reach/sac_seed0
需要 matplotlib（.venv 里还没装：uv pip install matplotlib）。中文用 Windows 的微软雅黑。

左：训练回合的回报（灰点 = 每个回合，线 = 最近 50 回合的滑动平均）；
中：训练回合的成功率（最近 50 回合的滑动平均；训练时动作带随机探索，所以比评估低）；
右：每 5000 步的评估（确定性策略 20 回合）的成功率和回报。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = [fm.FontProperties(fname=FONT).get_name()]
plt.rcParams["axes.unicode_minus"] = False

run = sys.argv[1] if len(sys.argv) > 1 else "results/stage1_reach/sac_seed0"
import csv
with open(os.path.join(run, "monitor.monitor.csv")) as f:      # Monitor 会在给定的名字后面自动加 .monitor.csv
    next(f)                                                    # 第一行是 JSON 注释（开始时间等），跳过
    rows = list(csv.DictReader(f))
mon = dict(r=np.array([float(x["r"]) for x in rows]), l=np.array([int(x["l"]) for x in rows]),
           success=np.array([x["success"] == "True" for x in rows]))   # 成功列存的是文字 True / False
steps = np.cumsum(mon["l"])                                   # 每个回合结束时累计的训练步数
def smooth(x, k=50):
    return np.array([x[max(0, i - k + 1):i + 1].mean() for i in range(len(x))])

ev = np.load(os.path.join(run, "evaluations.npz"))
ev_t, ev_r = ev["timesteps"], ev["results"].mean(1)
ev_s = ev["successes"].mean(1) if "successes" in ev.files and ev["successes"].size else None

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
ax[0].plot(steps, mon["r"], ".", color="0.75", ms=3)
ax[0].plot(steps, smooth(mon["r"]), color="C0", lw=2)
ax[0].set(title="训练回合的回报", xlabel="训练步数", ylabel="回报")
ax[1].plot(steps, smooth(mon["success"].astype(float)), color="C2", lw=2)
ax[1].set(title="训练回合的成功率（滑动 50）", xlabel="训练步数", ylim=(-0.02, 1.02))
if ev_s is not None:
    ax[2].plot(ev_t, ev_s, "o-", color="C2", label="成功率")
ax2 = ax[2].twinx()
ax2.plot(ev_t, ev_r, "s--", color="C0", label="平均回报")
ax[2].set(title="评估（确定性策略，20 回合）", xlabel="训练步数", ylabel="成功率", ylim=(-0.02, 1.02))
ax2.set_ylabel("回报")
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(run, "curves.png")
plt.savefig(out)
print("已保存", out)
