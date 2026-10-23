"""拦截任务对比图 → docs/intercept_ab.png：脚本拦截器 / SAC（A：KF 观测，B：原始测量历史，v4，各 3 个种子）。
数据：results/stage5_intercept/eval/*_kftuned.json（eval/intercept_ab.py --kf-tuned 的结果；KF 按每种条件的真实噪声配置）。
    ~/venvs/mjx/bin/python eval/plot_intercept_ab.py      # 需要 matplotlib（.venv 里没装）
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
plt.rcParams["font.family"] = "Microsoft YaHei"

E = "results/stage5_intercept/eval/"
load = lambda pat: [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(E + pat))]
G = [("脚本拦截器", load("scripted_kftuned.json"), "gray", "--"),
     ("A：SAC + KF 观测", load("kf_sac_seed*_v4_kftuned.json"), "#2a6fdb", "-"),
     ("B：SAC + 原始测量历史", load("history_sac_seed*_v4_kftuned.json"), "#e07b22", "-")]
bins = ["0–0", "0–3", "3–6", "6–10", "10–15", "15–20"]
labels = ["0", "0–3", "3–6", "6–10", "10–15", "15–20"]
conds = ["标准（1 mm、延迟 1 帧）", "噪声 ×3（3 mm、1.5°）", "延迟 3 帧（150 ms）"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
x = np.arange(6)
for name, runs, c, ls in G:
    m = np.array([[r["bins"][b]["succ"] / r["bins"][b]["n"] * 100 for b in bins] for r in runs])
    ax[0].plot(x, m.mean(0), ls, marker="o", color=c, label=name)
    s = np.array([[r["bins"][b]["steps"] or np.nan for b in bins] for r in runs]) * 0.05
    ax[1].plot(x, np.nanmean(s, 0), ls, marker="o", color=c, label=name)
for a, t, yl in [(ax[0], "成功率（3 个种子平均）", "成功率 %"), (ax[1], "成功回合的用时（中位数）", "秒")]:
    a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("船初速度 cm/s"); a.set_ylabel(yl); a.set_title(t); a.grid(alpha=0.3)
ax[0].set_ylim(80, 101); ax[0].legend(fontsize=8, loc="lower left")
w = 0.26
for i, (name, runs, c, ls) in enumerate(G):
    v = [np.mean([r["cond"][k]["succ"] / r["cond"][k]["n"] * 100 for r in runs]) for k in conds]
    xs = np.arange(3) + (i - 1) * w
    ax[2].bar(xs, v, w, color=c, label=name, edgecolor="white")
    for xi, vi in zip(xs, v):
        ax[2].text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=7)
ax[2].set_xticks(np.arange(3)); ax[2].set_xticklabels(["标准", "噪声 ×3", "延迟 3 帧"]); ax[2].set_ylim(0, 105)
ax[2].set_title("抗干扰（0–20 cm/s，每种子 60 回合；KF 按噪声配置）"); ax[2].set_ylabel("成功率 %"); ax[2].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("docs/intercept_ab.png", dpi=110)
print("已保存 docs/intercept_ab.png")
