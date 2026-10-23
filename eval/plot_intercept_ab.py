"""拦截任务 A（KF 预测）/ B（原始测量历史）对比图 → docs/intercept_ab.png。
数据：results/stage5_intercept/eval_ab.json（eval/intercept_ab.py、eval/intercept_kf_tuned.py、eval/intercept_scripted.py 的结果）。
    ~/venvs/mjx/bin/python eval/plot_intercept_ab.py      # 需要 matplotlib（.venv 里没装）
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
plt.rcParams["font.family"] = "Microsoft YaHei"

R = json.load(open("results/stage5_intercept/eval_ab.json", encoding="utf-8"))
bins = ["0–0", "0–3", "3–6", "6–10", "10–15", "15–20"]
labels = ["0", "0–3", "3–6", "6–10", "10–15", "15–20"]
A = [v for k, v in R.items() if "/kf/" in k and "v1" not in k]
B = [v for k, v in R.items() if "/history/" in k]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.arange(6)
for runs, name, c in [(A, "A：KF 预测", "#2a6fdb"), (B, "B：原始测量历史", "#e07b22")]:
    m = np.array([[r["bins"][b]["succ"] / 30 * 100 for b in bins] for r in runs])
    ax[0].plot(x, m.mean(0), "-o", color=c, label=name)
    ax[0].fill_between(x, m.min(0), m.max(0), color=c, alpha=0.15)
    w = np.array([[r["bins"][b]["walls"] for b in bins] for r in runs])
    ax[1].plot(x, w.mean(0), "-o", color=c, label=name)
sc = R["scripted"]
ax[0].plot(x, np.array(sc["succ_per_bin_of_30"]) / 30 * 100, "--", color="gray", label="脚本拦截器")
ax[1].plot(x, sc["walls_per_bin"], "--", color="gray", label="脚本拦截器")
for a, t, yl in [(ax[0], "成功率（3 个种子平均，阴影 = 最差–最好）", "成功率 %"), (ax[1], "平均撞墙次数", "次 / 回合")]:
    a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("船初速度 cm/s"); a.set_ylabel(yl); a.set_title(t); a.legend(); a.grid(alpha=0.3)
ax[0].set_ylim(0, 105)
conds = ["标准（1 mm、延迟 1 帧）", "噪声 ×3（3 mm、1.5°）", "延迟 3 帧（150 ms）"]
a_v = [np.mean([r["cond"][c]["succ"] / 60 * 100 for r in A]) for c in conds]
b_v = [np.mean([r["cond"][c]["succ"] / 60 * 100 for r in B]) for c in conds]
kt = R["kf_tuned_noise3"]
a_tuned = np.mean([v for k, v in kt.items() if k != "n"]) / kt["n"] * 100
xc = np.arange(3)
ax[2].bar(xc - 0.2, a_v, 0.4, color="#2a6fdb", label="A：KF")
ax[2].bar(xc + 0.2, b_v, 0.4, color="#e07b22", label="B：原始测量历史")
ax[2].bar([1 - 0.2], [a_tuned], 0.4, color="none", edgecolor="#2a6fdb", hatch="//", label="A：KF 按 3 mm 重新配置")
for xi, v in zip(list(xc - 0.2) + list(xc + 0.2) + [0.8], a_v + b_v + [a_tuned]):
    ax[2].text(xi, v + 1.5, f"{v:.0f}", ha="center")
ax[2].set_xticks(xc); ax[2].set_xticklabels(["标准", "噪声 ×3", "延迟 3 帧"]); ax[2].set_ylim(0, 100)
ax[2].set_title("抗干扰（0–20 cm/s，每种子 60 回合）"); ax[2].set_ylabel("成功率 %"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("docs/intercept_ab.png", dpi=110)
print("已保存 docs/intercept_ab.png")
