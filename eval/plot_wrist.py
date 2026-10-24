"""顶部相机遮挡 / 腕部相机实验的对比图 → docs/wrist_camera.png。
五种传感器设置 × 三种方法（脚本、A：SAC + KF、B：SAC + 原始测量历史）：
  理想（顶部相机永远看得到，results/stage5_intercept/eval/*_v4_kftuned.json）
  一个标签（桅杆前）：只遮挡、遮挡 + 腕部相机（results/stage7_wrist/eval/*.json）
  船头船尾两个标签 + 动作滤波 0.6：只遮挡、遮挡 + 腕部相机（results/stage8_two_tags/eval/*.json）；都是 eval/intercept_ab.py --kf-tuned
    ~/venvs/mjx/bin/python eval/plot_wrist.py
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
plt.rcParams["font.family"] = "Microsoft YaHei"


def load(pat):
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(pat))]


def overall(runs):
    """每个种子：6 个速度档 × 30 回合的总成功率 %。"""
    return [100 * sum(r["bins"][b]["succ"] for b in r["bins"]) / sum(r["bins"][b]["n"] for b in r["bins"]) for r in runs]


S5, S7, S8 = "results/stage5_intercept/eval/", "results/stage7_wrist/eval/", "results/stage8_two_tags/eval/"
settings = ["理想：顶部相机\n永远看得到", "一个标签\n会被手臂挡住", "一个标签\n+ 腕部相机", "船头船尾两个标签\n会被手臂挡住", "两个标签\n+ 腕部相机"]
methods = [("脚本拦截器", "gray", [S5 + "scripted_kftuned.json", S7 + "scripted_occl_kftuned.json", S7 + "scripted_occl_wrist_kftuned.json",
                                    S8 + "scripted_occl_kftuned.json", S8 + "scripted_occl_wrist_kftuned.json"]),
           ("A：SAC + KF 融合", "#2a6fdb", [S5 + "kf_sac_seed*_v4_kftuned.json", S7 + "kf_occl_sac_seed*_kftuned.json", S7 + "kf_occl_wrist_sac_seed*_kftuned.json",
                                          S8 + "kf_occl_sac_seed*_kftuned.json", S8 + "kf_occl_wrist_sac_seed*_kftuned.json"]),
           ("B：SAC + 原始测量", "#e07b22", [S5 + "history_sac_seed*_v4_kftuned.json", S7 + "history_occl_sac_seed*_kftuned.json", S7 + "history_occl_wrist_sac_seed*_kftuned.json",
                                           S8 + "history_occl_sac_seed*_kftuned.json", S8 + "history_occl_wrist_sac_seed*_kftuned.json"])]
fig, ax = plt.subplots(figsize=(13, 4.6))
w = 0.26
for i, (name, c, pats) in enumerate(methods):
    for j, pat in enumerate(pats):
        v = overall(load(pat))
        x = j + (i - 1) * w
        ax.bar(x, np.mean(v), w, color=c, label=name if j == 0 else None, edgecolor="white")
        if len(v) > 1:                                        # 每个种子画一个点
            ax.scatter([x] * len(v), v, s=14, color="black", zorder=3)
        ax.text(x, np.mean(v) + 2, f"{np.mean(v):.0f}", ha="center", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(settings); ax.set_ylim(0, 108); ax.set_ylabel("成功率 %（0–20 cm/s，6 档 × 30 回合）")
ax.set_title("拦截成功率：顶部相机遮挡、标签数量与腕部相机（SAC 3 个种子，黑点 = 每个种子）")
ax.legend(loc="upper left", bbox_to_anchor=(0.215, 0.97), fontsize=8); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("docs/wrist_camera.png", dpi=110)
print("已保存 docs/wrist_camera.png")
