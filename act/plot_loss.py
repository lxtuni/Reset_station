"""训练损失曲线：纯模仿 6 万步 + DAgger 从 3 万步检查点接着训练 3 万步。日志每 200 步打印一行。
用法（仓库根目录）：python act/plot_loss.py
"""
import re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    txt = open(path, errors="ignore").read()
    l1 = np.array([float(x) for x in re.findall(r" l1_loss:([0-9.]+)", txt)])
    kl = np.array([float(x) for x in re.findall(r" kld_loss:([0-9.]+)", txt)])
    return l1, kl


a, ka = read("outputs_train_full.log"); b, kb = read("outputs_train_d1.log")
xa = (np.arange(len(a)) + 1) * 200; xb = 30000 + (np.arange(len(b)) + 1) * 200
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(xa, a, label="imitation only (400 demos)")
ax[0].plot(xb, b, label="DAgger fine-tune from step 30k (560 episodes)")
ax[0].set_yscale("log"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("L1 loss (log scale)")
ax[0].set_title("L1 loss: predicted vs teacher actions"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].plot(xa, ka); ax[1].plot(xb, kb)
ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].set_xlabel("training step"); ax[1].set_title("KL loss"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("outputs/visuals/loss_curve.png", dpi=130)
print(len(a), len(b), "L1:", a[0], a[149], a[-1], "| d1:", b[0], b[-1], "| KL:", ka[0], ka[10], ka[-1])
