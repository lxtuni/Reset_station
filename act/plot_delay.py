"""延迟注入实验的图：成功率随注入延迟的变化，同步对异步。读 outputs/delay_sweep.csv，存 docs/act/delay.png。"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [l.strip().split(",") for l in open("outputs/delay_sweep.csv") if l.strip()]
d = np.array([[float(r[1]), int(r[3]), float(r[4]), float(r[5]), int(r[7])] for r in rows])
base = d[d[:, 0] == 0][0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, name, mk in [(0, "synchronous (arm holds while waiting)", "o-"), (1, "asynchronous (keeps executing old chunk)", "s-")]:
    s = d[d[:, 1] == a]
    if a == 1:
        s = np.vstack([base, s])                       # 没有延迟时两者相同
    n = s[:, 4]; p = s[:, 2] / 100
    ax[0].errorbar(s[:, 0], s[:, 2], yerr=100 * np.sqrt(p * (1 - p) / n), fmt=mk, capsize=3, label=name)
    ax[1].plot(s[:, 0], s[:, 3], mk, label=name)
ax[0].set_xlabel("injected inference delay (ms)"); ax[0].set_ylabel("success rate (%)"); ax[0].set_ylim(0, 100)
ax[0].set_title("fast boat 10-20 cm/s, 60 episodes each (bars: 1 s.e.)"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].set_xlabel("injected inference delay (ms)"); ax[1].set_ylabel("wall hits per episode"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("docs/act/delay.png", dpi=130)
