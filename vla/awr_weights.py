"""方案 B 的核心：用奖励给每一帧算一个“优势”，再换成训练时的权重（优势加权回归，Advantage-Weighted Regression）。

  1. 回报 G_t：从第 t 步到回合结束的奖励之和（带折扣 0.99）
  2. 打分网络 V(s)：一个小的全连接网络，输入仿真器的真实状态和时间，学着预测 G_t。它代表“在这个状态下，平均能拿多少分”
     为了不让它给自己见过的回合打分，把回合分成两半，互相训练、互相打分
  3. 优势 A_t = 接下来 10 步实际拿到的奖励 + 0.99^10 × V(10 步之后的状态) − V(当前状态)
     意思是：从这一帧开始执行的这 10 步动作，比平均水平好多少。大于 0 就是比平常好
  4. 权重 w_t = exp(A_t / β)，β 取优势的标准差，上限 20。好的动作多学，差的动作少学，但不是完全丢掉
权重存在每个回合文件旁边（ep_0000.w.npy），convert_vla.py 合并数据时会读它。
用法：python vla/awr_weights.py <自采数据的总权重> data/raw_b1_0 data/raw_b1_1 ...
"""
import glob, sys
import numpy as np, torch

MASS, dirs = float(sys.argv[1]), sys.argv[2:]
G, NSTEP, CLIP = 0.99, 10, 20.0
files = [f for d in dirs for f in sorted(glob.glob(f"{d}/ep_*.npz"))]
eps = []
for f in files:
    d = np.load(f)
    r, s = d["reward"].astype(np.float64), d["sobs"].astype(np.float32)
    T = len(r)
    ret = np.zeros(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = r[t] + G * acc; ret[t] = acc
    x = np.concatenate([s, (np.arange(T)[:, None] / 400.0).astype(np.float32)], 1)
    eps.append(dict(f=f, r=r, x=x, ret=ret, kicked=d["kicked"], succ=bool(d["success"]), stuck=bool(d["stuck"])))
print(f"{len(eps)} 个回合，{sum(len(e['r']) for e in eps)} 帧，其中成功 {sum(e['succ'] for e in eps)} 个；"
      f"成功回合的平均回报 {np.mean([e['ret'][0] for e in eps if e['succ']]):.1f}，失败回合 {np.mean([e['ret'][0] for e in eps if not e['succ']]):.1f}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
X = np.concatenate([e["x"] for e in eps]); mu, sd = X.mean(0), X.std(0) + 1e-6
Y = np.concatenate([e["ret"] for e in eps]); ym, ys = Y.mean(), Y.std()


def fit(train):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 1)).to(dev)
    x = torch.tensor((np.concatenate([eps[i]["x"] for i in train]) - mu) / sd, device=dev)
    y = torch.tensor((np.concatenate([eps[i]["ret"] for i in train]) - ym) / ys, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for it in range(3000):
        idx = torch.randint(0, len(x), (2048,), device=dev)
        loss = torch.nn.functional.mse_loss(net(x[idx]).squeeze(1), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


order = np.random.default_rng(0).permutation(len(eps)); folds = [order[::2], order[1::2]]
err = []
for k in range(2):
    net = fit(folds[1 - k])
    with torch.no_grad():
        for i in folds[k]:
            e = eps[i]
            e["v"] = net(torch.tensor((e["x"] - mu) / sd, device=dev)).squeeze(1).cpu().numpy() * ys + ym
            err.append(e["v"] - e["ret"])
err = np.concatenate(err)
print(f"打分网络在没见过的回合上：预测误差的标准差 {err.std():.2f}，回报本身的标准差 {ys:.2f}（解释了 {1 - err.var() / ys ** 2:.0%} 的波动）")

for e in eps:
    T = len(e["r"]); v = np.append(e["v"], np.zeros(NSTEP + 1))          # 回合结束之后的价值是 0
    r = np.append(e["r"], np.zeros(NSTEP))
    e["adv"] = np.array([sum(G ** k * r[t + k] for k in range(NSTEP)) + G ** NSTEP * v[t + NSTEP] - v[t] for t in range(T)])
A = np.concatenate([e["adv"] for e in eps]); beta = A.std()
for e in eps:
    e["w"] = np.minimum(np.exp(e["adv"] / beta), CLIP)
W = np.concatenate([e["w"] for e in eps]); scale = MASS / W.sum()
for e in eps:
    np.save(e["f"][:-4] + ".w.npy", (e["w"] * scale).astype(np.float32))

cat = lambda key, cond: np.concatenate([e[key][cond(e)] for e in eps] or [np.zeros(0)])
kick_ok = cat("w", lambda e: e["kicked"] & e["succ"]); kick_bad = cat("w", lambda e: e["kicked"] & (not e["succ"]))
print(f"优势：标准差 {beta:.2f}；权重（缩放前）：中位数 {np.median(W):.2f}，平均 {W.mean():.2f}，碰到上限 {CLIP:.0f} 的占 {np.mean(W >= CLIP):.1%}")
print(f"各类帧的平均权重（缩放前）：成功回合 {cat('w', lambda e: np.full(len(e['r']), e['succ'])).mean():.2f}，"
      f"失败回合 {cat('w', lambda e: np.full(len(e['r']), not e['succ'])).mean():.2f}，"
      f"推一把且最后成功 {kick_ok.mean() if len(kick_ok) else float('nan'):.2f}，推一把但最后失败 {kick_bad.mean() if len(kick_bad) else float('nan'):.2f}")
print(f"有效样本量（权重集中程度）：{W.sum() ** 2 / (W ** 2).sum():.0f} / {len(W)}；自采数据总权重缩放到 {MASS:.0f}")
print("WEIGHTS DONE")
