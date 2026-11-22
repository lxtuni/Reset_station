"""强化学习后训练的第七种做法：按状态触发的分叉搜索，再蒸馏（专家迭代，Expert Iteration）。

前几次失败归结为三个条件没有同时满足：数据里要有正确动作；动作要和画面对得上；学习信号要能分清是动作好还是回合容易。
“搜索加蒸馏”（collect_search.py）找到了能救回 75% 失败回合的动作，但它是在固定的第 60 步插入的，和画面无关，所以学不会。
这次改三处：
  1. 触发按状态：夹爪和桅杆的距离连续 K 步没有再缩小、也没夹住（“停滞”），就在这一刻分叉。距离取自仿真器，只在采集时用
  2. 候选动作有 5 个（4 个方向各 8 步，外加后退并上抬 16 步），全部试一遍，取最快成功的那个，而不是第一个成功的
  3. 好坏用成对回放判断：同一个种子、同一个分叉点，原策略继续走 和 插入候选动作，两个结果直接比。
     只有“原来失败、现在成功”或“都成功但快了至少 20 步”才算有效，不需要打分网络
  救回之后如果又停滞，就在新的停滞点再搜一次（最多 3 层）。
保存的片段从分叉点开始（脱身动作 + SmolVLA 自己重新靠近并夹住），不含分叉前停滞的那几步，避免把“停着不动”也学进去。
没有教师参与。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/collect_branch.py MODEL_DIR ENGINE OUT_DIR 回合数 种子起点
"""
import os, sys, time
import numpy as np
from resid_env import ResidEnv

M, ENG, OUT, N, S0 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(OUT, exist_ok=True)
env = ResidEnv(M, ENG, alpha=1.0, fast_bias=0.7)
PRIMS = {1: (0.0, 0.0, 1.0), 2: (-1.0, 0.0, 0.6), 3: (0.0, 1.0, 0.4), 4: (0.0, -1.0, 0.4)}
NAMES = {1: "上抬", 2: "后退并上抬", 3: "向左并上抬", 4: "向右并上抬"}
CANDS = [(1, 8), (2, 8), (3, 8), (4, 8), (2, 16)]                # (哪个动作, 持续几步)。试跑时 8 个候选里这 5 个占了被采用次数的 22/26
K, EPS, T_MIN, T_MAX, GAIN, DEPTH = 12, 0.005, 25, 170, 20, 3    # 停滞判据：12 步（0.6 秒）内距离没有再缩小 5 毫米以上


def rollout(seed, plan=()):
    """plan：[(插入时刻, 动作编号, 持续步数, 停滞开始的时刻), ...]。返回结果，以及最后一次插入之后第一次停滞的时刻。"""
    rng = np.random.default_rng(seed * 7919 + 13)
    env.noise_fn = lambda: rng.standard_normal((env.chunk, env.adim)).astype(np.float32)
    env.reset(seed=seed)
    t_free = max([p[0] + p[2] for p in plan], default=0)        # 最后一次插入结束的时刻，之后才重新开始判断停滞
    tops, wrists, states, actions = [], [], [], []
    dist, grasped, best, t_best, trig = None, False, np.inf, 0, None
    while True:
        t = len(actions)
        if t == t_free:
            best, t_best = np.inf, t
        if t >= t_free and trig is None and dist is not None:
            if grasped or dist < best - EPS:
                best, t_best = min(best, dist), t
            elif t - t_best >= K and T_MIN <= t <= T_MAX:
                trig = (t, t_best, dist)
        delta = np.zeros(5, np.float32)
        for (t_i, k, dur, _) in plan:
            if t_i <= t < t_i + dur:
                delta = np.array([*PRIMS[k], 0.0, 1.0], np.float32) - env.a_vla
        tops.append(env.last_top); wrists.append(env.last_wrist); states.append(env.last_q)
        _, r, term, trunc, info = env.step(delta)
        actions.append(env.prev_a.copy())
        dist, grasped = float(env.last_info["dist"]), bool(env.last_info["grasped"])
        if term or trunc:
            break
    return dict(ok=bool(info["success"]), steps=len(actions), trig=trig, traj=(tops, wrists, states, actions), v0=float(env.last_info["v0"]))


better = lambda new, old: new["ok"] and (not old["ok"] or new["steps"] <= old["steps"] - GAIN)

same = 0
for s in range(S0, S0 + 2):
    a = rollout(s); b = rollout(s); same += int(a["ok"] == b["ok"] and a["steps"] == b["steps"] and a["trig"] == b["trig"])
print(f"可复现性检查：2 个种子各跑两遍，结果完全相同的有 {same} 个", flush=True)

st = dict(ok=0, fail=0, trig_ok=0, trig_fail=0, searched=0, resc=0, faster=0, still_fail=0, saved=0, frames=0, replays=0,
          ctrl=0, ctrl_diff=0, flaky=0, conf=0, conf_fail=0)
used, depth_used, trig_dist = {}, {}, []
t0 = time.time()
for ep in range(N):
    seed = S0 + ep
    base = rollout(seed)
    st["ok" if base["ok"] else "fail"] += 1
    if base["trig"] is not None:
        st["trig_ok" if base["ok"] else "trig_fail"] += 1
    cur, plan = base, []
    worth = lambda c: c["trig"] is not None and (not c["ok"] or c["steps"] - c["trig"][0] >= 50)
    # 对照重放：重放并不完全确定（多个进程同时用显卡时，偶尔同一个种子两次结果不同）。所以搜索之前先把原策略原样再跑一遍，
    # 如果这一遍自己就变好了，说明这个种子的结果本来就不稳，不能拿来做成对比较，跳过
    if worth(base):
        ctrl = rollout(seed); st["ctrl"] += 1; st["replays"] += 1
        st["ctrl_diff"] += int((ctrl["ok"], ctrl["steps"]) != (base["ok"], base["steps"]))
        if better(ctrl, base) or ctrl["trig"] != base["trig"]:
            st["flaky"] += 1
            cur = dict(base, trig=None)                       # 不搜索
    # 原策略成功的回合：只有停滞之后还拖了 50 步以上才值得搜（否则不可能快 20 步）
    while worth(cur) and len(plan) < DEPTH:
        t_tr, t_stall, d_tr = cur["trig"]
        if not plan:
            st["searched"] += 1; trig_dist.append(d_tr)
        best = None
        for (k, dur) in CANDS:
            res = rollout(seed, plan + [(t_tr, k, dur, t_stall)]); st["replays"] += 1
            if better(res, cur) and (best is None or res["steps"] < best[0]["steps"]):
                best = (res, (t_tr, k, dur, t_stall))
        if best is None:
            break
        conf = rollout(seed, plan + [best[1]]); st["conf"] += 1; st["replays"] += 1     # 复核：选中的做法再跑一遍，仍然更好才算数
        if not better(conf, cur):
            st["conf_fail"] += 1
            break
        cur, plan = conf, plan + [best[1]]
        used[(best[1][1], best[1][2])] = used.get((best[1][1], best[1][2]), 0) + 1
    if plan:
        st["resc" if not base["ok"] else "faster"] += 1
        depth_used[len(plan)] = depth_used.get(len(plan), 0) + 1
        tops, wrists, states, actions = cur["traj"]
        for i, (t_i, k, dur, _) in enumerate(plan):
            end = plan[i + 1][3] if i + 1 < len(plan) else cur["steps"]     # 到下一次停滞开始之前为止
            if end - t_i < 10:
                continue
            sl = slice(t_i, end)
            np.savez_compressed(f"{OUT}/ep_{st['saved']:04d}.npz", top=np.array(tops[sl]), wrist=np.array(wrists[sl]), state=np.array(states[sl]),
                                action=np.array(actions[sl], np.float32), seed=seed, v0=cur["v0"], t_int=t_i, prim=k, dur=dur, base_ok=base["ok"])
            st["saved"] += 1; st["frames"] += end - t_i
    elif not base["ok"] and cur["trig"] is not None:
        st["still_fail"] += 1
    if (ep + 1) % 10 == 0 or ep + 1 == N:
        print(f"{ep + 1}/{N} 回合：原策略成功 {st['ok']}（其中出现停滞 {st['trig_ok']}），失败 {st['fail']}（其中出现停滞 {st['trig_fail']}）| "
              f"搜索 {st['searched']} 次：失败变成功 {st['resc']}，成功变更快 {st['faster']}，失败且没救回 {st['still_fail']} | "
              f"对照重放 {st['ctrl']} 次（结果有出入 {st['ctrl_diff']}，因不稳跳过 {st['flaky']}），复核 {st['conf']} 次（没通过 {st['conf_fail']}）| "
              f"保存 {st['saved']} 段 {st['frames']} 帧，重放 {st['replays']} 次 | 已用 {(time.time() - t0) / 60:.0f} 分钟", flush=True)
print("各候选动作被采用的次数：", {f"{NAMES[k]}{d}步": v for (k, d), v in sorted(used.items(), key=lambda x: -x[1])})
print("搜索层数：", dict(sorted(depth_used.items())), "| 触发时离桅杆的距离（厘米）中位数", round(float(np.median(trig_dist)) * 100, 1) if trig_dist else None)
print(f"完成：{N} 回合，保存 {st['saved']} 段，{st['frames']} 帧到 {OUT}")
