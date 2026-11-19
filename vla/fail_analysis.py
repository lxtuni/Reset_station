"""分析 SmolVLA（TensorRT 引擎，不加任何修正）失败的回合到底是怎么失败的。

环境里回合只有两种结束方式：成功（夹住桅杆并稳住 0.5 秒）或者 200 步（10 秒）超时。所以失败 = 超时。
对每个回合记录：有没有夹住过、夹住了多少步、最长连续夹住几步、离桅杆最近多近、撞墙几次、最后一步的状态。
用法（仓库根目录，PYTHONPATH=act:vla）：python vla/fail_analysis.py MODEL_DIR ENGINE [种子起点=0]
"""
import sys, json
import numpy as np
from common import BINS
from resid_env import ResidEnv

M, ENG = sys.argv[1], sys.argv[2]; S0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
env = ResidEnv(M, ENG, v_range=(0.0, 0.0))
rows = []
for vr in BINS:
    env.fixed_v = vr
    for ep in range(30):
        obs, _ = env.reset(seed=S0 + ep)
        g, dist, hold, best_hold, touch_wall, first_g, a_hist = [], [], 0, 0, 0, None, []
        while True:
            a_hist.append(env.a_vla.copy())
            obs, r, term, trunc, info = env.step(np.zeros(5, np.float32))
            x = env.last_info
            g.append(bool(x["grasped"])); dist.append(float(x["dist"]))
            hold = hold + 1 if x["grasped"] else 0; best_hold = max(best_hold, hold)
            if x["grasped"] and first_g is None:
                first_g = len(g)
            if term or trunc:
                break
        a_hist = np.array(a_hist)
        rows.append(dict(bin=f"{100 * vr[0]:.0f}-{100 * vr[1]:.0f}", seed=S0 + ep, success=bool(x["success"]), steps=len(g),
                         ever_grasped=bool(any(g)), grasp_frac=float(np.mean(g)), best_hold=int(best_hold), first_grasp=first_g,
                         min_dist=float(min(dist)), end_dist=float(dist[-1]), end_grasped=bool(g[-1]), walls=int(x["wall_hits"]),
                         end_boat_speed=float(x["boat_speed"]), grip_cmd_mean=float(a_hist[:, 4].mean()),
                         a_abs=float(np.abs(a_hist[:, :3]).mean())))
json.dump(rows, open(f"outputs/fail_analysis_{S0}.json", "w"), indent=1)
F = [r for r in rows if not r["success"]]; OK = [r for r in rows if r["success"]]
print(f"共 {len(rows)} 回合，成功 {len(OK)}，失败 {len(F)}")
never = [r for r in F if not r["ever_grasped"]]
brief = [r for r in F if r["ever_grasped"] and r["best_hold"] < 10]
held = [r for r in F if r["best_hold"] >= 10]
print(f"失败回合分类：从没夹住过 {len(never)} | 夹住过但每次不到 0.5 秒 {len(brief)} | 连续夹住超过 0.5 秒却没算成功 {len(held)}")
for name, grp in [("从没夹住过", never), ("夹住过但很短", brief), ("夹住超过 0.5 秒", held)]:
    if grp:
        print(f"  [{name}] 最近距离中位数 {np.median([r['min_dist'] for r in grp]) * 100:.1f} cm，结束时距离 {np.median([r['end_dist'] for r in grp]) * 100:.1f} cm，"
              f"撞墙 {np.mean([r['walls'] for r in grp]):.1f} 次，夹住的时间占比 {np.mean([r['grasp_frac'] for r in grp]) * 100:.0f}%，"
              f"结束时还夹着 {np.mean([r['end_grasped'] for r in grp]) * 100:.0f}%，结束时船速 {np.median([r['end_boat_speed'] for r in grp]) * 100:.1f} cm/s，"
              f"按速度档 {dict((b, sum(r['bin'] == b for r in grp)) for b in sorted({r['bin'] for r in grp}))}")
print(f"成功回合：第一次夹住在第 {np.median([r['first_grasp'] for r in OK]):.0f} 步，用时 {np.median([r['steps'] for r in OK]):.0f} 步，撞墙 {np.mean([r['walls'] for r in OK]):.2f} 次")
print(f"失败回合里撞过墙的比例 {np.mean([r['walls'] > 0 for r in F]) * 100:.0f}%，成功回合里 {np.mean([r['walls'] > 0 for r in OK]) * 100:.0f}%")
print("ANALYSIS DONE")
