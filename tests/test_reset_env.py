"""ReachEnv 的自动测试（不开窗口）。在项目根目录运行：python tests/test_reset_env.py

1. Gymnasium 自带的 check_env：检查接口格式（返回值的类型、形状、范围）是否符合规定。
2. 随机策略 50 回合：作为"最差"的参照。
3. 脚本策略 50 回合：直接朝目标走。成功率应该接近 100 %，否则说明环境本身有问题（目标够不着、奖励写错……）。
4. 速度：每秒能跑多少个控制步（决定训练要多久）。
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymnasium.utils.env_checker import check_env
from envs.reset_env import ReachEnv, scripted_action

env = ReachEnv()
check_env(env, skip_render_check=True)
print("1. check_env 通过")


def run(policy, n=50, seed=0):
    ok, steps, coll, ret, final = [], [], [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep)
        R, hit = 0.0, False
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            R += r; hit |= info["collided"]
            if term or trunc:
                break
        ok.append(info["success"]); steps.append(env.steps); coll.append(hit); ret.append(R); final.append(info["dist"])
    ok = np.array(ok)
    return (f"成功 {ok.mean():.0%} | 成功时步数 中位 {np.median(np.array(steps)[ok]) if ok.any() else float('nan'):.0f}"
            f" | 碰撞回合 {np.mean(coll):.0%} | 回报 平均 {np.mean(ret):.1f} | 结束时距离 中位 {1000*np.median(final):.0f} mm")


rng = np.random.default_rng(0)
print("2. 随机策略：", run(lambda o: rng.uniform(-1, 1, 3)))
print("3. 脚本策略：", run(scripted_action))

env.reset(seed=123)
t0, n = time.time(), 0
while time.time() - t0 < 5.0:
    _, _, term, trunc, _ = env.step(env.action_space.sample()); n += 1
    if term or trunc:
        env.reset()
sps = n / (time.time() - t0)
print(f"4. 速度：{sps:.0f} 控制步/秒（单核）→ 10 万步约 {100000/sps/60:.0f} 分钟")
