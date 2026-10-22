"""IK 测试（不开窗口，只打印结果）。在项目根目录运行：python tests/test_ik.py

测试 A：已知可达的目标 —— 先随机摆一个"夹爪竖直朝下"的姿态，用正运动学算出尖端位置当目标，
        再让 IK 从 home 姿态出发去解。目标一定够得着，所以成功率应接近 100 %，用来验证求解器本身。
测试 B：船区里的随机目标 —— 桅杆抓取高度上随机取点（含够不着的），看成功率随离底座距离怎么变，
        和 docs/reach_map.png 的 0.37 m 边界对比。
测试 C：跟踪 —— 模拟 RL 的用法：目标每步最多移动 1 cm，每步只迭代 3 次（从上一步的解热启动），看误差。
"""
import os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 让 "from envs.ik import IK" 找得到
from envs.ik import IK, IK_JOINTS

m = mujoco.MjModel.from_xml_path("assets/robotstudio_so101/reset_station.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
q_home = d.qpos.copy()
ik = IK(m)
rng = np.random.default_rng(0)


def summary(name, infos, t):
    ok = np.array([i["success"] for i in infos])
    it = np.array([i["iters"] for i in infos])
    ep = np.array([i["err_pos"] for i in infos]) * 1000
    er = np.degrees([i["err_rot"] for i in infos])
    print(f"{name}: 成功 {ok.mean():.1%}（{ok.sum()}/{len(ok)}）| 成功时迭代 中位 {np.median(it[ok]):.0f}、最多 {it[ok].max()}"
          f" | 成功时误差 位置 ≤ {ep[ok].max():.2f} mm、角度 ≤ {er[ok].max():.2f}° | 每次求解 {1000*t/len(infos):.2f} ms")
    reasons = {}
    for i in infos:
        if not i["success"]:
            reasons[i["reason"]] = reasons.get(i["reason"], 0) + 1
    if reasons:
        print("   失败原因：", reasons)


# ---------- 测试 A：已知可达 ----------
targets = []
probe = mujoco.MjData(m)
while len(targets) < 300:
    probe.qpos[:] = q_home
    for j in IK_JOINTS:
        jid = m.joint(j).id
        probe.qpos[m.jnt_qposadr[jid]] = rng.uniform(*m.jnt_range[jid])
    mujoco.mj_kinematics(m, probe)
    R = probe.xmat[ik.roll_body].reshape(3, 3)
    a = -(R @ ik.roll_axis)
    p = ik.tcp(probe)
    # 只要夹爪几乎竖直、高度在水面附近、并且在隔板前面（船区，x > 0.085）的姿态。
    # 不加 x 条件时，有约 2 % 的目标在机械臂身后（x < 0），只有"手臂向后折"的另一组解才够得着；从 home 出发的 IK
    # 会走向底座转角 ±110° 的限位，解不到。那片区域是平台和后墙，船去不了，所以不测。
    if a[2] < -0.999 and -0.02 < p[2] < 0.10 and p[0] > 0.085:
        targets.append(p.copy())
t0 = time.time()
infos_A = [ik.solve(q_home, p)[1] for p in targets]
summary("A 已知可达", infos_A, time.time() - t0)

# ---------- 测试 B：船区随机目标，按距离分组 ----------
pts = np.c_[rng.uniform(0.10, 0.45, 2000), rng.uniform(-0.25, 0.25, 2000), rng.uniform(0.0, 0.04, 2000)]
t0 = time.time()
infos_B = [ik.solve(q_home, p)[1] for p in pts]
dt = time.time() - t0
summary("B 船区随机", infos_B, dt)
r = np.hypot(pts[:, 0], pts[:, 1]); ok = np.array([i["success"] for i in infos_B])
print("   按离底座的水平距离分组的成功率：")
for lo in np.arange(0.10, 0.46, 0.05):
    s = (r >= lo) & (r < lo + 0.05)
    if s.any():
        print(f"     {lo:.2f}–{lo+0.05:.2f} m：{ok[s].mean():6.1%}  （{s.sum()} 个点）")

# ---------- 测试 C：跟踪（RL 用法） ----------
ik_fast = IK(m, max_iter=3)                                 # 每个控制步只迭代 3 次
q = q_home.copy()
mujoco.mj_kinematics(m, probe); probe.qpos[:] = q; mujoco.mj_kinematics(m, probe)
p = ik.tcp(probe).copy()
path = [p]
goal = np.array([0.25, 0.10, 0.02])                          # 离底座 0.27 m，在竖直可达范围（≤ 0.32 m）内
while np.linalg.norm(goal - path[-1]) > 1e-9:               # 从 home 的尖端位置，每步 1 cm 走向 goal
    step = goal - path[-1]
    path.append(path[-1] + step * min(1.0, 0.01 / np.linalg.norm(step)))
errs = []
for target in path[1:]:
    q, info = ik_fast.solve(q, target)
    errs.append(info["err_pos"] * 1000)
for _ in range(10):                                         # 到终点后再给 10 步让它收敛
    q, info = ik_fast.solve(q, goal)
    errs.append(info["err_pos"] * 1000)
print(f"C 跟踪：{len(path)-1} 步、每步 1 cm、每步 3 次迭代 → 途中位置误差最大 {max(errs[:len(path)-1]):.2f} mm，"
      f"终点误差 {errs[-1]:.2f} mm，角度 {np.degrees(info['err_rot']):.2f}°")
