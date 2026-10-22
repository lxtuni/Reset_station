"""完整复位链路：阶段 1 预抓取 → 阶段 2 夹住 → 阶段 3 拖到终点、转向、松开。三个技能（策略）依次接力。

每个阶段是一个独立的环境对象（ReachEnv / GraspEnv / DragEnv，同一个场景 XML），接力时把上一个环境的完整状态
（仿真状态 + 指令变量）拷进下一个环境，所以物理上是同一个连续的过程。
可选"阶段 0"：BODY2 先从缸的远端游进可抓区（和 view_scene.py 一样用沿船头的推力代替游泳），停桨后等它基本停下再开始。

运行（项目根目录，激活 .venv）：
    python eval/chain.py                 # 50 个回合的统计（用训练好的 SAC；某阶段没有模型时退回脚本策略）
    python eval/chain.py scripted        # 三个阶段都用脚本策略（对照）
    python eval/chain.py pure            # 只用纯 SAC（不用残差版，对照）
    python eval/chain.py grasp=scripted drag=sac   # 每个阶段单独指定：scripted / sac / res（残差 SAC）/ resc（阶段 2：用接力起点训练的残差 SAC）
    python eval/chain.py sac swim        # 加上阶段 0：船从远处游进来
    python eval/chain.py sac view        # 开 viewer 看（加 swim 同理），每个回合结束停 1 秒
"""
import os, sys, time
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.reset_env import ReachEnv, scripted_action, GRIP_OPEN
from envs.grasp_env import GraspEnv, scripted_grasp
from envs.drag_env import DragEnv, scripted_drag, snapshot, restore, TARGET_XY

MODELS = {"reach": "results/stage1_reach/sac_seed0/best_model.zip",
          "grasp": "results/stage2_grasp/sac_seed0_demo30000/best_model.zip",
          "drag": "results/stage3_drag/sac_seed0_demo30000/best_model.zip"}
MAX_STEPS = {"reach": 100, "grasp": 100, "drag": 200}


RESIDUAL = {"grasp": "results/stage2_grasp/residual/sac_seed0/best_model.zip",
            "drag": "results/stage3_drag/residual/sac_seed0/best_model.zip"}


def load_policies(mode):
    """mode = "sac"：每个阶段优先用残差 SAC（脚本 + 0.3 × RL），其次纯 SAC，都没有就用脚本；
    mode = "pure"：只用纯 SAC（对照）；mode = "scripted"：全用脚本。"""
    from envs.residual import SCALE
    scripted = {"reach": scripted_action, "grasp": scripted_grasp, "drag": scripted_drag}
    pols, used = {}, {}
    for k, path in MODELS.items():
        res = RESIDUAL.get(k)
        if mode == "sac" and res and os.path.exists(res):
            from stable_baselines3 import SAC
            mdl = SAC.load(res)
            pols[k] = (lambda m, b: (lambda o: np.clip(b(o) + SCALE * m.predict(o, deterministic=True)[0], -1, 1)))(mdl, scripted[k])
            used[k] = "残差 SAC"
        elif mode in ("sac", "pure") and os.path.exists(path):
            from stable_baselines3 import SAC
            mdl = SAC.load(path)
            pols[k] = (lambda m: (lambda o: m.predict(o, deterministic=True)[0]))(mdl)
            used[k] = "SAC"
        else:
            pols[k], used[k] = scripted[k], "脚本"
    # 命令行里的 阶段=策略 覆盖上面的默认选择，例如 grasp=scripted drag=sac
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=")
            if v == "scripted":
                pols[k], used[k] = scripted[k], "脚本"
            else:
                from stable_baselines3 import SAC
                path = {"res": RESIDUAL.get(k), "resc": "results/stage2_grasp/residual_chain/sac_seed0/best_model.zip"}.get(v, MODELS[k])
                mdl = SAC.load(path)
                if v in ("res", "resc"):
                    pols[k] = (lambda m, bb: (lambda o: np.clip(bb(o) + SCALE * m.predict(o, deterministic=True)[0], -1, 1)))(mdl, scripted[k])
                    used[k] = "残差 SAC"
                else:
                    pols[k] = (lambda m: (lambda o: m.predict(o, deterministic=True)[0]))(mdl)
                    used[k] = "SAC"
    return pols, used


class Chain:
    def __init__(self, mode="sac"):
        self.reach, self.grasp, self.drag = ReachEnv(), GraspEnv(), DragEnv()
        self.pols, self.used = load_policies(mode)
        self.m = self.reach.m

    def _to_grasp(self):
        s = snapshot_reach(self.reach)
        restore(self.grasp, s)
        g = self.grasp
        g.boat_xy0 = g.d.xpos[g.bid][:2].copy(); g.hold = 0; g.steps = 0; g.prev_action = np.zeros(5)
        return g._obs()

    def _to_drag(self):
        restore(self.drag, snapshot(self.grasp))
        dr = self.drag
        dr.boat_xy0 = dr.d.xpos[dr.bid][:2].copy(); dr.hold = dr.release = 0; dr.steps = 0; dr.prev_action = np.zeros(5)
        return dr._obs()

    def swim_in(self, rng, on_step=None):
        """阶段 0：船从远端出发、朝可抓区游（推力 = 该速度下的阻力），进区停桨，等它慢下来（< 0.5 cm/s）或 10 s。"""
        e = self.reach
        for _ in range(100):
            mujoco.mj_resetDataKeyframe(e.m, e.d, e.key_home)
            e.d.qpos[e.grip_qa] = GRIP_OPEN; e.d.ctrl[e.grip_act] = GRIP_OPEN
            x, y = rng.uniform(0.45, 0.85), rng.uniform(-0.15, 0.15)
            tx, ty = rng.uniform(0.20, 0.27), rng.uniform(-0.08, 0.08)
            yaw = np.arctan2(ty - y, tx - x) + np.radians(rng.uniform(-8, 8))
            e.d.qpos[e.bqa:e.bqa + 7] = [x, y, e.water.water_z - 0.0168 + 0.025, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(e.m, e.d)
            if not any(c.geom1 in e.body2_geoms or c.geom2 in e.body2_geoms for c in e.d.contact[:e.d.ncon]):
                break
        v = rng.uniform(0.05, 0.15)
        e.d.qvel[e.bva:e.bva + 2] = [v * np.cos(yaw), v * np.sin(yaw)]
        swimming, t = True, 0
        while t < 40 / 0.05:                                        # 最多 40 s
            for _ in range(e.n_sub):
                mujoco.mj_step1(e.m, e.d); e.water.apply(e.d)
                if swimming:
                    head = e.d.xmat[e.bid].reshape(3, 3)[:, 0].copy(); head[2] = 0; head /= np.linalg.norm(head)
                    e.d.xfrc_applied[e.bid, :3] += (e.water.c1_lin[0] * v + e.water.c2_lin[0] * v * v) * head
                else:
                    # 进区后主动刹车（反向划水）：力 = -0.6 N·s/m × 水平速度。假设：真机能反向划水减速（还没测过）。
                    # 不刹车时船滑行、撞隔板弹回，停下时离底座中位 0.31 m，超出竖直可达的 0.30 m（10-22）。
                    e.d.xfrc_applied[e.bid, :2] += -0.6 * e.d.qvel[e.bva:e.bva + 2]
                mujoco.mj_step2(e.m, e.d)
            t += 1
            bx, by = e.d.xpos[e.bid][:2]
            if swimming and np.hypot(bx, by) < 0.27 and bx > 0.15:
                swimming = False                                  # 进入可抓区：停桨
            if on_step:
                on_step(e)
            if not swimming and np.linalg.norm(e.d.qvel[e.bva:e.bva + 2]) < 0.005:
                break
        # 把"已经静止在可抓区"的状态交给阶段 1
        e.q_cmd = e.d.qpos.copy(); e.tcp_cmd = e.ik.tcp(e.d).copy(); e.roll_cmd = float(e.d.qpos[e.roll_qa])
        e.prev_action = np.zeros(4); e.steps = 0
        bx, by = e.d.xpos[e.bid][:2]
        return e._obs(), dict(swim_s=t * 0.05, boat_r=float(np.hypot(bx, by)))

    def run(self, seed, swim=False, on_step=None):
        """跑一整个复位过程。返回每个阶段的结果。on_step(env) 每个控制步调用一次（给 viewer 用）。"""
        out = dict(seed=seed)
        self.released = False
        rng = np.random.default_rng(seed)
        if swim:
            obs, info0 = self.swim_in(rng, on_step)
            out.update(info0)
        else:
            obs, _ = self.reach.reset(seed=seed)
        for stage, env in [("reach", self.reach), ("grasp", self.grasp), ("drag", self.drag)]:
            if stage == "grasp":
                obs = self._to_grasp()
            elif stage == "drag":
                obs = self._to_drag()
            ok, n = False, 0
            for n in range(1, MAX_STEPS[stage] + 1):
                obs, r, term, trunc, info = env.step(self.pols[stage](obs))
                if on_step:
                    on_step(env)
                if info["success"]:
                    ok = True
                    break
            out[stage] = ok; out[stage + "_steps"] = n
            if not ok:
                out["failed_at"] = stage
                break
        if out.get("drag"):
            dr = self.drag
            out["pos_err_mm"] = 1000 * float(np.linalg.norm(TARGET_XY - dr.d.xpos[dr.bid][:2]))
            out["yaw_err_deg"] = float(np.degrees(dr._errs()[1]))
            p0 = dr.d.xpos[dr.bid][:2].copy()                         # 松手后再看 3 s，船漂了多远（已知局限）
            self.released = True                                      # 给 on_step 用：之后的画面算"松手后"
            for _ in range(60):
                dr.step(np.array([0, 0, 0.0, 0, 1.0]))
                if on_step:
                    on_step(dr)
            out["drift_3s_mm"] = 1000 * float(np.linalg.norm(dr.d.xpos[dr.bid][:2] - p0))
            out["total_s"] = 0.05 * (out["reach_steps"] + out["grasp_steps"] + out["drag_steps"])
        return out


def snapshot_reach(e):
    s = dict(qpos=e.d.qpos.copy(), qvel=e.d.qvel.copy(), ctrl=e.d.ctrl.copy(), q_cmd=e.q_cmd.copy(),
             tcp_cmd=e.tcp_cmd.copy(), roll_cmd=np.array(e.roll_cmd), grip_cmd=np.array(GRIP_OPEN))
    return s


if __name__ == "__main__":
    mode = "scripted" if "scripted" in sys.argv else ("pure" if "pure" in sys.argv else "sac")
    swim, view = "swim" in sys.argv, "view" in sys.argv
    ch = Chain(mode)
    print("各阶段用的策略：", ch.used, "| 阶段 0 游进来：", swim)
    if view:
        import mujoco.viewer
        envs = [ch.reach, ch.grasp, ch.drag]
        with mujoco.viewer.launch_passive(ch.m, ch.reach.d) as viewer:
            def show(env):
                # 三个环境各有自己的 MjData：把当前这个的状态拷到 viewer 显示的那份（reach.d）里
                if env is not ch.reach:
                    ch.reach.d.qpos[:] = env.d.qpos; ch.reach.d.qvel[:] = env.d.qvel; mujoco.mj_forward(ch.m, ch.reach.d)
                viewer.sync(); time.sleep(0.05)
            ep = 0
            while viewer.is_running():
                res = ch.run(seed=int(time.time()) % 100000 + ep, swim=swim, on_step=show); ep += 1
                print(res); time.sleep(1.0)
    else:
        R = [ch.run(seed=s, swim=swim) for s in range(50)]
        n = len(R)
        for st in ["reach", "grasp", "drag"]:
            print(f"{st:6s} 成功 {sum(r.get(st, False) for r in R)}/{n}")
        ok = [r for r in R if r.get("drag")]
        print(f"完整链路成功 {len(ok)}/{n} = {len(ok)/n:.0%}")
        if ok:
            f = lambda k: np.median([r[k] for r in ok])
            print(f"成功回合：总用时中位 {f('total_s'):.1f} s（不含游进来），终点位置误差中位 {f('pos_err_mm'):.0f} mm，"
                  f"朝向误差中位 {np.median([abs(r['yaw_err_deg']) for r in ok]):.1f}°，松手后 3 s 漂移中位 {f('drift_3s_mm'):.0f} mm")
        from collections import Counter
        print("失败在哪个阶段：", dict(Counter(r.get("failed_at") for r in R if "failed_at" in r)))
