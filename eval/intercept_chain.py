"""完整链路：拦截（追上、夹住、停稳）→ 拖动转向（拖到终点、掉头、卸力、松手、抬起）→ 松手后 3 s 看漂移。

    .venv/bin/python eval/intercept_chain.py [icpt=MODEL_DIR] [drag=scripted|MODEL_DIR] [n=30]
默认：拦截 = results/stage5_intercept/kf/sac_seed0_demo30000_v4，拖动 = scripted。每个船速档 n 回合（种子 0…n-1）。
两步之间传完整仿真状态（envs/state_io.py），拖动用新的手臂控制（DragEnv(smooth=True)）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.intercept_env import InterceptEnv
from envs.drag_env import DragEnv, scripted_drag, TARGET_XY, TARGET_YAW, wrap_pi
from envs.state_io import snapshot, restore

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ICPT = ARGS.get("icpt", "results/stage5_intercept/kf/sac_seed0_demo30000_v4")
DRAG = ARGS.get("drag", "scripted")
N = int(ARGS.get("n", 30))
BINS = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]

icpt_model = SAC.load(os.path.join(ICPT, "best_model.zip"))
drag_model = None if DRAG == "scripted" else SAC.load(os.path.join(DRAG, "best_model.zip"))
drag_pol = scripted_drag if drag_model is None else (lambda o: drag_model.predict(o, deterministic=True)[0])
ienv = InterceptEnv(predictor="history" if "history" in ICPT else "kf")
denv = DragEnv(start_states="results/stage6_intercept_drag/start_states_intercept.npz", smooth=True)


def episode(seed, vr, on_step=None):
    """一个完整回合。on_step(env, phase) 每个控制步调用一次（GIF 用）。返回结果 dict。"""
    ienv.v_range = vr
    obs, _ = ienv.reset(seed=seed)
    while True:
        obs, r, t, tr, info = ienv.step(icpt_model.predict(obs, deterministic=True)[0])
        if on_step:
            on_step(ienv, "intercept")
        if t or tr:
            break
    res = dict(v0=info["v0"], icpt=bool(info["success"]), t_icpt=ienv.steps * 0.05, walls=info["wall_hits"])
    if not res["icpt"]:
        return res
    # 接力：拦截的终点 = 拖动转向的起点（同 DragEnv.reset，只是不随机选起点）
    restore(denv, snapshot(ienv))
    denv.boat_xy0 = denv.d.xpos[denv.bid][:2].copy()
    denv.hold = denv.release = denv.steps = 0
    denv.prev_action = np.zeros(5)
    res["yaw0"] = float(np.degrees(abs(wrap_pi(denv._boat_yaw() - TARGET_YAW))))
    obs = denv._obs()
    while True:
        obs, r, t, tr, info = denv.step(drag_pol(obs))
        if on_step:
            on_step(denv, "drag")
        if t or tr:
            break
    res.update(drag=bool(info["success"]), t_drag=denv.steps * 0.05, pos_err=info["pos_err"], yaw_err=abs(info["yaw_err_deg"]))
    if res["drag"]:                                           # 松手后手臂不动，再看 3 s：船漂了多远
        p0 = denv.d.xpos[denv.bid][:2].copy()
        for _ in range(60):
            for _ in range(denv.n_sub):
                denv._physics_step()
            if on_step:
                on_step(denv, "after")
        res["drift"] = float(np.linalg.norm(denv.d.xpos[denv.bid][:2] - p0))
        res["final_err"] = float(np.linalg.norm(TARGET_XY - denv.d.xpos[denv.bid][:2]))
    return res


if __name__ == "__main__":
    print(f"拦截 = {ICPT}\n拖动 = {DRAG}")
    allr = []
    for vr in BINS:
        R = [episode(s, vr) for s in range(N)]
        allr += R
        ic = [r for r in R if r["icpt"]]; dr = [r for r in ic if r.get("drag")]
        print(f"船速 {100*vr[0]:.0f}–{100*vr[1]:.0f} cm/s：拦截 {len(ic)}/{N}，拖动转向 {len(dr)}/{len(ic)}，整条链路 {len(dr)}/{N} | "
              f"总用时中位 {np.median([r['t_icpt'] + r['t_drag'] for r in dr]) if dr else float('nan'):.1f} s | "
              f"要转的角度中位 {np.median([r['yaw0'] for r in ic]) if ic else float('nan'):.0f}° | "
              f"松手 3 s 后离终点中位 {100*np.median([r['final_err'] for r in dr]) if dr else float('nan'):.1f} cm", flush=True)
    dr = [r for r in allr if r.get("drag")]
    print(f"合计：整条链路 {len(dr)}/{len(allr)}（{len(dr)/len(allr):.0%}），松手后 3 s 漂移中位 {100*np.median([r['drift'] for r in dr]):.1f} cm、"
          f"最大 {100*max(r['drift'] for r in dr):.1f} cm")
