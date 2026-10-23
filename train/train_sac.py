"""用 SB3 的 SAC 训练阶段 1（reach）、阶段 2（grasp）或阶段 3（drag）。

运行（项目根目录，激活 .venv）：
    python train/train_sac.py                          # 默认：reach，5 万步，种子 0
    python train/train_sac.py 50000 1                  # 步数 50000，种子 1
    python train/train_sac.py 200000 0 grasp 20000     # 第 3 个参数选环境：reach / grasp / drag / grasp_res / drag_res /
                                                       #   icpt_kf / icpt_hist（拦截，A / B 方案）
                                                       # 第 4 个参数：先放进回放池的示范步数（默认 0 = 纯 RL）
示范数据（10-22 加）：阶段 2 纯 SAC 训练 9.5 万步成功率一直是 0 %（夹取要"对准 → 慢慢下降 → 合拢"一连串动作，
随机探索碰不上；乱动碰船又被重罚，于是学成"别靠近"）。做法：训练前用脚本策略跑 N 步，把这些转移 (s, a, r, s', done)
直接放进 SAC 的回放池。SAC 是 off-policy，能从"别人"产生的数据里学；之后靠自己的探索超过示范。
其中一半回合给示范动作加高斯噪声（σ = 0.3），让数据覆盖"做错一点"的情况。
输出（results/stage1_reach/ | stage2_grasp/ | stage3_drag/ 下的 sac_seed<种子>[_demo<N>]/）：
    monitor.monitor.csv  每个训练回合一行：回报 r、步数 l、时间 t、是否成功 success
    evaluations.npz      每 5000 步用确定性策略评估 20 回合的结果（EvalCallback 写的）
    best_model.zip       评估回报最高时的模型
    ckpt_*.zip           每 1 万步存一次的检查点（april 蓝屏过，断了能接着用）
    final_model.zip      训练结束时的模型
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from envs.reset_env import ReachEnv, scripted_action
from envs.grasp_env import GraspEnv, scripted_grasp
from envs.drag_env import DragEnv, scripted_drag
from envs.residual import ResidualWrapper
from envs.intercept_env import InterceptEnv, scripted_intercept
from stable_baselines3.common.callbacks import BaseCallback

V_MAX = 0.20                         # 拦截任务：船初速度上限 20 cm/s（BODY2 实测巡航约 0.2 m/s）

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TASK = sys.argv[3] if len(sys.argv) > 3 else "reach"
N_DEMO = int(sys.argv[4]) if len(sys.argv) > 4 else 0
Env, scripted = {"reach": (ReachEnv, scripted_action), "grasp": (GraspEnv, scripted_grasp),
                 "drag": (DragEnv, scripted_drag),
                 # 残差版（envs/residual.py）：动作 = 脚本策略 + 0.3 × RL；不需要示范
                 "grasp_res": (lambda: ResidualWrapper(GraspEnv(), scripted_grasp), None),
                 "drag_res": (lambda: ResidualWrapper(DragEnv(), scripted_drag), None),
                 # 残差 + 一半回合从阶段 1 的 SAC 的真实终点开始（技能接力，见 envs/grasp_env.py 的 stage1_states）
                 "grasp_res_chain": (lambda: ResidualWrapper(GraspEnv(stage1_states="results/stage2_grasp/start_states_reach_sac.npz"),
                                                            scripted_grasp), None),
                 # 拦截（envs/intercept_env.py）：A = 卡尔曼滤波预测进观测；B = 只给最近 4 帧原始测量，网络自己学
                 "icpt_kf": (lambda: InterceptEnv(predictor="kf", v_range=(0.0, V_MAX)), scripted_intercept),
                 "icpt_hist": (lambda: InterceptEnv(predictor="history", v_range=(0.0, V_MAX)), scripted_intercept)}[TASK]
OUT = ("results/" + {"reach": "stage1_reach", "grasp": "stage2_grasp", "drag": "stage3_drag",
                     "grasp_res": "stage2_grasp", "drag_res": "stage3_drag", "grasp_res_chain": "stage2_grasp",
                     "icpt_kf": "stage5_intercept/kf", "icpt_hist": "stage5_intercept/history"}[TASK]
       + ("/residual_chain" if TASK.endswith("_chain") else "/residual" if TASK.endswith("_res") else "") + f"/sac_seed{SEED}" + (f"_demo{N_DEMO}" if N_DEMO else ""))
os.makedirs(OUT, exist_ok=True)
import json
json.dump(dict(task=TASK, seed=SEED, total=TOTAL, demo=N_DEMO, start=time.time()),   # 给 train/progress.py 看进度用
          open(os.path.join(OUT, "run_info.json"), "w"))
torch.set_num_threads(4)             # 网络很小，线程开多了反而在调度上浪费时间

# 训练用的环境：Monitor 包装器把每个回合的回报、长度和 info 里的 success 记进 monitor.csv
train_env = Monitor(Env(), os.path.join(OUT, "monitor"), info_keywords=("success",))
# 评估用的环境：单独一份，评估时不加探索噪声（deterministic），看策略"真实水平"
eval_env = Monitor(Env())

model = SAC(
    "MlpPolicy",                     # 策略网络和 Q 网络都是多层感知机 (MLP)：全连接层 + ReLU
    train_env,
    learning_rate=3e-4,              # 学习率：每次梯度更新，权重沿梯度走多大一步（Adam 优化器）
    buffer_size=300_000,             # 回放池最多存 30 万条转移 (s, a, r, s', done)，满了挤掉最旧的
    learning_starts=1_000,           # 前 1000 步只用随机动作收集数据，不更新网络（池子里先要有东西）
    batch_size=256,                  # 每次更新从池子里随机抽 256 条
    tau=0.005,                       # 目标 Q 网络的软更新系数：目标网络 = 0.995·旧 + 0.005·新，让学习目标稳定
    gamma=0.99,                      # 折扣因子 γ：未来奖励打折，约等于"往后看 100 步"
    train_freq=1,                    # 每走 1 步……
    gradient_steps=1,                # ……就做 1 次梯度更新
    ent_coef="auto",                 # 熵系数 α 自动调：让策略的随机程度保持在目标值附近
    policy_kwargs=dict(net_arch=[256, 256]),   # 两个隐藏层，各 256 个神经元（actor 和两个 critic 各一套）
    seed=SEED,
    verbose=0,
)

# ---------- 示范数据：脚本策略的转移直接放进回放池 ----------
if N_DEMO:
    denv, rng, n, ep, n_succ = Env(), np.random.default_rng(SEED + 12345), 0, 0, 0
    while n < N_DEMO:
        obs, _ = denv.reset(seed=100_000 + ep)
        noisy = ep % 2 == 1                                   # 奇数回合加噪声
        ep += 1
        while True:
            a = scripted(denv.kf_obs()) if TASK == "icpt_hist" else scripted(obs)   # B：脚本看后台 KF 的观测
            if noisy:
                a = np.clip(a + rng.normal(0, 0.3, a.shape), -1, 1)
            nobs, r, term, trunc, info = denv.step(a)
            # SB3 的回放池要求形状 (环境数=1, ...)；回合是被时间截断时要在 infos 里标出来，算 Q 值时它就不会当成"终止"
            model.replay_buffer.add(obs[None], nobs[None], a[None].astype(np.float32), np.array([r]),
                                    np.array([term or trunc]), [{"TimeLimit.truncated": trunc and not term}])
            obs, n = nobs, n + 1
            if term or trunc:
                n_succ += info["success"]
                break
    print(f"示范：{n} 步、{ep} 回合放进回放池（其中一半加噪声），示范回合成功率 {n_succ/ep:.0%}")

class SpeedCurriculum(BaseCallback):
    """课程学习（curriculum）：训练环境里船的最高初速度从 3 cm/s 线性升到 V_MAX，用前 60 % 的步数；
    评估环境一直用完整的 0–V_MAX，看到的是"真实水平"。"""
    def _on_step(self):
        frac = min(self.num_timesteps / (0.6 * TOTAL), 1.0)
        train_env.env.v_range = (0.0, 0.03 + frac * (V_MAX - 0.03))
        return True


callbacks = CallbackList([
    *([SpeedCurriculum()] if TASK.startswith("icpt") else []),
    EvalCallback(eval_env, eval_freq=5_000, n_eval_episodes=20, deterministic=True,
                 best_model_save_path=OUT, log_path=OUT, verbose=1),
    CheckpointCallback(save_freq=10_000, save_path=OUT, name_prefix="ckpt"),
])

t0 = time.time()
model.learn(total_timesteps=TOTAL, callback=callbacks, progress_bar=False)
model.save(os.path.join(OUT, "final_model"))
dt = time.time() - t0
print(f"训练完成：{TOTAL} 步，用时 {dt/60:.1f} 分钟（{TOTAL/dt:.0f} 步/秒），结果在 {OUT}/")

if TASK.startswith("icpt"):                           # 拦截：按船速分档评估，每档 30 回合，和脚本拦截器同一批种子
    env = Env()
    for vr in [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]:
        env.v_range = vr; S, N, H = [], [], []
        for ep in range(30):
            obs, _ = env.reset(seed=ep)
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            S.append(info["success"]); N.append(env.steps); H.append(info["wall_hits"])
        S = np.array(S)
        print(f"船速 {100*vr[0]:.0f}–{100*vr[1]:.0f} cm/s：成功 {S.sum()}/30 | 成功时步数 中位 "
              f"{np.median(np.array(N)[S]) if S.any() else float('nan'):.0f} | 平均撞墙 {np.mean(H):.2f} 次")
    sys.exit(0)

# 训练完立刻用确定性策略评估 50 回合（和 tests/ 里随机 / 脚本策略同一批种子，方便对比）
env = Env()
succ, steps, coll, final = [], [], [], []
for ep in range(50):
    obs, _ = env.reset(seed=ep)
    hit = False
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        hit |= info["collided"]
        if term or trunc:
            break
    succ.append(info["success"]); steps.append(env.steps); coll.append(hit); final.append(info["dist"])
succ = np.array(succ)
print(f"评估 50 回合：成功 {succ.mean():.0%} | 成功时步数 中位 {np.median(np.array(steps)[succ]) if succ.any() else float('nan'):.0f}"
      f" | 碰撞回合 {np.mean(coll):.0%} | 结束距离 中位 {1000*np.median(final):.0f} mm")
