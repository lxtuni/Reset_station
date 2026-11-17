"""ACT 接入复位台：采集、回放、评测共用的部分。所有脚本都从仓库根目录运行：python act/xxx.py"""
import sys
sys.path.insert(0, ".")
import numpy as np, mujoco
from envs.intercept_env import InterceptEnv
from envs.kalman import BoatKF

TEACHER = "results/stage9_smooth/kf_occl/sac_seed0_demo30000/smooth_model.zip"   # stage9 的 SAC + 卡尔曼滤波
H, W = 240, 320                                   # 每路相机的图像尺寸（高，宽）
BINS = [(0.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20)]   # 和 eval/intercept_ab.py 一样


def make_env(v_range=(0.0, 0.20)):
    """和 eval/intercept_ab.py 评测 stage9 kf_occl 模型（--kf-tuned）时完全一样的环境设置。"""
    env = InterceptEnv(predictor="kf", v_range=v_range, occlusion=True, wrist=False, two_tags=True, action_filter=0.8)
    env.block_outward = True
    env.sigma_pos, env.sigma_yaw, env.latency = 0.001, np.radians(0.5), 1
    env.kf = BoatKF(sigma_pos=0.001, sigma_yaw=env.sigma_yaw, gate=0.006)
    env.kf.reset(np.zeros(2), 0.0)
    return env


class Cams:
    """渲染 ACT 看到的两路图像：俯视相机 top_cam 和腕部相机 wrist_cam，各 (240, 320, 3) uint8。"""
    def __init__(self, env):
        self.env = env
        self.ren = mujoco.Renderer(env.m, H, W)

    def __call__(self):
        self.ren.update_scene(self.env.d, camera="top_cam"); top = self.ren.render().copy()
        self.ren.update_scene(self.env.d, camera="wrist_cam"); wrist = self.ren.render().copy()
        return top, wrist


def joints(env):
    """6 个关节角度（5 个臂关节 + 夹爪），ACT 的状态输入。"""
    return env.d.qpos[env.arm_qa].astype(np.float32).copy()
