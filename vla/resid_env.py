"""残差强化学习的环境：冻结的 SmolVLA 给出动作，一个小网络（SAC）在它上面加一个小的修正量。

每个控制周期：
  1. 两路相机 + 关节角度 → SmolVLA（TensorRT 引擎，权重始终不变）→ 未来 50 步动作，取第 1 步 a_vla
  2. 小网络看到“观测”→ 输出修正量 δ（每个分量在 -1 到 1 之间）
  3. 执行  a = clip(a_vla + alpha × δ, -1, 1)
  4. 环境给出奖励（就是当初训练 SAC 教师用的那个奖励），只有小网络根据它更新

小网络的观测（66 个数），全部来自相机和关节，不用船的真实位置：
  关节角度 6 + SmolVLA 预测的前 10 步动作 50 + 上一步实际执行的动作 5 + 上一步 SmolVLA 的动作 5
SmolVLA 预测的动作序列里隐含了它从图像里看到的船的位置和去向，所以把它当作“压缩过的图像信息”交给小网络。
"""
import numpy as np, gymnasium as gym

TASK = "intercept the drifting boat and grasp its mast"
K1, K2 = "observation.images.camera1", "observation.images.camera2"


class ResidEnv(gym.Env):
    def __init__(self, model_dir, engine, alpha=0.3, fast_bias=0.5, v_range=None, pen=0.02):
        import torch, tensorrt as trt
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        from common import make_env, Cams, joints
        self.torch, self.joints = torch, joints
        self.LT, self.LM = OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        p = SmolVLAPolicy.from_pretrained(model_dir)               # 只为了拿配置和前后处理，权重留在 CPU 上不参与计算
        self.chunk, self.adim = p.config.chunk_size, p.config.max_action_dim
        self.pre, self.post = make_pre_post_processors(p.config, pretrained_path=model_dir)
        del p
        self._rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._eng = self._rt.deserialize_cuda_engine(open(engine, "rb").read())
        self._ctx = self._eng.create_execution_context()
        self._names = [self._eng.get_tensor_name(i) for i in range(self._eng.num_io_tensors)]
        self._DT = {trt.DataType.FLOAT: torch.float32, trt.DataType.INT64: torch.int64, trt.DataType.INT32: torch.int32,
                    trt.DataType.BOOL: torch.bool}
        self._out = torch.empty(1, self.chunk, 5, device="cuda")
        self.env = make_env(); self.cams = Cams(self.env)
        self.alpha, self.fast_bias, self.fixed_v, self.pen = alpha, fast_bias, v_range, pen
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (66,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.rng = np.random.default_rng(0)

    def _vla(self):
        torch = self.torch
        top, wrist = self.cams(); q = self.joints(self.env)
        f = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)
        b = self.pre({K1: f(top), K2: f(wrist), "observation.state": torch.from_numpy(q).unsqueeze(0), "task": [TASK]})
        src = {"cam1": b[K1], "cam2": b[K2], "state": b["observation.state"], "tokens": b[self.LT], "mask": b[self.LM],
               "noise": torch.randn(1, self.chunk, self.adim, device="cuda")}
        keep = []
        for n in self._names:
            if n == "actions":
                self._ctx.set_tensor_address(n, self._out.data_ptr()); continue
            t = src[n].to("cuda", self._DT[self._eng.get_tensor_dtype(n)]).contiguous(); keep.append(t)
            self._ctx.set_tensor_address(n, t.data_ptr())
        self._ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
        acts = self.post(self._out.clone())[0].float().cpu().numpy()          # (50, 5)，已还原成真实动作
        return q.astype(np.float32), acts

    def _obs(self):
        q, acts = self._vla()
        self.a_vla = acts[0].astype(np.float32)
        return np.concatenate([q, acts[:10].ravel(), self.prev_a, self.prev_vla]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed_v is not None:
            self.env.v_range = self.fixed_v
        else:                                                               # 训练时一半回合专门抽快船
            self.env.v_range = (0.10, 0.20) if self.rng.random() < self.fast_bias else (0.0, 0.20)
        self.env.reset(seed=seed if seed is not None else int(self.rng.integers(1 << 30)))
        self.prev_a = np.zeros(5, np.float32); self.prev_vla = np.zeros(5, np.float32)
        return self._obs(), {}

    def step(self, delta):
        delta = np.asarray(delta, np.float32)
        a = np.clip(self.a_vla + self.alpha * delta, -1.0, 1.0).astype(np.float32)
        _, r, term, trunc, info = self.env.step(a)
        r = float(r) - self.pen * float(np.square(delta).sum())             # 修正量本身有一点代价：没必要就别改
        self.prev_a, self.prev_vla = a, self.a_vla
        info = {"success": bool(info["success"]), "wall_hits": info["wall_hits"], "da2": info["da2"], "steps": self.env.steps}
        obs = self._obs() if not (term or trunc) else np.zeros(66, np.float32)
        return obs, r, bool(term), bool(trunc), info
