"""方案 B 的训练入口：和 lerobot-train 完全一样，只改一处——按权重抽取训练样本。

优势加权回归的损失是  Σ w_t × (流匹配损失_t)。“按 w_t 的比例抽样，再算普通的损失”和它在期望上相同，
所以不用改 SmolVLA 的损失函数，只要把数据加载器换成按权重抽样的就行。
权重文件由 convert_vla.py 生成（数据集目录下的 awr_weights.npy，每一帧一个数）。
用法：AWR_WEIGHTS=data/vla_b1/awr_weights.npy python vla/train_awr.py <和 lerobot-train 相同的参数>
"""
import os
import numpy as np, torch

W = torch.from_numpy(np.load(os.environ["AWR_WEIGHTS"])).double()
_DL = torch.utils.data.DataLoader


class WeightedLoader(_DL):
    used = False

    def __init__(self, dataset, *a, **kw):
        if not WeightedLoader.used:                           # 第一个创建的就是训练用的加载器
            assert len(dataset) == len(W), f"权重 {len(W)} 个，数据集 {len(dataset)} 帧，对不上"
            kw["sampler"] = torch.utils.data.WeightedRandomSampler(W, num_samples=len(W), replacement=True)
            kw["shuffle"] = False
            WeightedLoader.used = True
            print(f"已启用按权重抽样：{len(W)} 帧，权重最小 {W.min():.3f}，最大 {W.max():.3f}", flush=True)
        super().__init__(dataset, *a, **kw)


if __name__ == "__main__":        # 必须有这一行：读数据的子进程启动时会重新导入本文件，没有它子进程会再跑一遍训练而崩溃
    from lerobot.scripts import lerobot_train
    torch.utils.data.DataLoader = WeightedLoader
    lerobot_train.main()
    assert WeightedLoader.used
