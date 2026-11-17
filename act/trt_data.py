"""给 polygraphy 提供真实输入（不用它默认的随机数）。函数名 load_data 是 polygraphy 规定的。"""
import numpy as np


def load_data():
    d = np.load("sample.npz")
    yield {k: d[k] for k in d.files}                  # 输入名和 npz 里的键一致
