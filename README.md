# reset_station：BODY2 复位台（MuJoCo + 强化学习）

机械臂 SO-101 把漂在水槽里的两栖四足机器人 BODY2 夹住、拖回中心、调头。先全部在 MuJoCo 里做，用强化学习训练；以后接真机。

## 场景（2026-10-20）

- 水槽 1 m × 0.5 m，水深 0.12 m；SO-101 立着装在后墙内侧的平台上，底座比水面高 4 cm。
- 隔板（x = 0.08 m）把机械臂区和船区分开，船进不到平台和底座下面。
- BODY2 先用盒子代替：船体 0.262 × 0.116 × 0.05 m、0.5 kg，中央一根扁桅杆（3 × 1 × 6 cm）给夹爪从上往下夹；转 `wrist_roll` 可让船原地调头。
- 机械臂只能在离底座约 0.06–0.37 m 内竖直朝下夹桅杆（`docs/reach_map.png`），所以复位分两步：BODY2 自己游进可抓区 → 机械臂夹住、定位、调头。
- 水：`envs/water.py`，体素浮力 + 线性/二次阻力（低保真，系数是先验）。

## 运行

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install mujoco gymnasium stable-baselines3
python view_scene.py 0
```

WSL 里 viewer 很卡时：`export GALLIUM_DRIVER=d3d12`（让 OpenGL 走显卡）。

## 文件

| 路径 | 内容 |
|---|---|
| `assets/robotstudio_so101/reset_station.xml` | 场景：水槽、平台、隔板、BODY2 替身、home 初始姿态 |
| `assets/robotstudio_so101/so101.xml` 等 | SO-101 模型，来自 MuJoCo Menagerie（Apache 2.0，见同目录 LICENSE） |
| `envs/water.py` | 浮力和阻力 |
| `view_scene.py` | 查看场景：BODY2 随机出发、朝可抓区游进来 |
| `docs/reach_map*.py`、`docs/reach_map.png` | 有效抓取范围的采样和画图 |

## 阶段

1. Reach：夹爪移到桅杆正上方 ← 下一步
2. Grasp：夹住桅杆
3. 拖到中心 + 调头
4. 完整水动力、域随机化、相机 + AprilTag
5. 模仿学习 / 残差 RL / ACT、SmolVLA 对比
