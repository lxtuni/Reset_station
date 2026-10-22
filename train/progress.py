"""文字版训练进度：进度条、已用时间、速度、预计剩余时间、最近一次评估的成功率。

    python train/progress.py                 # 列出所有训练（运行中的在前）
    python train/progress.py watch           # 每 10 秒刷新一次，Ctrl+C 退出
    python train/progress.py results/stage2_grasp/sac_seed0_demo30000   # 只看某一个

数据来源：train_sac.py 开始时写的 run_info.json（总步数、开始时间），Monitor 写的 monitor.monitor.csv（已完成的回合），
EvalCallback 写的 evaluations.npz（评估）。有 final_model.zip 就算"已结束"。
已完成步数按"已结束的回合"累计，所以会比真实步数落后最多一个回合（≤ 250 步）。
"""
import csv, glob, json, os, sys, time
from datetime import datetime, timedelta

import numpy as np


def fmt(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def status(run):
    info_path = os.path.join(run, "run_info.json")
    if not os.path.exists(info_path):
        return None
    info = json.load(open(info_path))
    total, t0 = info["total"], info["start"]
    done_flag = os.path.exists(os.path.join(run, "final_model.zip"))
    steps = 0
    mon = os.path.join(run, "monitor.monitor.csv")
    if os.path.exists(mon):
        with open(mon) as f:
            next(f)
            steps = sum(int(r["l"]) for r in csv.DictReader(f))
    demo = info.get("demo", 0)
    now = time.time()
    elapsed = (os.path.getmtime(os.path.join(run, "final_model.zip")) if done_flag else now) - t0
    learn_steps = max(steps, 1)
    speed = learn_steps / max(elapsed, 1)
    frac = 1.0 if done_flag else min(steps / total, 0.999)
    eta = 0 if done_flag else (total - steps) / max(speed, 1e-6)
    lines = [f"{os.path.relpath(run, 'results')}   {'已结束' if done_flag else '运行中'}"
             + (f"   （先放了 {demo} 步示范）" if demo else "")]
    n = int(frac * 20)
    lines.append(f"[{'#' * n}{'-' * (20 - n)}] {100 * frac:3.0f}%   {steps:,} / {total:,} 步")
    if done_flag:
        lines.append(f"总用时 {fmt(elapsed)}   速度 {speed:.0f} 步/秒")
    else:
        finish = datetime.now() + timedelta(seconds=eta)
        lines.append(f"已用 {fmt(elapsed)}   速度 {speed:.0f} 步/秒   预计还要 {fmt(eta)}（约 {finish:%H:%M} 完成）")
    ev = os.path.join(run, "evaluations.npz")
    if os.path.exists(ev):
        try:
            e = np.load(ev)
            if e["successes"].size:
                s = e["successes"].mean(1); t = e["timesteps"]; i = int(np.argmax(s))
                lines.append(f"最近评估：成功 {100 * s[-1]:.0f} %（{t[-1] / 1e4:g} 万步）   最高：{100 * s[i]:.0f} %（{t[i] / 1e4:g} 万步）")
        except Exception:
            pass                                               # 训练正在写这个文件时可能读到一半
    return (not done_flag, t0, "\n".join(lines))


def show(runs):
    rows = [r for r in (status(x) for x in runs) if r]
    if not rows:
        return "没有找到训练记录（results/*/…/run_info.json）"
    rows.sort(key=lambda r: (not r[0], -r[1]))               # 运行中的在前，新的在前
    return "\n\n".join(r[2] for r in rows)


def report(run, every):
    """单行汇报（给对话里的进度提示用）：每 every 秒打印一行；结束、或日志超过 5 分钟没更新（可能崩了）时打印最后一行并退出。"""
    mon = os.path.join(run, "monitor.monitor.csv")
    while True:
        st = status(run)
        if st:
            running, _, text = st
            lines = text.split("\n")
            print(" | ".join([lines[1]] + lines[2:]), flush=True)
            if not running:
                return
            if os.path.exists(mon) and time.time() - os.path.getmtime(mon) > 300:
                print("⚠ 训练日志 5 分钟没有更新，训练可能已经停止或出错", flush=True)
                return
        time.sleep(every)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":       # python train/progress.py report <run> [秒]
        report(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
        sys.exit()
    args = [a for a in sys.argv[1:] if a != "watch"]
    runs = args or sorted(set(os.path.dirname(p) for p in glob.glob("results/**/run_info.json", recursive=True)
                              if "_veraltet" not in p))
    if "watch" in sys.argv:
        try:
            while True:
                os.system("clear")
                print(f"{datetime.now():%H:%M:%S}  （每 10 秒刷新，Ctrl+C 退出）\n")
                print(show(runs))
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        print(show(runs))
