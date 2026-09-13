# -*- coding: utf-8 -*-
"""
训练曲线可视化（支持 CSV 和日志文件）
用法：
    python plot_log.py                                   # 自动找最新的 history_*.csv / train_*.log
    python plot_log.py checkpoints/history_6r.csv        # 指定 CSV
    python plot_log.py train_6r.log out.png              # 指定日志与输出名

论文基准取自 Gohr 2019 表 2 的神经网络目标（N5-N8）。
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存文件
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Gohr 2019 表 2：各轮数的神经网络准确率目标（经典 DDT 基准另见 README）
PAPER_BASELINE = {5: 0.929, 6: 0.788, 7: 0.616, 8: 0.514}
# 经典差分分布表（DDT）基准，用于对比参照
DDT_BASELINE = {5: 0.911, 6: 0.758, 7: 0.591, 8: 0.512}


def find_input():
    """不传参数时，自动找最新的 history_*.csv，其次是最新的 train_*.log"""
    csvs = glob.glob("checkpoints/history_*.csv") + glob.glob("history_*.csv")
    if csvs:
        return max(csvs, key=os.path.getmtime)
    logs = glob.glob("train_*.log")
    return max(logs, key=os.path.getmtime) if logs else None


def parse_csv(path):
    """解析训练输出的 CSV（列：epoch,loss,val_acc,lr）"""
    epochs, losses, accs, lrs = [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["loss"]))
                accs.append(float(row["val_acc"]))
                lrs.append(float(row["lr"]))
            except (KeyError, ValueError):
                continue
    return epochs, losses, accs, lrs


def parse_log(path):
    """解析训练日志（逐行正则）"""
    epochs, losses, accs, lrs = [], [], [], []
    pat = re.compile(r"epoch (\d+)/\d+ \| loss ([\d.]+) \| val_acc ([\d.]+) \| lr ([\d.]+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                accs.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    return epochs, losses, accs, lrs


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else find_input()
    if not src:
        sys.exit("找不到训练数据（既没有 checkpoints/history_*.csv，也没有 train_*.log）")
    out = sys.argv[2] if len(sys.argv) > 2 else "training_curves.png"

    print(f"读取: {src}")
    if src.endswith(".csv"):
        epochs, losses, accs, lrs = parse_csv(src)
        rounds = re.search(r"(\d+)r", os.path.basename(src))
        rounds = int(rounds.group(1)) if rounds else None
    else:
        epochs, losses, accs, lrs = parse_log(src)
        rounds = None

    if not epochs:
        sys.exit("文件里没有可用的 epoch 记录。")

    best_i = accs.index(max(accs))
    print(f"共 {len(epochs)} 个 epoch | 最佳 val_acc = {max(accs):.6f} (epoch {epochs[best_i]})")
    if rounds and rounds in PAPER_BASELINE:
        print(f"论文目标（{rounds} 轮）= {PAPER_BASELINE[rounds]} | 差距 = {max(accs) - PAPER_BASELINE[rounds]:+.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # ① 验证准确率（最重要）
    ax = axes[0]
    ax.plot(epochs, accs, color="#c0392b", lw=1.6, label="本复现 val_acc")
    if rounds and rounds in PAPER_BASELINE:
        ax.axhline(PAPER_BASELINE[rounds], color="#2980b9", ls="--", lw=1.4,
                   label=f"论文 N{rounds} = {PAPER_BASELINE[rounds]}")
    if rounds and rounds in DDT_BASELINE:
        ax.axhline(DDT_BASELINE[rounds], color="#7f8c8d", ls=":", lw=1.2,
                   label=f"经典 DDT 基准 = {DDT_BASELINE[rounds]}")
    ax.axhline(0.5, color="#bdc3c7", ls="-.", lw=1.0, label="随机猜测 0.50")
    ax.scatter([epochs[best_i]], [accs[best_i]], color="#c0392b", s=90, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"最佳 {max(accs):.4f}\n(epoch {epochs[best_i]})",
                (epochs[best_i], accs[best_i]),
                textcoords="offset points", xytext=(10, -30), color="#c0392b", fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("验证准确率")
    ax.set_title(f"区分能力（{rounds} 轮 SPECK32/64）" if rounds else "区分能力")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")

    # ② 损失
    ax = axes[1]
    ax.plot(epochs, losses, color="#2471a3", lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE)")
    ax.set_title("训练损失（越低越好）")
    ax.grid(alpha=0.3)

    # ③ 循环学习率
    ax = axes[2]
    ax.plot(epochs, lrs, color="#27ae60", lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("循环学习率（每 10 epoch 一个周期）")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{os.path.basename(src)} — 共 {len(epochs)} 个 epoch，最佳 {max(accs):.4f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")

    # 转 RGB：matplotlib 默认存 RGBA，部分 Windows 看图工具打不开
    try:
        from PIL import Image
        _im = Image.open(out)
        if _im.mode == "RGBA":
            _bg = Image.new("RGB", _im.size, (255, 255, 255))
            _bg.paste(_im, mask=_im.split()[3])
            _bg.save(out)
    except Exception:
        pass

    print(f"曲线图已保存：{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
