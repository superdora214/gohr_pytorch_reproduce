# -*- coding: utf-8 -*-
"""
训练日志可视化：把 train.py 的日志文件画成曲线图
用法：
    python plot_log.py                              # 默认读 train_6r_full.log
    python plot_log.py train_7r.log                 # 读指定日志
    python plot_log.py train_6r_full.log out.png    # 指定输出图片
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存文件
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def find_latest_log():
    """不传参数时，自动找目录下最新的 train_*.log"""
    files = glob.glob("train_*.log")
    return max(files, key=os.path.getmtime) if files else None


LOG = sys.argv[1] if len(sys.argv) > 1 else (find_latest_log() or "train_6r_full.log")
OUT = sys.argv[2] if len(sys.argv) > 2 else "training_curves.png"
print(f"读取日志: {LOG}")

if not os.path.exists(LOG):
    sys.exit(f"找不到日志文件：{LOG}（请在仓库根目录运行，或用参数指定路径）")

epochs, losses, accs, lrs = [], [], [], []
pat = re.compile(r"epoch (\d+)/(\d+) \| loss ([\d.]+) \| val_acc ([\d.]+) \| lr ([\d.]+)")
with open(LOG, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(3)))
            accs.append(float(m.group(4)))
            lrs.append(float(m.group(5)))

if not epochs:
    sys.exit("日志里还没有任何 epoch 记录，等训练跑完第一个 epoch 再试。")

best_i = accs.index(max(accs))
print(f"已解析 {len(epochs)} 个 epoch")
print(f"当前最佳验证准确率 = {max(accs):.4f}（第 {epochs[best_i]} 个 epoch）")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

# ① 验证准确率（最重要的图）
ax = axes[0]
ax.plot(epochs, accs, color="#c0392b", lw=2, marker="o", ms=3, label="val_acc")
ax.axhline(0.72, color="#7f8c8d", ls="--", lw=1.2, label="论文基准 0.72")
ax.axhline(0.5, color="#bdc3c7", ls=":", lw=1.2, label="随机猜 0.50")
ax.scatter([epochs[best_i]], [accs[best_i]], color="#c0392b", s=70, zorder=5)
ax.annotate(f"最佳 {max(accs):.4f}", (epochs[best_i], accs[best_i]),
            textcoords="offset points", xytext=(8, -12), color="#c0392b", fontsize=10)
ax.set_xlabel("epoch"); ax.set_ylabel("验证准确率")
ax.set_title("区分能力（越接近 1 越好）")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# ② 损失
ax = axes[1]
ax.plot(epochs, losses, color="#2471a3", lw=2, marker="o", ms=3)
ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE)")
ax.set_title("训练损失（越低越好）")
ax.grid(alpha=0.3)

# ③ 循环学习率
ax = axes[2]
ax.plot(epochs, lrs, color="#27ae60", lw=1.5)
ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
ax.set_title("循环学习率（每10轮一个周期）")
ax.grid(alpha=0.3)

fig.suptitle(f"{LOG}　—　共 {len(epochs)} 个 epoch", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=130, facecolor="white")

# 转为 RGB：matplotlib 默认存成 RGBA（带透明通道），部分 Windows 看图工具打不开
try:
    from PIL import Image
    _im = Image.open(OUT)
    if _im.mode == "RGBA":
        _bg = Image.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.split()[3])
        _bg.save(OUT)
except Exception:
    pass

print(f"曲线图已保存：{os.path.abspath(OUT)}")
