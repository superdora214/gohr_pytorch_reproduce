# -*- coding: utf-8 -*-
"""
生成"修复前后对比图"：同一 6 轮任务，输入表示 bug 修复前后的验证准确率对比。

用法：python plot_comparison.py
输出：本目录下的 comparison.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
BUGGY_CSV = os.path.join(HERE, "history_buggy.csv")
FIXED_CSV = os.path.join(HERE, "..", "..", "..", "checkpoints", "history_6r.csv")
OUT = os.path.join(HERE, "comparison.png")

PAPER_N6 = 0.788


def load(path):
    epochs, accs = [], []
    if not os.path.exists(path):
        return epochs, accs
    with open(path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(row["epoch"]))
                accs.append(float(row["val_acc"]))
            except (KeyError, ValueError):
                continue
    return epochs, accs


def main():
    eb, ab = load(BUGGY_CSV)
    ef, af = load(FIXED_CSV)

    if not ab and not af:
        raise SystemExit("找不到数据文件（history_buggy.csv / checkpoints/history_6r.csv）")

    fig, ax = plt.subplots(figsize=(11, 5.2))

    if ab:
        best_b = max(ab)
        ax.plot(eb, ab, color="#c0392b", lw=1.8,
                label=f"修复前（输入表示错误）最佳 {best_b:.4f}")
        ax.axhline(best_b, color="#c0392b", ls=":", lw=1, alpha=0.6)
    if af:
        best_f = max(af)
        ax.plot(ef, af, color="#27ae60", lw=1.8,
                label=f"修复后（面向字表示）最佳 {best_f:.4f}")

    ax.axhline(PAPER_N6, color="#2980b9", ls="--", lw=1.5,
               label=f"论文 N6 = {PAPER_N6}")
    ax.axhline(0.5, color="#bdc3c7", ls="-.", lw=1, label="随机猜测 0.50")

    # 标注修复点
    if ef:
        ax.axvline(ef[0], color="#f39c12", ls="--", lw=1.2, alpha=0.8)
        ax.annotate("修复输入表示\n（代码改 1 行）",
                    (ef[0], 0.62), fontsize=10, color="#d35400",
                    ha="left", va="bottom",
                    xytext=(8, 0), textcoords="offset points")

    ax.set_xlabel("epoch（训练轮次）", fontsize=11)
    ax.set_ylabel("验证准确率 val_acc", fontsize=11)
    ax.set_title("输入表示 bug 修复前后对比（6 轮 SPECK32/64 神经区分器）\n"
                 "同一个任务、同一套网络结构，仅修改输入张量的排列方式",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="center right")
    ax.set_ylim(0.48, 0.82)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, facecolor="white")

    try:
        from PIL import Image
        _im = Image.open(OUT)
        if _im.mode == "RGBA":
            _bg = Image.new("RGB", _im.size, (255, 255, 255))
            _bg.paste(_im, mask=_im.split()[3])
            _bg.save(OUT)
    except Exception:
        pass

    print(f"对比图已保存：{OUT}")
    if ab and af:
        print(f"修复前最佳 {max(ab):.4f}（{len(ab)} epoch） → 修复后最佳 {max(af):.4f}（{len(af)} epoch）")
        print(f"提升 {max(af) - max(ab):+.4f}")


if __name__ == "__main__":
    main()
