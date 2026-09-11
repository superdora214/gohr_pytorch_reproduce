# -*- coding: utf-8 -*-
"""
Gohr 神经区分器 —— 训练启动器（交互式菜单）
用法：双击同目录的「开始训练.bat」，或命令行 python 开始训练.py
"""
import os
import subprocess
import sys

PY = r"C:\Users\lmjkj\anaconda3\envs\pytorch-gpu\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))

MENU = [
    ("快速测试（3轮/10万样本/3epoch/约2分钟）", "--rounds 3 --train-size 100000 --eval-size 20000 --epochs 3 --depth 3"),
    ("训练 5 轮（论文级/1000万样本/200epoch）", "--rounds 5 --train-size 10000000 --eval-size 1000000 --epochs 200 --depth 10"),
    ("训练 6 轮（论文级/1000万样本/200epoch）", "--rounds 6 --train-size 10000000 --eval-size 1000000 --epochs 200 --depth 10"),
    ("训练 7 轮（论文级/1000万样本/200epoch）", "--rounds 7 --train-size 10000000 --eval-size 1000000 --epochs 200 --depth 10"),
    ("续训上一次（--resume）", None),
]


def main():
    os.chdir(HERE)
    print("=" * 62)
    print("  Gohr 神经区分器 —— 训练启动器")
    print("=" * 62)
    print()
    for i, (name, _) in enumerate(MENU, 1):
        print(f"  [{i}] {name}")
    print("  [0] 退出")
    print()
    choice = input("请输入选项数字并回车: ").strip()

    if choice == "0" or choice == "":
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(MENU)):
        print("输入无效。")
        return

    name, args = MENU[int(choice) - 1]

    if args is None:  # 续训
        print("\n请输入要续训的轮数（5/6/7）和总目标 epoch 数")
        rounds = input("轮数 [默认 6]: ").strip() or "6"
        epochs = input("总 epoch 数 [默认 200]: ").strip() or "200"
        args = f"--rounds {rounds} --train-size 10000000 --eval-size 1000000 --epochs {epochs} --depth 10 --resume"
        log = f"train_{rounds}r_resume.log"
    else:
        rounds = args.split("--rounds")[1].split()[0]
        log = f"train_{rounds}r_full.log"

    cmd = f'"{PY}" -u train.py {args}'
    print(f"\n即将执行：{name}")
    print(f"命令：{cmd}")
    print(f"日志：{log}（另存 CSV 和 TensorBoard 到 checkpoints/）")
    print("\n提示：训练开始后可以关掉这个窗口吗？—— 不能，关掉就等于停止。")
    print("      想边跑边看曲线：另开一个「启动TensorBoard.bat」。")
    print("      中断了想继续：再运行本启动器选 [5] 续训。\n")
    input("按回车开始训练...")

    print("\n" + "=" * 62)
    print("训练进行中，请勿关闭本窗口。日志实时输出如下：")
    print("=" * 62 + "\n")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u", "train.py"] + args.split(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, bufsize=1,
                             universal_newlines=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        p.wait()

    print("\n训练结束。")
    input("按回车退出...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中断（Ctrl+C）。想继续请选 [5] 续训。")
        sys.exit(0)
