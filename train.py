"""
Gohr 神经区分器训练主脚本（PyTorch 复现）
=========================================
用法示例：
  python train.py --rounds 5 --train-size 1000000 --epochs 25
  python train.py --rounds 6 --train-size 3000000 --epochs 25 --depth 10

对照 Gohr 论文 train_5_rounds.py 的配置（depth=10, 10^7样本, 200 epoch）。
本脚本默认较小规模便于快速验证，可用参数放大到论文级。
"""
import argparse
import os
import sys
import time

import torch

# 保证能 import src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speck.speck import check_testvector
from src.data.dataset import make_train_data
from src.model.distinguisher import make_distinguisher


def cyclic_lr(num_epochs, high_lr=0.002, low_lr=0.0001):
    """对照官方 cyclic_lr：每 num_epochs 周期从 high 线性降到 low"""
    def lr_fn(epoch):
        return low_lr + ((num_epochs - 1) - epoch % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return lr_fn


def train(args):
    device = args.device
    print(f"[配置] 轮数={args.rounds} | 训练样本={args.train_size} | "
          f"验证={args.eval_size} | depth={args.depth} | epoch={args.epochs} | 设备={device}")

    # 验证 SPECK 实现正确
    if not check_testvector():
        sys.exit("SPECK 测试向量未通过，中止")
    print()

    # 生成数据（GPU 向量化，速度快）
    print("生成训练/验证数据...")
    t0 = time.time()
    X_tr, Y_tr = make_train_data(args.train_size, args.rounds, device=device)
    X_va, Y_va = make_train_data(args.eval_size, args.rounds, device=device)
    print(f"数据生成耗时: {time.time()-t0:.1f}s | 训练集 {X_tr.shape}")

    X_tr = X_tr.float()
    Y_tr = Y_tr.float().unsqueeze(1)
    X_va = X_va.float()
    Y_va = Y_va.float().unsqueeze(1)

    # 网络
    net = make_distinguisher(depth=args.depth).to(device)
    print(f"网络参数量: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    sched = cyclic_lr(10, 0.002, 0.0001)

    # ---- 训练记录：CSV（Excel 可开）+ TensorBoard（网页看曲线）----
    csv_path = os.path.join(args.outdir, f"history_{args.rounds}r.csv")
    csv_mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
    csv_f = open(csv_path, csv_mode, encoding="utf-8", newline="")
    if csv_mode == "w":
        csv_f.write("epoch,loss,val_acc,lr\n")
    csv_f.flush()
    print(f"CSV 日志: {csv_path}")

    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.outdir, f"tb_{args.rounds}r")
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard 日志: {tb_dir}")
        except ImportError:
            print("（未安装 tensorboard，跳过）")

    n = X_tr.shape[0]
    bs = args.batch_size
    ckpt_path = os.path.join(args.outdir, f"net{args.rounds}r.pt")      # 最佳权重
    last_path = os.path.join(args.outdir, f"last_{args.rounds}r.pt")    # 最后一个 epoch 的状态（续训用）

    # ---- 断点续训：从上次的权重 + 优化器状态 + epoch 号继续 ----
    best = 0.0
    start_epoch = 1
    if args.resume:
        # 优先用 last_（记录最后一个 epoch），没有才退回最佳权重
        resume_path = last_path if os.path.exists(last_path) else ckpt_path
        if os.path.exists(resume_path):
            state = torch.load(resume_path, map_location=device)
            if isinstance(state, dict) and "model" in state:
                net.load_state_dict(state["model"])
                if "optimizer" in state:
                    optimizer.load_state_dict(state["optimizer"])
                start_epoch = state.get("epoch", 0) + 1
                best = state.get("best", 0.0)
                print(f"[续训] 从 {os.path.basename(resume_path)} 恢复，epoch {start_epoch} 继续（历史最佳 {best:.4f}）")
            else:
                net.load_state_dict(state)
                print("[续训] 加载了旧格式权重（无优化器状态），epoch 从头计数")
        else:
            print(f"[续训] 未找到 {resume_path}，本次从头训练")

    for epoch in range(start_epoch, args.epochs + 1):
        lr = sched(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr
        net.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = X_tr[idx], Y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (n / bs)

        net.eval()
        with torch.no_grad():
            pred = net(X_va)
            acc = ((pred > 0.5).float().eq(Y_va).float().mean().item())
        print(f"epoch {epoch}/{args.epochs} | loss {avg_loss:.4f} | val_acc {acc:.4f} | lr {lr:.5f}")
        csv_f.write(f"{epoch},{avg_loss:.6f},{acc:.6f},{lr:.6f}\n")
        csv_f.flush()
        if writer:
            writer.add_scalar("loss", avg_loss, epoch)
            writer.add_scalar("val_acc", acc, epoch)
            writer.add_scalar("lr", lr, epoch)
        if acc > best:
            best = acc
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "rounds": args.rounds,
                "depth": args.depth,
            }, ckpt_path)
        # 每个 epoch 都存一份"最后状态"，保证续训不丢进度
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "rounds": args.rounds,
            "depth": args.depth,
        }, last_path)

    csv_f.close()
    if writer:
        writer.close()
    print(f"\n[{args.rounds}轮] 最佳验证准确率 = {best:.4f} | 权重保存至 {args.outdir}")
    print(f"CSV 记录: {csv_path}")
    return net, best


def main():
    parser = argparse.ArgumentParser(description="Gohr 神经区分器 PyTorch 复现")
    parser.add_argument("--rounds", type=int, default=5, help="SPECK 加密轮数")
    parser.add_argument("--train-size", type=int, default=1000000, help="训练样本数")
    parser.add_argument("--eval-size", type=int, default=100000, help="验证样本数")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--depth", type=int, default=10, help="残差块深度(论文用10)")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="./checkpoints")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="关闭 TensorBoard 记录（默认开启，日志写到 outdir/tb_Nr）")
    parser.add_argument("--resume", action="store_true",
                        help="从 outdir/net{N}r.pt 断点续训（恢复权重、优化器状态和 epoch）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
