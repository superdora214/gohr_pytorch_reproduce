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

    n = X_tr.shape[0]
    bs = args.batch_size
    best = 0.0
    for epoch in range(1, args.epochs + 1):
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
        if acc > best:
            best = acc
            ckpt = os.path.join(args.outdir, f"net{args.rounds}r.pt")
            torch.save(net.state_dict(), ckpt)

    print(f"\n[{args.rounds}轮] 最佳验证准确率 = {best:.4f} | 权重保存至 {args.outdir}")
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
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
