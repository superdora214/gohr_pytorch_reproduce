"""
训练数据生成模块（PyTorch 向量化版）
=====================================
对照 Gohr 官方 make_train_data + convert_to_binary，
重写为 PyTorch 批量向量化，支持 GPU 一次生成海量样本。

数据语义：
  - 每样本独立随机 64bit 密钥 + 随机明文 P0
  - 标签 Y=1（真实对）：P1 = P0 XOR diff（diff 默认 0x0040）
  - 标签 Y=0（随机对）：P1 完全随机
  - 加密 nr 轮后，把两密文转成 64bit 位向量作为神经网络输入
"""
import torch

from src.speck.speck import WORD_SIZE, batch_expand_key, batch_encrypt

# 默认输入差分（Gohr 论文用 0x0040）
DEFAULT_DIFF = (0x0040, 0)


def to_bitplanes(c0, c1, c2, c3):
    """
    官方 convert_to_binary 的向量化版。
    输入 4 个 shape[N] 的 16bit 张量，输出 shape[N, 64] uint8 位向量。
    位顺序 MSB first，布局与官方一致。
    """
    N = c0.shape[0]
    dev = c0.device
    X = torch.zeros((N, 4 * WORD_SIZE), dtype=torch.uint8, device=dev)
    for bi, blk in enumerate([c0, c1, c2, c3]):
        for i in range(WORD_SIZE):
            X[:, bi * WORD_SIZE + i] = (blk >> (WORD_SIZE - 1 - i)) & 1
    return X


def make_train_data(n, nr, diff=DEFAULT_DIFF, device='cuda', seed=None):
    """
    生成 n 个训练样本（对照官方 make_train_data 语义）。
    返回 (X[N,64] uint8, Y[N] uint8 标签0/1)
    """
    if seed is not None:
        torch.manual_seed(seed)
    # 标签 0/1（一半）
    Y = torch.randint(0, 2, (n,), device=device)
    # 随机密钥 4 字
    keys = torch.randint(0, 1 << WORD_SIZE, (4, n), dtype=torch.int64, device=device)
    # 明文 P0
    plain0l = torch.randint(0, 1 << WORD_SIZE, (n,), dtype=torch.int64, device=device)
    plain0r = torch.randint(0, 1 << WORD_SIZE, (n,), dtype=torch.int64, device=device)
    # P1 = P0 XOR diff（真实对）
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    # Y==0 的样本：P1 完全随机（随机对）
    nz = (Y == 0).sum().item()
    if nz > 0:
        plain1l[Y == 0] = torch.randint(0, 1 << WORD_SIZE, (nz,), dtype=torch.int64, device=device)
        plain1r[Y == 0] = torch.randint(0, 1 << WORD_SIZE, (nz,), dtype=torch.int64, device=device)
    # 密钥编排 + 加密
    subkeys = batch_expand_key(keys, nr)
    cl0, cr0 = batch_encrypt(plain0l, plain0r, subkeys)
    cl1, cr1 = batch_encrypt(plain1l, plain1r, subkeys)
    X = to_bitplanes(cl0, cr0, cl1, cr1)
    return X, Y


if __name__ == "__main__":
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    X, Y = make_train_data(1000, 5, device=device)
    print(f"X: {X.shape}, Y: {Y.shape}, 标签1占比: {Y.float().mean().item():.2f}")
    assert X.shape == (1000, 64)
    t0 = time.time()
    make_train_data(100000, 5, device=device)
    print(f"生成10万样本耗时: {time.time()-t0:.1f}s")
