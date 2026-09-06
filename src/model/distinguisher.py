"""
Bit-Sliced 卷积 ResNet（神经区分器网络）
=========================================
对照 Gohr 官方 train_nets.py 的 make_resnet，用 PyTorch 从零重写。

结构：
  输入 [N, 64] bit 向量（密文对位平面）
  → view [N, 16, 4]     (16 个词内位置 × 4 位面块)
  → permute [N, 4, 16]  → Conv1D 沿位置卷积
  → 1x1 Conv 扩到 num_filters 通道
  → depth 个残差块 (Conv1D+BN+ReLU ×2 + 恒等 Add)
  → Flatten → Dense(64) → Dense(64) → Dense(1, sigmoid)

训练配置：MSE loss + Adam + 循环学习率（同官方）。
"""
import torch
import torch.nn as nn


class BitSlicedResNet(nn.Module):
    def __init__(self, num_blocks=2, num_filters=32, word_size=16,
                 d1=64, d2=64, ks=3, depth=5):
        super().__init__()
        self.word_size = word_size
        self.channels_in = 2 * num_blocks   # =4

        # 预处理：1x1 conv 扩通道（bit-sliced 层）
        self.conv0 = nn.Conv1d(self.channels_in, num_filters, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(num_filters)

        # 残差塔
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            ))

        # 预测头
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * word_size, d1),
            nn.BatchNorm1d(d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.ReLU(),
            nn.Linear(d2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.word_size, self.channels_in)  # [N, 16, 4]
        x = x.permute(0, 2, 1)                            # [N, 4, 16] (ch, len)
        x = torch.relu(self.bn0(self.conv0(x)))
        shortcut = x
        for blk in self.blocks:
            out = blk(x)
            x = out + shortcut
            shortcut = x
        return self.head(x)


def make_distinguisher(depth=5, **kwargs):
    """构造神经区分器"""
    return BitSlicedResNet(depth=depth, **kwargs)
