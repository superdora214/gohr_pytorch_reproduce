"""
Bit-Sliced 卷积 ResNet（神经区分器网络）
=========================================
对照 Gohr 官方 train_nets.py 的 make_resnet，用 PyTorch 从零重写。
论文对应：Gohr 2019 第 4.2 节 "Network Structure"

整体流程（数据形状变化）：
  输入 [N, 64]            ← N 个样本，每个是"一对密文的 64 个比特"
  → view [N, 4, 16]       ← 论文的"面向字"表示：4 个 16 位字 × 每个字内 16 位
  → 1x1 Conv 扩到 32 通道  ← 论文的 "initial bit-sliced convolution"
  → depth 个残差块         ← 论文的 "residual tower"（5/6 轮用 depth=10）
  → Flatten → Dense(64) → Dense(64) → Dense(1, sigmoid)
                          ← 论文的 "prediction head"，输出 0~1 的分数

训练配置：MSE loss + Adam + 循环学习率（同官方）。
"""
import torch
import torch.nn as nn


class BitSlicedResNet(nn.Module):
    def __init__(self, num_blocks=2, num_filters=32, word_size=16,
                 d1=64, d2=64, ks=3, depth=5):
        """
        参数说明（改这些就能改网络大小）：
          num_blocks  : 密文对的块数，固定 2（一对密文 = 2 个 block）
          num_filters : 卷积通道数，论文用 32
          word_size   : 字长，SPECK32/64 是 16 位
          d1, d2      : 预测头两个隐藏层的宽度，论文用 64
          ks          : 卷积核宽度，论文用 3（对应"相邻比特间进位传播"的局部性）
          depth       : 残差块个数 ← 你训练时用的 --depth 10 就是这个
        """
        super().__init__()
        self.word_size = word_size
        self.channels_in = 2 * num_blocks   # =4：两个密文，每个分左右两个 16 位半字

        # ---- ① 初始 1×1 卷积（论文的 initial bit-sliced convolution）----
        # 宽度为 1 的卷积：只在"通道"方向融合信息，不跨位置。
        # 作用是把 4 个通道扩展到 32 个通道，方便后面的残差塔处理。
        self.conv0 = nn.Conv1d(self.channels_in, num_filters, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(num_filters)      # 批归一化：稳定训练、加速收敛

        # ---- ② 残差塔（论文的 residual tower，重复 depth 次）----
        # 每个残差块 = 两层卷积（宽度 ks=3）+ 批归一化 + ReLU
        #   卷积宽度取 3 的原因：匹配"模加法的进位只影响邻近比特"这一局部性
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

        # ---- ③ 预测头（论文的 prediction head）----
        # 注意：这里用全连接层而不是池化层。论文解释说：
        # "加密后的数据没有空间对称性，用池化提取局部特征是徒劳的"
        self.head = nn.Sequential(
            nn.Flatten(),                             # [N, 32, 16] → [N, 512]
            nn.Linear(num_filters * word_size, d1),   # 512 → 64
            nn.BatchNorm1d(d1),
            nn.ReLU(),
            nn.Linear(d1, d2),                        # 64 → 64
            nn.BatchNorm1d(d2),
            nn.ReLU(),
            nn.Linear(d2, 1),                         # 64 → 1：每个样本输出一个数
            nn.Sigmoid(),                             # 压到 0~1：越接近 1 越像"真实差分对"
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化：让每层输出的方差保持一致，避免训练初期梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播：数据从输入走到输出。
        x 的形状：[N, 64]，N 是一批样本数（比如 5000）
        """
        N = x.shape[0]

        # 第 1 步：把 64 个比特组织成论文要求的"面向字"表示
        #   一对密文 = 4 个 16 位字（密文1左字, 密文1右字, 密文2左字, 密文2右字），共 64 比特。
        #   view(N, 4, 16) 的含义：通道 j = 第 j 个 16 位字，位置 i = 该字内第 i 位。
        #   这正好是 PyTorch Conv1d 需要的 (batch, 通道, 长度) 格式，
        #   等价于论文的 Reshape((4,16)) + Permute((2,1)) 组合。
        #   ⚠️ 注意：不能写成 view(N, 16, 4) 再 permute —— 那会按"每 4 比特一组"切分，
        #   破坏密码的面向字结构，导致 5 轮以上准确率大幅下降（已验证）。
        x = x.view(N, self.channels_in, self.word_size)   # [N, 4, 16]

        # 第 2 步：初始卷积 + 归一化 + 激活
        x = torch.relu(self.bn0(self.conv0(x)))           # [N, 32, 16]

        # 第 4 步：残差塔 —— 这就是"残差"的含义：
        #   每个块的输出 = 块的计算结果 + 块的输入（跳跃连接）
        #   好处：梯度可以直接穿过加法回流到浅层，网络才敢堆这么深
        shortcut = x
        for blk in self.blocks:
            out = blk(x)
            x = out + shortcut      # ← 跳跃连接（skip connection）
            shortcut = x            # ← 关键：下一块的 shortcut 用更新后的 x（与论文一致）

        # 第 5 步：预测头输出 0~1 的分数
        return self.head(x)


def make_distinguisher(depth=5, **kwargs):
    """构造神经区分器（train.py 通过这个函数创建网络）"""
    return BitSlicedResNet(depth=depth, **kwargs)
