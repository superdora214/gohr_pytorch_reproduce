# -*- coding: utf-8 -*-
"""
SPECK32/64 神经密钥恢复攻击（对照 Gohr 2019 第 4.4 节）

攻击流程（以 8 轮为例）：
    明文侧：前置 1 轮（用密钥 0 做逆轮 —— 攻击者可免费控制第一轮输出差分）
    中间：  6 轮神经区分器（r 轮，由 --net-rounds 指定）
    密文侧：试钥剥离 1 轮（枚举 2^16 个候选子密钥）
    1 + 6 + 1 = 8 轮

打分公式（论文式 3）：
    score(k) = Σ_i log2( Z_i^k / (1 - Z_i^k) )
其中 Z_i^k 是网络对"用候选密钥 k 剥轮后第 i 对密文"的预测。
真密钥的得分应显著高于错误密钥 —— 这就是密钥排名（key rank）。

用法：
    python key_recovery.py                      # 默认 8 轮，64 对密文
    python key_recovery.py --pairs 128          # 用更多密文
    python key_recovery.py --trials 20          # 重复 20 次统计排名
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.speck.speck import (WORD_SIZE, MASK, batch_dec_one_round, dec_one_round,
                             encrypt, expand_key)
from src.data.dataset import to_bitplanes
from src.model.distinguisher import make_distinguisher

DEFAULT_DIFF = (0x0040, 0x0000)


def gen_attack_data(n_pairs, rounds, diff=DEFAULT_DIFF, device="cuda", seed=None):
    """
    生成攻击用的数据：一对密文 + 真实密钥。

    关键技巧：先在明文侧做一次"密钥为 0 的逆一轮"，
    这样用真密钥加密 rounds 轮后，剥掉最后一轮就得到
    "差分为 diff 的明文对经过 rounds-2 轮" —— 正好匹配区分器的训练分布。

    返回 (密文四元组, 真实子密钥列表)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 明文对：第二块的左半字翻转 diff[0] 位，右半字翻转 diff[1] 位
    p0l = torch.randint(0, 1 << WORD_SIZE, (n_pairs,), dtype=torch.int64, device=device)
    p0r = torch.randint(0, 1 << WORD_SIZE, (n_pairs,), dtype=torch.int64, device=device)
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]

    # 明文侧前置一轮（密钥 0 的逆轮），使第一轮输出差恰为 diff
    p0l, p0r = batch_dec_one_round(p0l, p0r, torch.zeros_like(p0l))
    p1l, p1r = batch_dec_one_round(p1l, p1r, torch.zeros_like(p1l))

    # 随机密钥 + 加密 rounds 轮
    key = torch.randint(0, 1 << WORD_SIZE, (4,), dtype=torch.int64)
    ks = expand_key(key.tolist(), rounds)

    # 用 expand_key 得到的子密钥逐个加密（张量版）
    def enc_all(x, y):
        for k in ks:
            k_t = torch.full_like(x, k)
            c0 = torch.bitwise_right_shift(x, 7) | ((x << (WORD_SIZE - 7)) & MASK)
            c0 = (c0 + y) & MASK
            c0 = c0 ^ k_t
            c1 = ((y << 2) & MASK) | torch.bitwise_right_shift(y, WORD_SIZE - 2)
            c1 = c1 ^ c0
            x, y = c0, c1
        return x, y

    c0l, c0r = enc_all(p0l, p0r)
    c1l, c1r = enc_all(p1l, p1r)
    return (c0l, c0r, c1l, c1r), ks


def key_recovery(net, cts, rounds, device="cuda", cand_batch=8192, net_batch=200000):
    """
    对 2^16 个候选子密钥打分，返回（得分张量, 每个候选的密钥值）。

    cts: (c0l, c0r, c1l, c1r) —— 一对密文（形状 [n_pairs]）
    """
    c0l, c0r, c1l, c1r = cts
    n_pairs = c0l.shape[0]
    n_cand = 1 << WORD_SIZE          # 65536 个候选子密钥
    trial_keys = torch.arange(n_cand, dtype=torch.int64, device=device)

    scores = torch.zeros(n_cand, dtype=torch.float32, device=device)
    net.eval()

    with torch.no_grad():
        for start in range(0, n_cand, cand_batch):
            end = min(start + cand_batch, n_cand)
            k = trial_keys[start:end]                      # [B]
            B = k.shape[0]

            # 用候选密钥剥掉最后一轮：[B, n_pairs]
            kk = k.unsqueeze(1).expand(B, n_pairs)
            d0l, d0r = batch_dec_one_round(
                c0l.unsqueeze(0).expand(B, -1), c0r.unsqueeze(0).expand(B, -1), kk)
            d1l, d1r = batch_dec_one_round(
                c1l.unsqueeze(0).expand(B, -1), c1r.unsqueeze(0).expand(B, -1), kk)

            # 转成 64 维位向量输入，分批推理
            X = to_bitplanes(d0l.reshape(-1), d0r.reshape(-1),
                             d1l.reshape(-1), d1r.reshape(-1))   # [B*n_pairs, 64]
            zs = []
            for i in range(0, X.shape[0], net_batch):
                xb = X[i:i + net_batch].float()
                zs.append(net(xb))
            Z = torch.cat(zs, dim=0).reshape(B, n_pairs)

            # 似然比打分（论文式 3）：log2(Z / (1-Z))，对所有密文对求和
            Z = Z.clamp(1e-7, 1 - 1e-7)
            scores[start:end] = torch.log2(Z / (1 - Z)).sum(dim=1)

    return scores, trial_keys


def main():
    ap = argparse.ArgumentParser(description="SPECK32/64 神经密钥恢复攻击")
    ap.add_argument("--rounds", type=int, default=8, help="被攻击的总轮数")
    ap.add_argument("--net-rounds", type=int, default=6, help="区分器的轮数（默认 6）")
    ap.add_argument("--pairs", type=int, default=64, help="使用的密文对数（论文用 64）")
    ap.add_argument("--trials", type=int, default=1, help="重复次数（统计平均排名）")
    ap.add_argument("--depth", type=int, default=10, help="网络深度（需与权重一致）")
    ap.add_argument("--weights", default=None, help="权重路径（默认 checkpoints/net{N}r.pt）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    weights = args.weights or os.path.join("checkpoints", f"net{args.net_rounds}r.pt")
    if not os.path.exists(weights):
        sys.exit(f"找不到权重文件：{weights}\n请先训练 {args.net_rounds} 轮区分器。")

    # 载入区分器
    net = make_distinguisher(depth=args.depth).to(args.device)
    state = torch.load(weights, map_location=args.device)
    net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    net.eval()
    print(f"区分器：{weights}（{args.net_rounds} 轮，depth={args.depth}）")

    expect = args.net_rounds + 2
    if args.rounds != expect:
        print(f"⚠️  提示：{args.net_rounds} 轮区分器最适合攻击 {expect} 轮（1 前置 + {args.net_rounds} 网络 + 1 试钥）")

    print(f"攻击配置：{args.rounds} 轮 | {args.pairs} 对密文 | 候选密钥 2^16 = 65536 个")
    print(f"设备：{args.device}\n")

    ranks = []
    t0 = time.time()
    for t in range(args.trials):
        cts, ks = gen_attack_data(args.pairs, args.rounds, device=args.device, seed=1000 + t)
        true_k = ks[-1]                              # 真实最后一轮子密钥

        scores, trial_keys = key_recovery(net, cts, args.rounds, device=args.device)

        true_score = scores[true_k].item()
        rank = int((scores > true_score).sum().item())        # 排在真密钥前面的个数
        ranks.append(rank)

        top = torch.topk(scores, k=5)
        print(f"[试次 {t+1}] 真实子密钥 = 0x{true_k:04x} | 得分 {true_score:.2f} | "
              f"排名 {rank}（0 = 第一名）")
        if args.trials == 1:
            print(f"  前 5 名候选：")
            for val, idx in zip(top.values.tolist(), top.indices.tolist()):
                flag = "  ← 真密钥" if idx == true_k else ""
                print(f"    0x{idx:04x}  得分 {val:8.2f}{flag}")

    dt = time.time() - t0
    print(f"\n=== 统计（{args.trials} 次）===")
    print(f"平均排名 {sum(ranks)/len(ranks):.1f} | 中位排名 {sorted(ranks)[len(ranks)//2]} | "
          f"排名 0（第一名）的比例 {sum(1 for r in ranks if r == 0)/len(ranks):.1%}")
    print(f"总耗时 {dt:.1f}s（平均每次 {dt/args.trials:.1f}s）")
    print(f"\n对比论文表 3：9 轮攻击用 7 轮区分器 + 64 对密文，N7 平均排名 52.1、中位 1.0、成功率 35.8%")


if __name__ == "__main__":
    main()
