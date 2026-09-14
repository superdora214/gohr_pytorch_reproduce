"""
PyTorch SPECK32/64 加密模块
=============================
对照 Gohr CRYPTO 2019 论文使用的官方实现（deep_speck/speck.py），
用 PyTorch 从零重写，支持批量向量化加密（GPU 加速）。

已通过 SPECK 官方测试向量验证。
"""
import torch

WORD_SIZE = 16
MASK = 0xFFFF
ALPHA = 7   # 循环左移/右移常数
BETA = 2


def rol(x, k):
    """16bit 循环左移 k 位（张量）"""
    return ((x << k) & MASK) | (x >> (WORD_SIZE - k))


def ror(x, k):
    """16bit 循环右移 k 位（张量）"""
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK)


def enc_one_round(c0, c1, k):
    """
    SPECK32 一轮加密（标量或张量）。
    c0=左块, c1=右块, k=子密钥（均 16bit）
    """
    c0 = ror(c0, ALPHA)
    c0 = (c0 + c1) & MASK
    c0 = c0 ^ k
    c1 = rol(c1, BETA)
    c1 = c1 ^ c0
    return c0, c1


def expand_key(k, rounds):
    """
    Gohr 官方密钥编排（标量版）。
    k: 4 个 16bit 密钥字（list/tensor）
    """
    if isinstance(k, torch.Tensor):
        k = k.tolist()
    k = [int(x) & MASK for x in k]
    ks = [0] * rounds
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[:len(k) - 1]))
    for i in range(rounds - 1):
        l[i % 3], ks[i + 1] = enc_one_round(l[i % 3], ks[i], i)
    return ks


def batch_expand_key(k, rounds):
    """
    向量化密钥编排。k: shape [4, N] 张量（N 个样本各自的 4 个密钥字）
    返回子密钥列表，每个元素 shape [N]
    """
    kt = k.t()
    ks_cur = kt[:, 3].clone()
    l = [kt[:, 2].clone(), kt[:, 1].clone(), kt[:, 0].clone()]
    ks_list = [ks_cur]
    for i in range(rounds - 1):
        idx = i % 3
        c0, c1 = l[idx], ks_list[-1]
        c0 = ror(c0, ALPHA)
        c0 = (c0 + c1) & MASK
        c0 = c0 ^ i
        c1 = rol(c1, BETA)
        c1 = c1 ^ c0
        l[idx] = c0
        ks_list.append(c1)
    return ks_list


def dec_one_round(c0, c1, k):
    """
    SPECK32 一轮解密（enc_one_round 的逆运算）。
    逆序撤销加密的 5 步：异或自逆、加法用减法、ror/rol 互换。
    """
    c1 = c1 ^ c0
    c1 = ror(c1, BETA)
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK
    c0 = rol(c0, ALPHA)
    return c0, c1


def batch_dec_one_round(c0, c1, k):
    """
    向量化的一轮解密。
    c0/c1: shape [N]；k 可以是标量（所有样本用同一候选密钥）或 shape [N]（每个样本不同）。
    用于密钥恢复攻击中"用候选密钥部分解密最后一轮"。
    """
    c1 = c1 ^ c0
    c1 = ror(c1, BETA)
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK
    c0 = rol(c0, ALPHA)
    return c0, c1


def encrypt(pt_l, pt_r, subkeys):
    """完整加密（标量版）。pt_l/r: 明文左右块"""
    x, y = pt_l, pt_r
    for k in subkeys:
        x, y = enc_one_round(x, y, k)
    return x, y


def decrypt(ct_l, ct_r, subkeys):
    """完整解密（标量版），按相反顺序应用子密钥"""
    x, y = ct_l, ct_r
    for k in reversed(subkeys):
        x, y = dec_one_round(x, y, k)
    return x, y


def batch_encrypt(plain_l, plain_r, subkeys):
    """批量加密。plain_l/r: shape[N]，subkeys 每元素 shape[N]"""
    x, y = plain_l.clone(), plain_r.clone()
    for k in subkeys:
        x = ror(x, ALPHA)
        x = (x + y) & MASK
        x = x ^ k
        y = rol(y, BETA)
        y = y ^ x
    return x, y


# ------------------------------------------------------------------
# 官方测试向量验证
# ------------------------------------------------------------------
def check_testvector():
    """SPECK32/64 官方测试向量：验证实现正确性"""
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt[0], pt[1], ks)
    expected = (0xa868, 0x42f2)
    ok = (ct[0] == expected[0] and ct[1] == expected[1])
    if ok:
        print("Testvector verified. SPECK32/64 实现正确!")
    else:
        print(f"Testvector FAILED: got {ct}, expected {expected}")
    return ok


if __name__ == "__main__":
    check_testvector()
