"""
========================================================================
第3章: Multi-Head Attention (多头注意力) + GQA (分组查询注意力)
========================================================================

Attention 是 Transformer 的灵魂。一句话概括:
  "让每个 token 看看其他所有 token，决定应该关注谁。"

数学公式:
  Attention(Q, K, V) = softmax(Q × K^T / √d) × V

直觉解释:
  Q (Query):  "我在找什么信息？"
  K (Key):    "我有什么信息可以提供？"
  V (Value):  "我的具体内容是什么。"

  Q × K^T → 相似度分数矩阵（谁和谁相关）
  / √d    → 缩放，防止分数过大导致 softmax 梯度消失
  softmax  → 转为概率分布（加起来=1）
  × V     → 按权重加和，得到每个 token 的最终表示

多头 (Multi-Head) 的意义:
  一个头只能关注一种模式（比如"语法关系"）。
  多个头可以同时关注不同模式（"语法"、"语义"、"共指"等等）。
  实现上: 把 hidden_size 切成 num_heads 份，每份独立做 Attention，最后拼回来。

GQA (Grouped Query Attention) — MiniMind 的关键优化:
  标准 MHA: Q, K, V 都有 num_heads 个头 → 显存大
  MQA:      Q 有 num_heads 个头, K/V 只有 1 个头 → 太极端, 效果损失
  GQA:      Q 有 num_heads 个头, K/V 有 num_kv_heads 个头 (介于两者之间)
            MiniMind 默认: 8 个 Q 头, 4 个 KV 头 → 每 2 个 Q 头共享 1 组 KV

  节省了 K/V 的参数量和 KV Cache 显存，推理速度更快，效果接近 MHA。

因果掩码 (Causal Mask):
  语言模型是"从左到右"生成的，每个 token 只能看到它前面（含自己）的 token。
  实现: 在 attention score 矩阵的右上三角填 -inf，softmax 后变成 0。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从前两章导入
from new.O1_rmsnorm import RMSNorm
from new.O2_rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA 的关键辅助函数: 把 KV 头复制 n_rep 次，对齐 Q 头的数量。

    例如: Q 有 8 个头, KV 有 4 个头 → n_rep = 8/4 = 2
          每个 KV 头复制 2 份: [kv0, kv1, kv2, kv3] → [kv0, kv0, kv1, kv1, kv2, kv2, kv3, kv3]
          这样第 0,1 号 Q 头共享 kv0, 第 2,3 号 Q 头共享 kv1, ...

    参数:
        x: (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 每个 KV 头需要复制的次数 = num_q_heads / num_kv_heads

    返回:
        (batch, seq_len, num_kv_heads * n_rep, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # KV 头数 = Q 头数，不需要复制（退化为标准 MHA）

    # 核心操作拆解:
    # x[:, :, :, None, :]  → 在第3维(kv_head后面)插入一个新维度
    #   形状: (bs, slen, num_kv_heads, 1, head_dim)
    # .expand(..., n_rep, ...)  → 沿新维度复制 n_rep 次 (不真的复制内存, 只是"视图")
    #   形状: (bs, slen, num_kv_heads, n_rep, head_dim)
    # .reshape(...)  → 合并 kv_head 和 rep 两个维度
    #   形状: (bs, slen, num_kv_heads * n_rep, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力层，支持:
    - GQA (Grouped Query Attention): Q 头数 ≥ KV 头数
    - RoPE: 旋转位置编码
    - QK-Norm: 对 Q 和 K 做 RMSNorm (稳定训练)
    - Flash Attention: PyTorch 2.0+ 的高效实现
    - KV Cache: 推理时缓存已计算的 K/V, 避免重复计算
    - 因果掩码: 只能看到左边的 token
    """

    def __init__(self, config):
        super().__init__()

        # ---- 头数配置 ----
        # num_key_value_heads: KV 头数 (GQA 的核心参数)
        # 如果没指定, 就等于 Q 头数 (退化为标准 MHA)
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads      # Q 头数, 如 8
        self.n_local_kv_heads = self.num_key_value_heads     # KV 头数, 如 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 复制倍数, 如 2
        self.head_dim = config.head_dim                      # 每头维度, 如 96

        # ---- 因果标志 ----
        self.is_causal = True  # 语言模型必须是因果的(只看左边)

        # ---- 投影矩阵 ----
        # Q 投影: hidden_size → num_q_heads × head_dim
        #   例: 768 → 8 × 96 = 768
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)

        # K 投影: hidden_size → num_kv_heads × head_dim
        #   例: 768 → 4 × 96 = 384   ← 比 Q 小! 这就是 GQA 省参数的地方
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # V 投影: 同 K
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # 输出投影: 把多头拼接的结果映射回 hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # ---- QK Norm ----
        # 对 Q 和 K 做 RMSNorm, 防止注意力分数过大
        # 这是近年来的改进, 原始 Transformer 没有这个
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ---- Dropout ----
        self.attn_dropout = nn.Dropout(config.dropout)   # attention 权重的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 残差连接的 dropout
        self.dropout = config.dropout

        # ---- Flash Attention 检测 ----
        # PyTorch 2.0+ 内置了 scaled_dot_product_attention, 使用 FlashAttention 算法
        # 优势: 显存从 O(n²) 降到 O(n), 速度快 2-4 倍
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        参数:
            x: 输入张量, (batch, seq_len, hidden_size)
            position_embeddings: (cos, sin) 元组, 来自预计算的 RoPE
            past_key_value: KV Cache, 推理时用于避免重复计算
            use_cache: 是否返回 KV Cache
            attention_mask: padding mask, 标记哪些位置是 padding

        返回:
            output: (batch, seq_len, hidden_size)
            past_kv: KV Cache (如果 use_cache=True)
        """
        bsz, seq_len, _ = x.shape

        # ======== 第1步: 线性投影, 得到 Q, K, V ========
        xq = self.q_proj(x)  # (bsz, seq_len, num_q_heads * head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, num_kv_heads * head_dim)
        xv = self.v_proj(x)  # (bsz, seq_len, num_kv_heads * head_dim)

        # ======== 第2步: reshape 成多头形式 ========
        # 把最后一维拆成 (num_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        #   (bsz, seq_len, 8, 96) — 8个Q头

        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        #   (bsz, seq_len, 4, 96) — 4个KV头

        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        #   (bsz, seq_len, 4, 96)

        # ======== 第3步: QK Norm ========
        # 在旋转之前对 Q/K 做归一化, 防止注意力分数爆炸
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # ======== 第4步: 应用 RoPE 旋转位置编码 ========
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # Q 和 K 现在携带了位置信息
        # 注意: V 不做旋转! 位置信息只需要在 Q·K 点积中体现

        # ======== 第5步: KV Cache (推理加速) ========
        # 推理时, 每次只输入 1 个新 token, 但需要和所有历史 token 做 attention
        # KV Cache 存储了历史 token 的 K/V, 避免重复计算
        if past_key_value is not None:
            # 把新的 K/V 拼接到缓存后面
            xk = torch.cat([past_key_value[0], xk], dim=1)  # seq_len 维度拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # ======== 第6步: 转置 + GQA 复制 KV 头 ========
        # transpose(1,2): (bsz, seq_len, heads, dim) → (bsz, heads, seq_len, dim)
        # 这是因为 attention 的矩阵乘法在 seq_len 和 head_dim 维度上进行
        xq = xq.transpose(1, 2)
        # repeat_kv: 把 4 个 KV 头复制成 8 个, 对齐 Q 头数
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        # 现在 xq, xk, xv 形状都是 (bsz, num_q_heads, seq_len, head_dim)

        # ======== 第7步: 计算注意力 ========
        # 两种实现: Flash Attention (快) 或 手动实现 (通用)

        # --- 7a: Flash Attention ---
        # 条件: 有 Flash 支持 + 序列长度>1 + 不需要特殊 mask 处理
        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )
        else:
            # --- 7b: 手动实现 Attention (方便理解) ---

            # Q × K^T / √d → 注意力分数矩阵
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores 形状: (bsz, heads, seq_len, total_kv_len)

            # 因果掩码: 右上三角填 -inf
            # 只对最后 seq_len 列操作 (因为有 KV Cache 时, K 的长度 > 当前 seq_len)
            if self.is_causal:
                causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device)
                causal_mask = causal_mask.triu(1)  # 上三角(不含对角线)为 -inf, 其余为 0
                scores[:, :, :, -seq_len:] += causal_mask
                # 加完之后, position i 只能看到 position 0..i 的分数, 后面的都是 -inf

            # Padding 掩码: 把 padding 位置的分数也设为 -inf
            if attention_mask is not None:
                # attention_mask: (bsz, total_kv_len), 1=有效, 0=padding
                # 变换: (bsz, 1, 1, total_kv_len) 以便广播
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            # Softmax → 概率分布 (每行加起来=1)
            # .float() 是为了数值稳定, .type_as(xq) 转回原精度
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)

            # Attention Dropout (训练时随机丢弃一些注意力权重)
            attn_weights = self.attn_dropout(attn_weights)

            # 加权求和: 概率 × V
            output = attn_weights @ xv

        # ======== 第8步: 拼接多头, 输出投影 ========
        # transpose(1,2): (bsz, heads, seq_len, dim) → (bsz, seq_len, heads, dim)
        # reshape: → (bsz, seq_len, heads * dim) = (bsz, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 输出投影 + 残差 dropout
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    from new.O2_rope import precompute_freqs_cis

    # 模拟一个简单的 config
    class SimpleConfig:
        hidden_size = 64
        num_attention_heads = 4     # 4 个 Q 头
        num_key_value_heads = 2     # 2 个 KV 头 → GQA, 每 2 个 Q 头共享 1 组 KV
        head_dim = 16               # 64 / 4 = 16
        rms_norm_eps = 1e-6
        dropout = 0.0
        flash_attn = False          # 关闭 Flash 以便调试

    config = SimpleConfig()
    attn = Attention(config)

    # 输入
    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, config.hidden_size)

    # 预计算 RoPE
    freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=32)
    cos = freqs_cos[:seq_len]
    sin = freqs_sin[:seq_len]

    # 前向传播
    output, past_kv = attn(x, position_embeddings=(cos, sin), use_cache=True)
    print("输入形状:", x.shape)            # (2, 8, 64)
    print("输出形状:", output.shape)        # (2, 8, 64)
    print("KV Cache K 形状:", past_kv[0].shape)  # (2, 8, 2, 16) — 2个KV头

    # 统计参数量
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"\n参数统计:")
    print(f"  Q 投影: {config.hidden_size} × {config.num_attention_heads * config.head_dim} = {config.hidden_size * config.num_attention_heads * config.head_dim}")
    print(f"  K 投影: {config.hidden_size} × {config.num_key_value_heads * config.head_dim} = {config.hidden_size * config.num_key_value_heads * config.head_dim}")
    print(f"  V 投影: 同 K")
    print(f"  如果用标准 MHA, K/V 参数量会翻倍!")
    print(f"  总参数量: {total_params}")
