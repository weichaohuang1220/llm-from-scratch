"""
========================================================================
第6章: MiniMindBlock — Transformer 解码器层
========================================================================

现在我们把前面的所有积木组装起来！

一个 Transformer Block (也叫 Decoder Layer) 包含:
  1. Self-Attention: 让 token 之间互相交流
  2. FeedForward:    让每个 token 自己做非线性变换
  3. RMSNorm:        每个子层前做归一化
  4. 残差连接:        每个子层的输出加上输入 (跳跃连接)

数据流 (Pre-Norm 架构):

  输入 hidden_states
    │
    ├─── 保存为 residual (残差)
    │
    ▼
  RMSNorm (input_layernorm)        ← 先归一化
    │
    ▼
  Self-Attention                   ← Attention 子层
    │
    ▼
  + residual                       ← 残差连接: output = attention_output + residual
    │
    ├─── (这个和就是新的 hidden_states)
    │
    ▼
  RMSNorm (post_attention_layernorm) ← 再归一化
    │
    ▼
  FeedForward (或 MoE)               ← FFN 子层
    │
    ▼
  + hidden_states                    ← 又一次残差连接
    │
    ▼
  输出

Pre-Norm vs Post-Norm:
  原始 Transformer 用 Post-Norm: Norm 放在子层输出之后 → x + Norm(SubLayer(x))
  现代 LLM 用 Pre-Norm:         Norm 放在子层输入之前 → x + SubLayer(Norm(x))
  Pre-Norm 训练更稳定, 不需要 warmup, 已成为主流。

残差连接为什么重要？
  没有残差连接, 深层网络的梯度会指数级衰减("梯度消失")。
  残差连接提供了一条"信息高速公路", 梯度可以直接从最后一层流到第一层。
  数学上: ∂L/∂x = ∂L/∂y × (1 + ∂f/∂x), 那个 "+1" 保证梯度不会消失。
"""

import torch
import torch.nn as nn

from new.O1_rmsnorm import RMSNorm
from new.O3_attention import Attention
from new.O4_feedforward import FeedForward
from new.O5_moe import MOEFeedForward


class MiniMindBlock(nn.Module):
    """
    一个 Transformer 解码器层

    参数:
        layer_id (int): 层编号 (0, 1, 2, ...), 目前未使用但预留给可能的层级特殊处理
        config: 模型配置
    """

    def __init__(self, layer_id: int, config):
        super().__init__()

        # Self-Attention 子层
        self.self_attn = Attention(config)

        # Pre-Norm 1: Attention 之前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Pre-Norm 2: FFN 之前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN 子层: 根据配置选择普通 FFN 或 MoE FFN
        # use_moe=False → 普通 FeedForward (所有token共享一个FFN)
        # use_moe=True  → MOEFeedForward  (多个专家, 每个token只选一部分)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        """
        参数:
            hidden_states: (batch, seq_len, hidden_size) — 当前层的输入
            position_embeddings: (cos, sin) — RoPE 位置编码
            past_key_value: 该层的 KV Cache
            use_cache: 是否缓存 KV
            attention_mask: padding mask

        返回:
            hidden_states: (batch, seq_len, hidden_size) — 当前层的输出
            present_key_value: 更新后的 KV Cache
        """

        # ======== 子层1: Self-Attention + 残差 ========

        # 保存输入作为残差
        residual = hidden_states

        # Pre-Norm: 先归一化, 再送入 Attention
        # 为什么不直接用 hidden_states？
        # → 归一化让输入的数值范围稳定, Attention 的计算更稳健
        # → 但我们不想丢失原始信息, 所以用残差连接把原始值加回来
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # ← 归一化后的输入
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        # 残差连接: Attention 输出 + 原始输入
        hidden_states = hidden_states + residual
        # 为什么用加法？
        # → 如果 Attention 学到的变化很小, hidden_states ≈ residual, 信息不丢失
        # → 如果 Attention 学到了重要变化, 它会叠加在原始信息上

        # ======== 子层2: FeedForward + 残差 ========
        # 注意这里的写法: hidden_states + mlp(norm(hidden_states))
        # 等价于:
        #   residual2 = hidden_states
        #   normed = self.post_attention_layernorm(hidden_states)
        #   ffn_output = self.mlp(normed)
        #   hidden_states = ffn_output + residual2
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    import math
    from new.O2_rope import precompute_freqs_cis

    class SimpleConfig:
        hidden_size = 64
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 16
        hidden_act = 'silu'
        intermediate_size = math.ceil(64 * math.pi / 64) * 64
        rms_norm_eps = 1e-6
        dropout = 0.0
        flash_attn = False
        use_moe = False  # 先测试普通 FFN

    config = SimpleConfig()
    block = MiniMindBlock(layer_id=0, config=config)

    # 输入
    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, config.hidden_size)

    # RoPE
    freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=32)
    cos = freqs_cos[:seq_len]
    sin = freqs_sin[:seq_len]

    # 前向传播
    output, kv_cache = block(x, position_embeddings=(cos, sin), use_cache=True)
    print("输入形状:", x.shape)     # (2, 8, 64)
    print("输出形状:", output.shape)  # (2, 8, 64)  ← 形状完全不变!

    # 验证残差连接的效果
    # 如果我们把所有参数初始化为0, 残差连接会让输出 = 输入
    block_zero = MiniMindBlock(layer_id=0, config=config)
    with torch.no_grad():
        for p in block_zero.parameters():
            p.zero_()
    output_zero, _ = block_zero(x, position_embeddings=(cos, sin))
    print(f"\n全零参数时, 输出是否等于输入: {torch.allclose(output_zero, x, atol=1e-5)}")
    print("→ 残差连接保证了即使子层输出为0, 信息也不会丢失!")

    # 参数量
    total = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for p in block.self_attn.parameters())
    ffn_params = sum(p.numel() for p in block.mlp.parameters())
    norm_params = sum(p.numel() for p in block.input_layernorm.parameters()) + \
                  sum(p.numel() for p in block.post_attention_layernorm.parameters())
    print(f"\n参数分布:")
    print(f"  Attention: {attn_params} ({attn_params/total*100:.0f}%)")
    print(f"  FFN:       {ffn_params} ({ffn_params/total*100:.0f}%)")
    print(f"  Norm:      {norm_params} ({norm_params/total*100:.0f}%)")
    print(f"  总计:      {total}")
