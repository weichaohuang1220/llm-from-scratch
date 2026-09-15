"""
========================================================================
第4章: FeedForward — 前馈神经网络 (SwiGLU 变体)
========================================================================

在 Transformer Block 中, Attention 负责 "token 之间的交互",
而 FeedForward 负责 "每个 token 自身的非线性变换"。

标准 FFN (原始 Transformer):
  FFN(x) = ReLU(x × W1) × W2
  两层线性变换 + 一个激活函数，结构简单。

SwiGLU FFN (MiniMind/LLaMA 用的):
  FFN(x) = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down

  关键变化:
  1. ReLU → SiLU (又叫 Swish): SiLU(x) = x × sigmoid(x)
     SiLU 比 ReLU 更平滑, 没有"硬拐角", 梯度更好
  2. 增加了 "门控" 机制 (Gate):
     - gate_proj: 计算 "门控信号" (哪些维度重要)
     - up_proj:   计算 "候选值" (实际内容)
     - 两者逐元素相乘 (⊙): 门控信号筛选候选值
     - down_proj:  降维回 hidden_size

  为什么叫 "GLU" (Gated Linear Unit)?
  因为有一路是"门"（经过激活函数），另一路是"线性"的，
  两路相乘就是"门控线性单元"。

维度变化:
  输入:     (batch, seq_len, hidden_size)     如 768
  gate_proj: hidden_size → intermediate_size  如 768 → 2016
  up_proj:   hidden_size → intermediate_size  如 768 → 2016
  逐元素相乘: intermediate_size               如 2016
  down_proj: intermediate_size → hidden_size  如 2016 → 768
  输出:     (batch, seq_len, hidden_size)     如 768

intermediate_size 的计算:
  MiniMind 用 ceil(hidden_size × π / 64) × 64
  例: ceil(768 × 3.14159 / 64) × 64 = ceil(37.7) × 64 = 38 × 64 = 2432
  这个 π 的选择比较有趣, 是 MiniMind 作者的一个设计选择。
  ×64 是为了对齐到 GPU 友好的倍数。
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class FeedForward(nn.Module):
    """
    SwiGLU 前馈神经网络

    参数:
        config: 模型配置
        intermediate_size: 中间层维度 (可选, 默认从 config 读取)
                          MoE 场景下每个专家可能用不同的 intermediate_size
    """

    def __init__(self, config, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size

        # gate_proj: "门控"投影 — 决定哪些维度被激活
        # 输入 hidden_size → 输出 intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

        # up_proj: "上投影" — 提供候选内容
        # 与 gate_proj 并行计算, 维度相同
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

        # down_proj: "下投影" — 降维回 hidden_size
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

        # 激活函数: SiLU (Swish)
        # ACT2FN 是 HuggingFace 的激活函数注册表
        # config.hidden_act = 'silu' → ACT2FN['silu'] = SiLU 函数
        # SiLU(x) = x × sigmoid(x) = x × (1 / (1 + e^(-x)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        SwiGLU 前向传播, 一行代码包含了三步:

        拆解:
        1. self.gate_proj(x)         → 门控信号 (hidden → intermediate)
        2. self.act_fn(...)           → 过 SiLU 激活 (非线性)
        3. self.up_proj(x)            → 候选值 (hidden → intermediate)
        4. 步骤2 * 步骤3              → 门控筛选 (逐元素相乘)
        5. self.down_proj(...)        → 降维 (intermediate → hidden)

        直觉:
        gate_proj 像一个"过滤器", 经过 SiLU 激活后值在 (-0.3, +∞) 之间
        up_proj 像"原始信号"
        两者相乘: gate 值大的维度被保留, gate 值小/负的维度被抑制
        这比单纯 ReLU 更精细 — ReLU 只有"开/关"两种状态, SwiGLU 是平滑的
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    import math

    class SimpleConfig:
        hidden_size = 64
        hidden_act = 'silu'
        # MiniMind 的 intermediate_size 计算方式
        intermediate_size = math.ceil(64 * math.pi / 64) * 64  # = 256

    config = SimpleConfig()
    ffn = FeedForward(config)

    x = torch.randn(2, 8, 64)
    output = ffn(x)
    print("输入形状:", x.shape)    # (2, 8, 64)
    print("输出形状:", output.shape)  # (2, 8, 64)

    # 拆开看中间过程
    gate = ffn.act_fn(ffn.gate_proj(x))
    up = ffn.up_proj(x)
    gated = gate * up
    print("\n中间过程:")
    print(f"  gate_proj 输出: {gate.shape}  (hidden→intermediate)")
    print(f"  up_proj 输出:   {up.shape}  (hidden→intermediate)")
    print(f"  门控相乘后:     {gated.shape}")
    print(f"  down_proj 输出: {output.shape}  (intermediate→hidden)")

    # 对比 SiLU vs ReLU
    test_x = torch.linspace(-3, 3, 7)
    print(f"\n激活函数对比 (x = {test_x.tolist()}):")
    print(f"  ReLU(x) = {torch.relu(test_x).tolist()}")
    print(f"  SiLU(x) = {torch.nn.functional.silu(test_x).tolist()}")
    print("  注意: ReLU 在 x<0 时硬性截断为0, SiLU 更平滑, 允许微小负值通过")

    # 参数量统计
    total = sum(p.numel() for p in ffn.parameters())
    print(f"\n参数量: {total}")
    print(f"  = 3 × hidden × intermediate = 3 × {config.hidden_size} × {config.intermediate_size} = {3 * config.hidden_size * config.intermediate_size}")
    print(f"  (gate + up + down 三个矩阵)")
