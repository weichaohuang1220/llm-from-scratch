"""
========================================================================
第2章: RoPE — Rotary Position Embedding (旋转位置编码)
========================================================================

核心问题: Attention 是"集合运算"，不关心顺序。
         "我爱你" 和 "你爱我" 如果没有位置信息，Attention 算出完全一样的结果。
         所以必须把 "第几个位置" 这个信息注入进去。

RoPE 的核心思想:
--------------
把 Q 和 K 向量看成若干个二维平面上的点，按位置旋转不同角度。
这样当 Q 和 K 做点积时，结果自然只依赖于它们的"相对位置差"。

直觉举例:
  假设 head_dim=4, 向量 [x0, x1, x2, x3]
  分成 2 组二维坐标: (x0, x1) 和 (x2, x3)

  在 position=pos 时:
    第1组旋转角度 = pos × θ₀   (θ₀ 频率高, 旋转快)
    第2组旋转角度 = pos × θ₁   (θ₁ 频率低, 旋转慢)

  其中 θ_k = 1 / (base^(2k/dim)),  base=1000000
  低维转得快（捕捉近距离关系），高维转得慢（捕捉远距离关系），像傅里叶级数。

二维旋转公式:
  x' = x·cosθ - y·sinθ
  y' = x·sinθ + y·cosθ

实现 trick — rotate_half:
  不真的做2×2矩阵乘法，而是用一个等价变换:
  rotate_half([a, b, c, d]) = [-c, -d, a, b]   (后半取负放前面)
  然后: x_rotated = x * cos + rotate_half(x) * sin
  数学上完全等价于逐对旋转，但实现上更高效(纯向量运算，无需循环)。
"""

import torch
import math



"""
预计算所有位置的 cos 和 sin 值。
因为这些值只与 (位置, 维度) 有关，与输入数据无关，所以只需算一次。

参数:
    dim (int): 每个注意力头的维度 (head_dim), 如 96
    end (int): 最大序列长度, 如 32768
    rope_base (float): 频率基数, 控制旋转速度的衰减。
                        MiniMind 用 1e6 (比 LLaMA 的 1e4 大，有利于长文本)

返回:
    freqs_cos: (end, dim) 的 cos 值矩阵
    freqs_sin: (end, dim) 的 sin 值矩阵
"""
def precompute_freqs_cis(dim: int, end: int, theta=10000.0):
  # Step 1: 算频率 θ_i = 1 / (theta ^ (2i/dim)),  i ∈ [0, dim/2)
  # 注意只算 dim/2 个频率, 因为每个频率管一对 (2i, 2i+1)
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float()/dim))
  
  t = torch.arrange(end, device = freqs.device)
  
  freqs = torch.outer(t, freqs).float()
  
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis

  