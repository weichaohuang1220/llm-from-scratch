"""
========================================================================
第 1 章：RMSNorm — Root Mean Square Layer Normalization
========================================================================

【这一章你会学到】
1. 为什么 Transformer 每一层都要做归一化
2. LayerNorm 和 RMSNorm 的差别，以及为什么现代 LLM 抛弃了 LayerNorm
3. 混合精度下"先升 float32 再降回去"的小技巧
4. 用一行 PyTorch 代码就能写完核心计算

------------------------------------------------------------------------
【公式】
    RMS(x) = sqrt( (1/n) * Σ x_i² + ε )
    y      = (x / RMS(x)) * γ

  - x:   一个 token 的特征向量, shape = (..., dim)
  - n:   dim
  - ε:   防止除零的极小值（1e-5 或 1e-6）
  - γ:   可学习缩放参数, shape = (dim,), 初始化为全 1

------------------------------------------------------------------------
【LayerNorm vs RMSNorm】
                      LayerNorm                RMSNorm
    中心化(减均值)        ✅                     ❌
    缩放(除标准差)        ✅                     ✅ (用 RMS 替代 std)
    可学习偏置 β         ✅                     ❌
    可学习缩放 γ         ✅                     ✅
    速度                慢                     快 ~30%
    现代 LLM 是否采用     ❌(已淘汰)             ✅(LLaMA/Qwen/Mistral)

  关键洞察:
  论文 (Zhang & Sennrich, 2019) 实证发现, "减均值" 这一步对
  Transformer 的最终效果几乎没有贡献, 砍掉之后训练更快、效果不变。
========================================================================
"""

import torch
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 全 1 的原因是训练开始时让 RMSNorm 像"不存在"一样，不破坏初始分布。
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
       return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight
    
    def forward(self, x):
        #x.float() 提升精度-> _norm(x.float) -> weight(缩放) ->结果.type_as(x)
        return self._norm(x.float()).type_as(x)
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2,4,8)
    
    rms_norm = RMSNorm(dim = 8)
    y = rms_norm(x)
    print("input shape", x.shape)
    print("output shape", y.shape)
        
    