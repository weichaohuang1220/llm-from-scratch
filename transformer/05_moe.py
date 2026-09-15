"""
========================================================================
第5章: MoE — Mixture of Experts (混合专家)
========================================================================

MoE 的核心思想:
  普通 FFN: 每个 token 都经过同一个 FFN → 所有参数都被激活
  MoE FFN:  准备 N 个"专家"(每个就是一个小 FFN),
            每个 token 只挑 Top-K 个专家处理 → 参数量大但实际计算量小

直觉比喻:
  想象一家医院有 4 个专科医生 (4 个专家):
  - 骨科医生、心内科医生、皮肤科医生、眼科医生
  每个病人 (token) 不需要看所有医生，只需挂 1-2 个号 (top-k=1 或 2)
  一个 "导诊台" (Router/Gate) 根据病人的症状 (token 的特征) 决定挂哪个科

MiniMind 的 MoE 配置:
  - num_experts = 4          → 4 个专家
  - num_experts_per_tok = 1  → 每个 token 只选 1 个专家
  - 每个专家是一个独立的 FeedForward (SwiGLU)

Router (路由器/门控):
  一个简单的线性层: hidden_size → num_experts
  输出经过 softmax → 每个专家的概率分数
  选 Top-K 个专家, 用它们的概率作为权重加权求和

辅助损失 (Auxiliary Loss):
  问题: 如果不加约束, Router 可能总是选同几个专家, 其他专家"饿死"
  解决: 添加一个辅助损失, 鼓励负载均衡 (每个专家被选中的频率差不多)
  公式: aux_loss = num_experts × Σ(f_i × P_i)
    f_i = 第 i 个专家被选中的比例 (load)
    P_i = 第 i 个专家的平均路由概率 (importance)
  当所有专家均匀分配时, aux_loss 最小

训练中的小技巧 — 保持梯度流:
  如果某个专家在当前 batch 中完全没被选中, 它的参数不会收到梯度。
  为了避免这个问题, 代码中加了: y[0,0] += 0 * sum(p.sum() for p in expert.parameters())
  这个值为 0 不改变输出, 但让 PyTorch 知道这些参数参与了计算图, 梯度能流过去。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from new.O4_feedforward import FeedForward


class MOEFeedForward(nn.Module):
    """
    Mixture of Experts 前馈层

    结构:
      输入 → Router(门控) → 选Top-K专家 → 各专家独立计算 → 加权求和 → 输出
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Router (门控/路由器):
        # 一个线性层, 把 token 的特征映射到每个专家的分数
        # hidden_size → num_experts (如 768 → 4)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 专家列表: num_experts 个独立的 FeedForward
        # 每个专家可以有不同的 intermediate_size (通过 moe_intermediate_size 控制)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        MoE 前向传播

        参数:
            x: (batch_size, seq_len, hidden_dim)

        返回:
            y: (batch_size, seq_len, hidden_dim)  — 形状不变
        """
        batch_size, seq_len, hidden_dim = x.shape

        # ======== 第1步: 展平 batch 和 seq_len 维度 ========
        # (batch, seq_len, hidden) → (batch * seq_len, hidden)
        # 每个 token 独立路由, 不需要知道它在哪个 batch 或位置
        x_flat = x.view(-1, hidden_dim)
        # 形状: (N, hidd2en_dim),  N = batch_size × seq_len

        # ======== 第2步: Router 计算路由分数 ========
        # gate(x_flat): (N, hidden_dim) → (N, num_experts)
        # softmax: 转为概率分布 (每个 token 对每个专家的偏好)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # scores 形状: (N, num_experts), 如 (N, 4)
        # scores[i] = [0.1, 0.6, 0.2, 0.1] 表示 token i 最想找专家1

        # ======== 第3步: 选 Top-K 个专家 ========
        topk_weight, topk_idx = torch.topk(
            scores,
            k=self.config.num_experts_per_tok,  # 通常 k=1
            dim=-1,
            sorted=False
        )
        # topk_weight: (N, k) — 被选中专家的分数
        # topk_idx:    (N, k) — 被选中专家的索引

        # 归一化权重: 让选中专家的权重加起来=1
        # 当 k=1 时这步没效果, k>1 时确保权重是合法的概率分布
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # ======== 第4步: 各专家独立处理自己的 token ========
        y = torch.zeros_like(x_flat)  # 输出容器, 全0初始化

        for i, expert in enumerate(self.experts):
            # mask: (N, k) 的布尔矩阵, 标记哪些 token 选了第 i 个专家
            mask = (topk_idx == i)

            if mask.any():
                # 找出选了这个专家的 token 的索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # token_idx: 一维张量, 如 [0, 3, 7] 表示第0、3、7个token选了这个专家

                # 取出对应的权重
                weight = topk_weight[mask].view(-1, 1)
                # weight: (num_selected, 1)

                # 专家计算 + 加权
                # expert(x_flat[token_idx]): 只对选中的 token 做 FFN
                # × weight: 乘以路由权重
                expert_output = expert(x_flat[token_idx]) * weight

                # index_add_: 把结果累加回对应位置
                # 如果一个 token 选了多个专家 (k>1), 结果会在 y 中累加
                y.index_add_(0, token_idx, expert_output.to(y.dtype))

            elif self.training:
                # ---- 保持梯度流的 trick ----
                # 如果这个专家没被任何 token 选中, 它的参数不会收到梯度
                # 加一个 0 × params, 值不变但梯度能流通
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # ======== 第5步: 计算辅助损失 (训练时) ========
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个专家被选中的比例
            # F.one_hot(topk_idx, num_experts): (N, k, num_experts) 的 one-hot
            # .float().mean(0): 在所有 token 上平均, 得到每个专家的负载
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)

            # importance: 每个专家的平均路由概率
            importance = scores.mean(0)

            # aux_loss: 负载 × 重要性, 再乘以专家数量和系数
            # 直觉: 如果某专家负载高 AND 概率也高 → 说明路由太集中 → 惩罚
            # 均匀分配时 aux_loss 最小
            self.aux_loss = (
                (load * importance).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            # 推理时不需要辅助损失
            self.aux_loss = scores.new_zeros(1).squeeze()

        # ======== 第6步: 恢复原始形状 ========
        return y.view(batch_size, seq_len, hidden_dim)


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    import math

    class SimpleConfig:
        hidden_size = 64
        hidden_act = 'silu'
        intermediate_size = math.ceil(64 * math.pi / 64) * 64
        # MoE 配置
        num_experts = 4
        num_experts_per_tok = 1       # 每个 token 只选 1 个专家
        moe_intermediate_size = 128   # 每个专家的 FFN 中间维度
        norm_topk_prob = True
        router_aux_loss_coef = 5e-4

    config = SimpleConfig()
    moe = MOEFeedForward(config)
    moe.train()  # 训练模式，计算 aux_loss

    x = torch.randn(2, 8, 64)
    output = moe(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print(f"辅助损失: {moe.aux_loss.item():.6f}")

    # 看看 Router 的路由决策
    with torch.no_grad():
        scores = F.softmax(moe.gate(x.view(-1, 64)), dim=-1)
        chosen = scores.argmax(dim=-1)
        print(f"\n路由决策 (16个token分别选了哪个专家):")
        print(f"  {chosen.tolist()}")
        # 统计每个专家被选中的次数
        for i in range(4):
            count = (chosen == i).sum().item()
            print(f"  专家{i}: 被选中 {count} 次 ({count/16*100:.0f}%)")

    # 参数量对比
    normal_ffn = FeedForward(config)
    moe_params = sum(p.numel() for p in moe.parameters())
    ffn_params = sum(p.numel() for p in normal_ffn.parameters())
    print(f"\n参数量对比:")
    print(f"  普通 FFN: {ffn_params}")
    print(f"  MoE (4专家): {moe_params}")
    print(f"  MoE 参数约为 FFN 的 {moe_params/ffn_params:.1f} 倍")
    print(f"  但每个 token 只激活 1/{config.num_experts} 的专家 → 实际计算量差不多!")
