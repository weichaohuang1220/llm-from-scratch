"""
========================================================================
第7章: MiniMindModel + MiniMindForCausalLM — 完整模型
========================================================================

现在把所有层堆叠起来，组成完整的语言模型。

模型分两层封装:
  MiniMindModel         — "骨架": 词嵌入 + N层Block + 最终归一化
  MiniMindForCausalLM   — "壳":   骨架 + LM Head(词表投影) + 损失计算

这种分层设计的好处:
  骨架可以复用于不同任务（分类、生成、嵌入等）
  只需要换最外层的"壳"就行

一个 token 的完整旅程:
  token_id (如 42)
    ↓ Embedding
  嵌入向量 (768维)
    ↓ Dropout
  带噪声的嵌入
    ↓ Block_0 (Attention + FFN)
    ↓ Block_1
    ↓ ...
    ↓ Block_7
  最终隐状态 (768维)
    ↓ RMSNorm (最终归一化)
  归一化后的隐状态
    ↓ lm_head (线性投影: 768 → 6400)
  logits (6400维, 词表中每个词的分数)
    ↓ softmax
  概率分布 (6400维, 加起来=1)
    ↓ argmax 或 采样
  下一个 token_id

权重共享 (Weight Tying):
  embed_tokens.weight 和 lm_head.weight 是同一个矩阵!
  - 嵌入层: token_id → 查表 → 768维向量 (矩阵的第 i 行)
  - LM Head: 768维向量 × 矩阵^T → 6400维 logits
  共享的好处: 参数量减半, 嵌入空间和输出空间对齐, 效果更好

损失函数 — 交叉熵:
  语言模型训练 = "完形填空": 给定前文, 预测下一个词
  对于序列 [A, B, C, D]:
    输入 [A, B, C] → 预测 [B, C, D]
    loss = -log P(B|A) - log P(C|A,B) - log P(D|A,B,C)
  这就是交叉熵损失, 越小表示预测越准
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from new.O1_rmsnorm import RMSNorm
from new.O2_rope import precompute_freqs_cis
from new.O5_moe import MOEFeedForward
from new.O6_transformer_block import MiniMindBlock


# ========================================================================
# Config — 模型超参数
# ========================================================================
class MiniMindConfig(PretrainedConfig):
    """
    模型配置类, 存储所有超参数。
    继承 PretrainedConfig 是为了兼容 HuggingFace 生态 (保存/加载/推送到Hub)。
    """
    model_type = "minimind"  # HuggingFace 用这个标识模型类型

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)

        # ---- 核心架构参数 ----
        self.hidden_size = hidden_size            # 隐藏层维度, 模型的"宽度"
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数, 模型的"深度"
        self.use_moe = use_moe                    # 是否使用 MoE

        # ---- 通用参数 ----
        self.dropout = kwargs.get("dropout", 0.0)          # Dropout 概率
        self.vocab_size = kwargs.get("vocab_size", 6400)   # 词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 序列起始 token ID
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 序列结束 token ID
        self.flash_attn = kwargs.get("flash_attn", True)   # 是否使用 Flash Attention

        # ---- 注意力参数 ----
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)     # Q 头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)     # KV 头数 (GQA)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # head_dim = 768 / 8 = 96

        # ---- FFN 参数 ----
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # intermediate_size: FFN 的中间维度
        # MiniMind 的独特计算: ceil(hidden_size × π / 64) × 64
        # 保证是 64 的倍数（GPU 友好）, π 是作者的一个有趣选择
        self.intermediate_size = kwargs.get(
            "intermediate_size",
            math.ceil(hidden_size * math.pi / 64) * 64
        )

        # ---- RoPE 参数 ----
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 频率基数

        # ---- YaRN 长度外推 (可选, 暂时忽略) ----
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = None

        # ---- MoE 参数 (仅 use_moe=True 时生效) ----
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# ========================================================================
# MiniMindModel — 模型骨架
# ========================================================================
class MiniMindModel(nn.Module):
    """
    Transformer 骨架: Embedding → N × Block → Final Norm

    不包含 LM Head (词表投影), 只输出最终隐状态。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # ---- 词嵌入层 ----
        # 一个 (vocab_size, hidden_size) 的查找表
        # 输入 token_id → 输出对应行的向量
        # 例: vocab_size=6400, hidden_size=768 → 参数量=6400×768=4,915,200
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # ---- Embedding Dropout ----
        self.dropout = nn.Dropout(config.dropout)

        # ---- N 层 Transformer Block ----
        # 这就是模型的"主体", 所有的"思考"都发生在这里
        self.layers = nn.ModuleList([
            MiniMindBlock(layer_id, config)
            for layer_id in range(self.num_hidden_layers)
        ])

        # ---- 最终归一化 ----
        # 所有 Block 之后再做一次 RMSNorm, 稳定输出
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ---- 预计算 RoPE 的 cos/sin ----
        # 这些值是固定的, 注册为 buffer (随模型保存但不作为参数训练)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
        )
        # register_buffer: 不是参数 (不需要梯度), 但会随模型 .to(device) 移动设备
        # persistent=False: 不保存到 state_dict (因为可以重新计算)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        """
        参数:
            input_ids: (batch, seq_len) — token ID 序列
            attention_mask: (batch, seq_len) — padding mask
            past_key_values: 每层的 KV Cache 列表
            use_cache: 是否返回 KV Cache

        返回:
            hidden_states: (batch, seq_len, hidden_size) — 最终隐状态
            presents: KV Cache 列表
            aux_loss: MoE 辅助损失之和
        """
        batch_size, seq_length = input_ids.shape

        # ---- 处理 KV Cache ----
        # HuggingFace 有时传入特殊格式的 cache, 我们重置为简单列表
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算起始位置 (用于 RoPE 和 KV Cache)
        # 如果有 KV Cache, 说明之前已经处理过一些 token
        # start_pos 就是之前处理过的 token 数量
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # ======== 第1步: 词嵌入 ========
        # input_ids: (batch, seq_len) 整数
        # → embed_tokens: 查表 → (batch, seq_len, hidden_size) 浮点向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ======== 第2步: 获取位置编码 ========
        # 从预计算的 cos/sin 中截取当前位置范围
        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_length],
            self.freqs_sin[start_pos: start_pos + seq_length]
        )

        # ======== 第3步: 逐层通过 Transformer Block ========
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # hidden_states 经过 N 层变换, 现在包含了丰富的上下文信息

        # ======== 第4步: 最终归一化 ========
        hidden_states = self.norm(hidden_states)

        # ======== 第5步: 收集 MoE 辅助损失 ========
        # 把所有 MoE 层的 aux_loss 加起来
        aux_loss = sum(
            [layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()  # 初始值为 0 (在正确的设备上)
        )

        return hidden_states, presents, aux_loss


# ========================================================================
# MiniMindForCausalLM — 因果语言模型
# ========================================================================
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    完整的因果语言模型 = 骨架 + LM Head + 损失计算

    继承:
      PreTrainedModel: HuggingFace 基类, 提供 save/load/push_to_hub 等功能
      GenerationMixin: 提供 generate() 方法的基础框架
    """
    config_class = MiniMindConfig  # 告诉 HuggingFace 用哪个配置类

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        # Transformer 骨架
        self.model = MiniMindModel(self.config)

        # LM Head: 线性投影, 把隐状态映射到词表空间
        # hidden_size → vocab_size (如 768 → 6400)
        # 输出的每个维度对应词表中一个词的"分数" (logit)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # ---- 权重共享 (Weight Tying) ----
        # 让 embed_tokens 和 lm_head 共享同一个权重矩阵
        # embed_tokens: (vocab_size, hidden_size) — 输入时: token_id → 第i行
        # lm_head:      (vocab_size, hidden_size) — 输出时: hidden × weight^T → logits
        # 共享好处:
        #   1. 参数量减半 (6400×768 = 4.9M 参数不用存两份)
        #   2. 语义一致: 嵌入空间和输出空间在同一个空间中
        #   3. 实践中效果更好
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """
        参数:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) — 训练时的目标 token, 推理时为 None
            logits_to_keep: 只计算最后 N 个位置的 logits (节省显存)
                           推理时只需要最后一个 token 的 logits

        返回:
            MoeCausalLMOutputWithPast: 包含 loss, aux_loss, logits, past_key_values
        """

        # ======== 第1步: 通过 Transformer 骨架 ========
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # ======== 第2步: LM Head → logits ========
        # logits_to_keep: 只对最后几个位置计算 logits
        # 推理时 logits_to_keep=0 → 只取最后一个位置 (只需要预测下一个词)
        # 训练时 logits_to_keep=全部 → 计算所有位置的 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # logits 形状: (batch, kept_positions, vocab_size)

        # ======== 第3步: 计算损失 (仅训练时) ========
        loss = None
        if labels is not None:
            # 语言模型的损失: 预测下一个 token
            # 输入: [A, B, C, D] → 预测目标: [B, C, D, <eos>]
            # logits[:-1] 对应位置 A,B,C 的预测
            # labels[1:]  对应目标 B,C,D

            x = logits[..., :-1, :].contiguous()   # 去掉最后一个位置 (没有目标)
            y = labels[..., 1:].contiguous()        # 去掉第一个位置 (没有预测)
            # x: (batch, seq_len-1, vocab_size) — 预测分布
            # y: (batch, seq_len-1)             — 真实 token_id

            # 交叉熵损失
            # view(-1, ...): 展平 batch 和 seq_len 维度
            # ignore_index=-100: 忽略 label 为 -100 的位置 (如 padding 或不需要计算损失的位置)
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),  # (batch×(seq_len-1), vocab_size)
                y.view(-1),              # (batch×(seq_len-1),)
                ignore_index=-100
            )

        # ======== 返回结构化输出 ========
        return MoeCausalLMOutputWithPast(
            loss=loss,                         # 语言模型损失
            aux_loss=aux_loss,                 # MoE 辅助损失
            logits=logits,                     # 词表上的分数
            past_key_values=past_key_values,   # KV Cache
            hidden_states=hidden_states        # 最终隐状态
        )


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    # 创建一个小模型
    config = MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=100,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        dropout=0.0,
        flash_attn=False,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 验证权重共享
    print(f"\n权重共享验证:")
    print(f"  embed_tokens.weight is lm_head.weight: "
          f"{model.model.embed_tokens.weight is model.lm_head.weight}")

    # 训练模式: 带 labels 计算损失
    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()  # 自回归: labels = input_ids (错位在 forward 内部处理)
    outputs = model(input_ids, labels=labels)
    print(f"\n训练模式:")
    print(f"  input_ids 形状: {input_ids.shape}")
    print(f"  logits 形状:    {outputs.logits.shape}")
    print(f"  loss:           {outputs.loss.item():.4f}")
    print(f"  理论随机损失:    {math.log(100):.4f} (= ln(vocab_size))")
    print(f"  → 未训练模型的 loss 应该接近 ln(vocab_size)，因为预测相当于随机猜")

    # 推理模式: 不带 labels
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        print(f"\n推理模式:")
        print(f"  logits 形状: {outputs.logits.shape}")
        print(f"  KV Cache 层数: {len(outputs.past_key_values)}")
        print(f"  loss: {outputs.loss}")  # None
