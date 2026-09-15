"""
========================================================================
第8章: 文本生成 — Temperature, Top-K, Top-P, Repetition Penalty
========================================================================

模型训练好了, 怎么用它生成文本？

核心循环 (自回归生成):
  1. 输入 prompt 的 token_ids
  2. 模型输出最后一个位置的 logits (vocab_size 维的分数)
  3. 从 logits 中采样一个 next_token
  4. 把 next_token 拼接到输入后面
  5. 重复 2-4, 直到生成 <eos> 或达到最大长度

关键问题: 第3步怎么从 logits 中选 token？

方法1: Greedy (贪心) — 总是选概率最高的
  问题: 输出单调重复, "安全但无聊"

方法2: Temperature 采样
  logits = logits / temperature
  - temperature < 1: 分布变尖锐, 更确定 → 接近贪心
  - temperature = 1: 原始分布
  - temperature > 1: 分布变平坦, 更随机 → 更有创造性
  直觉: 温度高 → 分子运动剧烈 → 更混乱/随机 (借用物理学概念)

方法3: Top-K 采样
  只保留概率最高的 K 个 token, 其余设为 -inf (softmax 后变为 0)
  例: K=50, 词表6400个词, 只从最可能的50个中选

方法4: Top-P (Nucleus) 采样
  按概率从高到低排序, 累积概率超过 P 后, 后面的全部砍掉
  例: P=0.9, 排序后 [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02, ...]
      累积:      [0.3, 0.55, 0.75, 0.9, ...] → 0.9 处截断, 只保留前4个
  优势: 自适应! 当模型很确定时(一个词概率0.95), 只保留很少的词;
        当模型不确定时(概率分散), 保留更多候选

方法5: Repetition Penalty (重复惩罚)
  如果一个 token 之前已经出现过, 它的 logit 除以 penalty (>1)
  这样重复词的概率降低, 减少 "无限循环" 的问题

MiniMind 默认参数: temperature=0.85, top_p=0.85, top_k=50, repetition_penalty=1.0

KV Cache 加速:
  没有 KV Cache: 每步都要重新计算所有 token 的 K/V → O(n²) 总计算量
  有 KV Cache:   缓存历史 K/V, 每步只算新 token 的 K/V → O(n) 总计算量
  对于长序列生成, 这是几十倍的加速!
"""

import torch
import torch.nn.functional as F


@torch.inference_mode()  # 关闭梯度计算, 节省显存, 加速推理
def generate(model, input_ids, attention_mask=None,
             max_new_tokens=512, temperature=0.85, top_p=0.85, top_k=50,
             eos_token_id=2, use_cache=True, repetition_penalty=1.0,
             do_sample=True):
    """
    自回归文本生成

    参数:
        model: MiniMindForCausalLM 模型
        input_ids: (batch, prompt_len) — 初始 token 序列
        max_new_tokens: 最多生成多少个新 token
        temperature: 采样温度 (越高越随机)
        top_p: Nucleus 采样阈值
        top_k: Top-K 采样的 K 值
        eos_token_id: 结束 token 的 ID, 生成到这个就停
        use_cache: 是否使用 KV Cache
        repetition_penalty: 重复惩罚系数 (1.0=不惩罚)
        do_sample: True=采样, False=贪心

    返回:
        input_ids: (batch, prompt_len + generated_len) — 完整序列
    """

    past_key_values = None
    # finished 标记: 每个样本是否已生成 <eos>
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for step in range(max_new_tokens):

        # ======== 第1步: 确定本次输入 ========
        # 有 KV Cache 时, 只需要输入新 token (历史信息在 cache 里)
        # 没有 cache 时, 输入完整序列
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        current_input = input_ids[:, past_len:]
        # 第一步: current_input = 完整 prompt
        # 后续步: current_input = 上一步生成的 1 个 token

        # ======== 第2步: 模型前向传播 ========
        outputs = model(
            current_input,
            attention_mask,
            past_key_values,
            use_cache=use_cache
        )

        # 更新 attention_mask (新增一个位置)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones(attention_mask.shape[0], 1)
            ], dim=-1)

        # 取最后一个位置的 logits (这就是对下一个 token 的预测)
        logits = outputs.logits[:, -1, :]
        # logits 形状: (batch, vocab_size)

        # ======== 第3步: Temperature 缩放 ========
        # logits / temperature:
        #   temperature=0.85 → logits × 1.18 → 分布稍微变尖锐 → 更倾向高概率词
        #   temperature=1.5  → logits × 0.67 → 分布变平坦 → 更随机
        logits = logits / temperature

        # ======== 第4步: Repetition Penalty ========
        # 对已经出现过的 token, 降低其 logit
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                # 找出当前样本中已出现的所有 unique token
                seen_tokens = torch.unique(input_ids[i])
                # 对应 logit 除以 penalty (>1 会降低概率)
                logits[i, seen_tokens] /= repetition_penalty

        # ======== 第5步: Top-K 过滤 ========
        # 只保留概率最高的 K 个, 其余设为 -inf
        if top_k > 0:
            # torch.topk: 找出最大的 K 个值
            top_k_values = torch.topk(logits, top_k)[0]  # (batch, K)
            # top_k_values[..., -1, None]: 第 K 大的值 (阈值)
            threshold = top_k_values[..., -1, None]
            # 小于阈值的全部设为 -inf
            logits[logits < threshold] = -float('inf')

        # ======== 第6步: Top-P (Nucleus) 过滤 ========
        if top_p < 1.0:
            # 按概率从大到小排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 计算累积概率
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # 标记累积概率超过 top_p 的位置
            mask = cumulative_probs > top_p
            # 关键: 保留第一个超过阈值的 token (否则可能全被过滤)
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0  # 第一个永远保留

            # 把 mask 映射回原始顺序, 并设为 -inf
            # scatter: 把排序后的 mask "散射" 回原始位置
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

        # ======== 第7步: 采样或贪心选择 ========
        if do_sample:
            # 概率采样: 按 softmax 概率随机选一个
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # multinomial: 按概率分布抽样, 概率高的更容易被选中
        else:
            # 贪心: 直接选概率最大的
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        # next_token: (batch, 1)

        # ======== 第8步: 处理已结束的序列 ========
        # 如果某个样本已生成过 <eos>, 后续全部填 eos (不再生成新内容)
        if eos_token_id is not None:
            next_token = torch.where(
                finished.unsqueeze(-1),
                next_token.new_full((next_token.shape[0], 1), eos_token_id),
                next_token
            )

        # ======== 第9步: 拼接到输入序列 ========
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 更新 KV Cache
        past_key_values = outputs.past_key_values if use_cache else None

        # ======== 第10步: 检查是否全部结束 ========
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break  # 所有样本都生成了 <eos>, 提前退出

    return input_ids


# ========================================================================
# 动手验证
# ========================================================================
if __name__ == "__main__":
    from new.O7_full_model import MiniMindConfig, MiniMindForCausalLM

    # 创建小模型 (随机权重, 生成的内容没有意义, 但可以验证流程)
    config = MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        vocab_size=100,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        dropout=0.0,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config)
    model.eval()

    # 模拟一个 prompt
    prompt = torch.tensor([[1, 10, 20, 30]])  # batch=1, 4个token
    print("Prompt:", prompt)

    # 贪心生成
    greedy_output = generate(model, prompt.clone(), max_new_tokens=20,
                             do_sample=False, eos_token_id=2)
    print(f"\n贪心生成 ({greedy_output.shape[1] - 4} tokens):")
    print(f"  {greedy_output[0].tolist()}")

    # 采样生成 (多次运行结果不同)
    sample_output = generate(model, prompt.clone(), max_new_tokens=20,
                             temperature=0.85, top_p=0.85, top_k=50,
                             do_sample=True, eos_token_id=2)
    print(f"\n采样生成 ({sample_output.shape[1] - 4} tokens):")
    print(f"  {sample_output[0].tolist()}")

    # 对比不同 temperature
    print("\n--- Temperature 对比 (同一 prompt, 不同温度) ---")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        out = generate(model, prompt.clone(), max_new_tokens=10,
                       temperature=temp, do_sample=True, eos_token_id=None)
        print(f"  temp={temp}: {out[0, 4:].tolist()}")
    print("  温度越低 → 输出越确定(接近贪心); 温度越高 → 输出越随机")
