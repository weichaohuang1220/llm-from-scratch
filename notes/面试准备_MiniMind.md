# MiniMind 面试准备材料

> 针对黄威朝简历中"MiniMind - 全链路轻量级大模型复现"项目的面试准备
> 64M 参数中文 LLM，涵盖预训练 / SFT / LoRA / DPO / GRPO

---

## 一、项目介绍话术（三个版本）

### 版本1：30秒电梯演讲

> "我基于 MiniMind 开源框架，独立复现了一个 64M 参数的中文大语言模型，完整走完了预训练、SFT 监督微调、LoRA 轻量微调、以及 DPO 和 GRPO 偏好对齐的完整流程。过程中我手撕了 Transformer 的核心组件，包括 RMSNorm、RoPE、GQA 分组注意力、SwiGLU 和 MoE，对从数据处理到推理部署的 LLM 全栈技术有了系统性的理解。"

### 版本2：2分钟详细介绍（STAR）

**Situation 背景：**
"我想系统性地掌握大模型从0到1的训练流程，而不是只停留在调用 API 层面。MiniMind 是一个很好的学习载体——它用 64M 参数就能跑通完整的 LLM 训练链路，适合在单卡上实验。"

**Task 目标：**
"目标是独立完成一个能进行中文多轮对话、符合人类偏好的小型语言模型，并深入理解每个环节背后的原理。"

**Action 行动：**

1. **模型架构复现**：从零实现了模型的核心模块——RMSNorm 归一化、RoPE 旋转位置编码、支持 GQA 的多头注意力、SwiGLU 前馈网络、以及可选的 MoE 混合专家层。

2. **预训练阶段**：用中文语料做自回归预训练，采用 bfloat16 混合精度、AdamW 优化器、梯度累积等工程技巧，让模型学会语言建模。

3. **SFT 阶段**：用多轮对话数据做监督微调，理解了 SFT 数据和预训练数据的区别——SFT 只对 assistant 回答部分计算 loss，prompt 部分用 -100 忽略。

4. **LoRA 微调**：学习了低秩分解的思想，只训练约 0.5% 的参数就能达到接近全参微调的效果。

5. **偏好对齐**：对比了 DPO 和 GRPO 两种方案——DPO 是离线的、基于 pairwise 数据直接优化；GRPO 是在线强化学习，通过组内 reward 归一化替代了传统 PPO 中的 critic 模型。

**Result 结果：**
"最终模型能进行多轮中文对话，我对 LLM 整个技术栈（数据、架构、训练、对齐、推理）建立了清晰的心智模型，也理解了每一步的工程取舍。"

### 版本3：技术深度版（面试官追问时）

"我把项目分成三个抽象层次来理解——
- **底层是架构**：为什么 LLaMA 家族选 RMSNorm 而不是 LayerNorm、为什么用 RoPE 而不是绝对位置编码、为什么用 GQA 而不是 MHA。
- **中层是训练**：混合精度、梯度累积、warmup、cosine 学习率调度，这些是所有大模型训练的共性。
- **上层是对齐**：SFT → RLHF/DPO → GRPO 的演化逻辑——从"教会模型说话"到"让模型说我们想要的话"。
"

---

## 二、高频面经（必背）

### A. 架构类（15题）

**Q1. MiniMind 的模型架构和 LLaMA 有什么关系？**

A：MiniMind 基本是 LLaMA 架构的精简版。核心组件都一样：RMSNorm + RoPE + GQA + SwiGLU + Pre-Norm + 权重共享。主要区别是规模小（64M vs 7B）、词表小（6400）、层数少（8层）。

**Q2. 为什么用 RMSNorm 而不是 LayerNorm？**

A：
1. **计算更快**：省去了计算均值和减法的步骤，只保留缩放。
2. **效果相当**：实验表明去掉中心化对 Transformer 影响很小。
3. **公式简单**：`y = x / sqrt(mean(x²) + ε) × γ`，没有 β 偏置。
4. LLaMA、Qwen、DeepSeek 等主流模型都采用。

**Q3. RoPE 的核心思想是什么？为什么比绝对位置编码好？**

A：
- 核心思想：把位置信息编码为二维平面上的"旋转角度"。每个 head_dim 被拆成 dim/2 个二维对，每对按 `position × θ_k` 旋转。
- 优势：
  1. **相对位置天然可建模**：Q·K 点积的结果只依赖相对位置差，不依赖绝对位置。
  2. **外推性好**：配合 YaRN 等方法能外推到训练时没见过的长度。
  3. **无额外参数**：cos/sin 是预计算的，不占参数预算。

**Q4. 手撕 RoPE——怎么实现？**

A：
```python
# 预计算频率
freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
t = torch.arange(seq_len)
freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
cos, sin = torch.cos(freqs).repeat(1,2), torch.sin(freqs).repeat(1,2)

# 应用旋转
def rotate_half(x):
    return torch.cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)

q_rot = q * cos + rotate_half(q) * sin
```
关键理解 `rotate_half`：这是二维旋转矩阵 `[cos,-sin; sin,cos]` 的向量化等价实现。

**Q5. GQA 是什么？和 MHA、MQA 的区别？**

A：
- **MHA（Multi-Head Attention）**：Q、K、V 各有 N 个头。显存大，KV Cache 占用高。
- **MQA（Multi-Query Attention）**：Q 有 N 个头，K、V 只有 1 个头。极致省显存，但效果有损失。
- **GQA（Grouped-Query Attention）**：Q 有 N 个头，K、V 有 G 个头（1<G<N）。在 MHA 和 MQA 之间权衡。
- MiniMind 默认 8 个 Q 头、4 个 KV 头，每 2 个 Q 头共享 1 组 KV。
- **核心价值**：推理时 KV Cache 显存减半，速度更快，效果接近 MHA。

**Q6. GQA 在代码上怎么实现？**

A：用 `repeat_kv` 函数——把 KV 从 `(bs, seq, 4, dim)` 扩展为 `(bs, seq, 8, dim)`：
```python
x[:, :, :, None, :].expand(bs, seq, 4, 2, dim).reshape(bs, seq, 8, dim)
```
用 `expand` 不是 `repeat`——前者不复制内存，后者会。

**Q7. SwiGLU 为什么比 ReLU/GELU 好？**

A：
- SwiGLU = SiLU(x·W_gate) ⊙ (x·W_up) → W_down
- **门控机制**：gate 分支控制哪些维度重要，up 分支提供内容，两者相乘实现"软过滤"。
- **SiLU 比 ReLU 平滑**：SiLU(x) = x·sigmoid(x)，在 x<0 区域不是硬截断，梯度更好。
- **代价**：参数量比普通 FFN 多 50%（多了一个 gate 矩阵），所以 intermediate_size 一般设小一点。

**Q8. FFN 的 intermediate_size 一般怎么选？**

A：
- 标准 Transformer：`4 × hidden_size`
- LLaMA：`~2.67 × hidden_size`（用 SwiGLU 参数多，所以缩小）
- MiniMind：`ceil(hidden_size × π / 64) × 64`，π 是作者的一个有趣选择，×64 是对齐 GPU。

**Q9. 为什么 Pre-Norm 而不是 Post-Norm？**

A：
- Post-Norm（原始）：`x + Norm(SubLayer(x))` → 深层训练不稳定，需要 warmup。
- Pre-Norm（现代）：`x + SubLayer(Norm(x))` → 残差路径是"干净"的，梯度流更稳定，可以直接训练不需要 warmup。
- 现代大模型（GPT、LLaMA、Qwen）都用 Pre-Norm。

**Q10. MoE 是怎么工作的？为什么能省算力？**

A：
- 结构：一个 Router（门控）+ N 个专家（每个是独立的 FFN）。
- 流程：token 过 Router → softmax 得到每个专家的分数 → 选 Top-K 专家 → 只激活这些专家 → 加权求和。
- **核心收益**：参数量 = N × FFN，但计算量 = K × FFN（K<<N）。比如 N=8, K=2，参数变8倍但算力只变2倍。
- **代价**：需要额外的负载均衡机制，训练更不稳定。

**Q11. MoE 的辅助损失是干什么的？**

A：
- 问题：Router 可能退化到只选少数几个专家，其他专家"饿死"。
- 解决：加辅助损失 `aux_loss = Σ(load_i × importance_i) × N`
  - load_i：专家 i 被选中的比例
  - importance_i：专家 i 的平均路由概率
- 均匀分配时 aux_loss 最小，鼓励专家负载均衡。

**Q12. Flash Attention 的原理是什么？**

A：
- 标准 Attention：要存 N×N 的分数矩阵，显存 O(N²)。
- Flash Attention 的两个核心技巧：
  1. **Tiling（分块）**：把 Q、K、V 切块，在 SRAM 里做小矩阵乘法，不写回 HBM。
  2. **Online Softmax**：分块时实时更新 softmax 的分母和分子，避免存完整的分数矩阵。
- 结果：显存 O(N)，速度 2-4x 加速。
- PyTorch 2.0+ 的 `F.scaled_dot_product_attention` 自动用 Flash Attention。

**Q13. KV Cache 是干什么的？**

A：
- 推理时每生成一个 token，理论上要对所有历史 token 重新算一遍 K/V，O(N²) 总计算量。
- KV Cache 把历史 K/V 缓存下来，每步只算新 token 的 K/V → O(N) 总计算量。
- 显存代价：要存 `2 × N × num_layers × num_kv_heads × head_dim` 的张量，长序列时显存压力大。
- GQA 就是为了减小 KV Cache 设计的。

**Q14. 权重共享（Weight Tying）是什么？**

A：让词嵌入矩阵 `embed_tokens.weight` 和输出投影 `lm_head.weight` 共享同一个参数：
- 嵌入层是 `(vocab_size, hidden)` 的查找表。
- LM Head 是 `(hidden, vocab_size)` 的投影。
- 共享后参数量减半（MiniMind 词表6400 × 隐层768 ≈ 5M 参数），语义空间对齐，效果更好。

**Q15. 因果掩码是怎么实现的？**

A：在 attention scores 上加一个上三角为 `-inf` 的矩阵，softmax 后这些位置变成 0：
```python
mask = torch.full((seq_len, seq_len), -inf).triu(1)
scores += mask
```
语言模型必须是因果的——每个位置只能看自己和前面的 token，不能看未来。

---

### B. 训练类（10题）

**Q16. 预训练和 SFT 的数据有什么区别？Loss 计算有什么不同？**

A：
- **预训练数据**：无标注的连续文本（如书籍、网页），模型学语言建模。
- **SFT 数据**：`{prompt, response}` 对，通常是对话格式。
- **Loss 区别**：
  - 预训练：对所有 token 计算交叉熵 loss。
  - SFT：**只对 response 部分计算 loss**，prompt 部分的 label 设为 -100（PyTorch 的 `ignore_index` 机制会跳过）。
- 原因：SFT 的目标是学"怎么回答"，不是学"怎么提问"。

**Q17. 什么是 bfloat16 混合精度训练？**

A：
- 问题：全 fp32 训练显存占用大、速度慢；全 fp16 训练数值不稳定（范围小，容易溢出）。
- bfloat16：和 fp32 同样的指数范围（8位指数），但精度低（7位尾数）。范围大 → 训练稳定；尾数少 → 精度够用。
- 做法：
  - 前向和反向用 bfloat16（省显存，快）
  - 优化器状态（AdamW 的 m, v）用 fp32（保证数值精度）
  - 主权重用 fp32 存一份（更新时用）
- PyTorch 里用 `torch.autocast(dtype=torch.bfloat16)`。

**Q18. AdamW 和 Adam 的区别？**

A：
- Adam：weight decay 和梯度一起做（L2 正则化加到梯度上）。
- AdamW：weight decay 独立于梯度，直接在参数更新时减掉（解耦权重衰减）。
- AdamW 效果更好，是现代 Transformer 训练的标配。

**Q19. 什么是梯度累积？为什么要用？**

A：
- 目的：在显存有限的情况下模拟大 batch size。
- 做法：前向+反向 N 次，梯度累加不清零；第 N 次才做 optimizer.step() 和 zero_grad()。
- 等效 batch size = 物理 batch × 累积步数。
- 好处：大 batch 训练更稳定，效果更好。

**Q20. 学习率调度一般怎么做？**

A：典型的 warmup + cosine decay：
1. **Warmup**：前几百步学习率线性增长，防止初始大 LR 破坏预训练权重。
2. **Cosine decay**：按余弦曲线衰减到很小的值（如初始的 10%）。
3. 原因：训练初期权重随机，不稳定 → 慢启动；训练末期精细调优 → 小 LR。

**Q21. LoRA 的原理是什么？**

A：
- 核心假设：微调时权重的"变化量" ΔW 是低秩的。
- 做法：冻结原权重 W，训练 `ΔW = B × A`，其中 A 是 `(r, d)`、B 是 `(d, r)`，r<<d（如 r=8）。
- 前向：`output = xW + x(BA)`，推理时可以把 BA 合并回 W。
- 好处：
  - 参数量少（比如 0.5%）
  - 显存占用少（不需要存原权重的优化器状态）
  - 可以叠加多个 LoRA（不同任务切换）

**Q22. LoRA 的 rank r 怎么选？**

A：一般 r=8 或 r=16 够用。r 太大没意义（等价于全参微调），r 太小拟合不够。实践上先试 r=8，效果不够再加大。

**Q23. DPO 的核心思想是什么？和 RLHF/PPO 的区别？**

A：
- RLHF 三步走：SFT → 训练 Reward Model → PPO 用 RM 做在线 RL。
- DPO（Direct Preference Optimization）：数学推导证明 RLHF 目标可以直接用"policy 的 log 概率比"表达，**跳过显式的 Reward Model**。
- DPO Loss：
  ```
  L = -log σ(β × [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])
  ```
  其中 y_w 是 chosen（好回答），y_l 是 rejected（差回答）。
- 优势：训练简单（只需 pairwise 数据），稳定（没有在线采样的方差问题），不需要 RM。

**Q24. GRPO 又是什么？和 PPO、DPO 的关系？**

A：
- GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的 RL 方法，用于数学推理等可验证任务。
- 核心创新：**去掉 Critic（value model）**。
  - PPO：需要 actor（policy）+ critic（value function）两个模型。
  - GRPO：对每个 prompt 采样 G 条回答，用"组内相对 reward"（reward - group_mean）/ group_std 作为 advantage。
- 好处：显存省一半（不用 critic），适合强化推理能力。
- 典型应用：DeepSeek-R1 的推理训练。

**Q25. 预训练 → SFT → DPO/GRPO 这条链路，每一步解决什么问题？**

A：
- **预训练**：教模型"说话"（学会语言的统计规律）。
- **SFT**：教模型"对话"（学会指令跟随的格式）。
- **DPO/GRPO**：教模型"说人类喜欢的话"（对齐人类偏好，减少幻觉、提高有用性）。
- 三者是递进关系：预训练是基础能力，SFT 是格式规范，对齐是质量提升。

---

### C. 工程与细节（10题）

**Q26. 你的 64M 模型大概多少参数是嵌入层？**

A：
- 词表 6400 × 隐层 768 ≈ 4.9M
- 但由于权重共享（embed 和 lm_head 复用），实际只占 4.9M。
- 其他 ~59M 在 Transformer 层（8层 × 每层 ~7M）。

**Q27. tokenizer 是怎么训练的？**

A：MiniMind 用 BPE（Byte-Pair Encoding）训练了一个 6400 词表的中文 tokenizer，字表包含中英文字符、常见词。通过 `tokenizers` 库训练 → 存为 `tokenizer.json`。相比 GPT-4 的 100K+ 词表更小，适合小模型。

**Q28. 训练时你遇到过什么问题？**

A（诚实回答）：
- **Loss 不下降/震荡**：一般是 LR 太大，或者数据没 shuffle。
- **显存 OOM**：减小 batch size，加梯度累积；检查是不是开了 `retain_graph=True` 之类的。
- **推理结果乱码**：检查 tokenizer 的特殊 token 是否正确（bos、eos、pad）。
- **生成重复**：调 repetition_penalty 或降低 temperature。

**Q29. 推理时 temperature、top_p、top_k 怎么配合？**

A：
- **temperature**：缩放 logits。<1 更确定，>1 更随机。
- **top_k**：只保留概率最高的 K 个 token。
- **top_p（nucleus）**：累积概率达到 P 就截断，自适应候选集大小。
- 常用组合：`temperature=0.85, top_p=0.85, top_k=50`。这是比较均衡的"有创造性但不离谱"配置。

**Q30. 为什么 SFT 时 prompt 部分要 mask 掉 loss？**

A：
- SFT 的目标是让模型学会"在给定 prompt 时如何生成 response"。
- 如果 prompt 部分也算 loss，模型会浪费容量去"学习生成 prompt"，这没意义（prompt 是用户输入的，不需要预测）。
- 实现方式：把 prompt 位置的 label 设为 -100，`F.cross_entropy(..., ignore_index=-100)` 会跳过这些位置。

---

### D. 深度追问 & 开放题（5题）

**Q31. 如果让你优化 MiniMind，你会做什么？**

A（有话说的点）：
1. **长上下文**：当前 max_position=32K 但实际训练数据短，可以用 YaRN 做长度外推。
2. **推理加速**：用 PagedAttention（vLLM 的方案）优化 KV Cache 碎片化问题。
3. **量化部署**：INT8/INT4 量化，模型从 ~250MB 压到 60MB，CPU 也能跑。
4. **更好的数据**：SFT 数据质量比数量重要，可以用 LIMA 的思路做精筛。

**Q32. 为什么 Decoder-Only 架构打败了 Encoder-Decoder？**

A：
1. **简单**：只需一种 block，代码和训练都更简洁。
2. **通用**：通过 prompt 工程几乎能做所有任务（翻译、摘要、问答），不需要任务特定结构。
3. **Scaling Law 更好**：GPT-3 证明了纯 decoder 堆规模能涌现出惊人的能力。
4. **推理高效**：KV Cache 天然适配因果生成。

**Q33. Scaling Law 是什么？MiniMind 规模太小是否有局限？**

A：
- Scaling Law（Kaplan 2020、Chinchilla 2022）：模型性能随参数量、数据量、计算量呈幂律提升。
- Chinchilla 最优：参数量 N 和 token 数 D 应该 1:20，即 64M 模型至少喂 1.3B token 才训得"满"。
- MiniMind 64M 主要用来验证流程和学习原理，不追求 SOTA 效果——它确实有局限（如事实性、长程推理）。

**Q34. 大模型的幻觉问题怎么解？**

A：
- **数据层面**：清洗训练数据，去除错误信息。
- **训练层面**：DPO/RLHF 惩罚幻觉输出。
- **推理层面**：RAG 检索增强（用外部知识做 grounding），这也是你 ChatMind 项目做的事。
- **解码层面**：对比解码（contrastive decoding）、自一致性（self-consistency）。

**Q35. 你觉得 LLM 的下一步是什么？**

A（开放讨论，别讲得太绝对）：
- 多模态融合（原生多模态而非拼接）。
- 强推理能力（o1、R1 展示的 inference-time scaling）。
- Agent 方向（工具调用、自主规划，你 ChatMind 经验正好接上）。
- 推理效率（MoE、状态空间模型如 Mamba）。

---

## 三、容易挖坑的追问（防翻车）

### 坑1："你说理解了 RoPE，那 YaRN 你了解吗？"
**稳妥回答**："RoPE 我手撕过，基础版本原理和代码都清楚。YaRN 我读过论文，核心是对不同频率维度做不同的缩放因子，让高频维度保持原样、低频维度拉伸。但我还没在项目里实际调通 YaRN，只是理论上理解。"

### 坑2："你复现了 MoE，能讲讲 DeepSeek-MoE 的改进吗？"
**稳妥回答**："我实现的是标准 Top-K MoE。DeepSeek-MoE 的主要改进我了解两点：一是细粒度专家（更多小专家）加共享专家，二是更好的负载均衡损失。但具体实现细节我没有手写过，下一步想实验一下。"

### 坑3："你提到 DPO，那和 IPO、KTO 有什么区别？"
**稳妥回答**："我项目中主要跑通了 DPO。IPO 和 KTO 我看过论文摘要——IPO 解决 DPO 在偏好数据确定性很强时的过拟合问题；KTO 不需要 pairwise 数据，只需要 binary 的好/坏标注，数据收集更容易。细节我不敢说完全掌握。"

### 坑4："手撕 attention 代码"
**准备好的伪代码**：
```python
def attention(q, k, v, mask=None):
    # q, k, v: (B, H, T, D)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

### 坑5："Softmax 数值稳定性问题"
标准 softmax 会溢出：`exp(100)` 直接 overflow。
解决：`softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`，减去最大值不改变结果但避免溢出。PyTorch 的 softmax 默认就是稳定版本。

---

## 四、回答不上来时的话术模板

**诚实 + 学习意愿**：
> "这个点我没有深入研究过，但我猜测……（给出基于已有知识的合理推测），回去我会查论文确认。"

**框架化回答**：
> "这个问题我可以从三个层面想：原理上……、实现上……、工程上……。具体细节我可能说得不准，但大方向是这样。"

**承认边界**：
> "我这个项目主要聚焦在复现和理解基础原理，X 这个更前沿的方向我了解有限，希望以后有机会学习。"

**禁忌**：
- 不懂装懂，被追问两轮就暴露
- 堆术语但讲不清楚
- 说"我不知道"然后沉默（至少说说你的猜测）

---

## 五、反问面试官的问题（加分项）

技术深度：
1. "团队在模型训练中最大的挑战是什么？是算力、数据、还是算法？"
2. "你们在 SFT 和对齐阶段，DPO 和 PPO/GRPO 更倾向哪个？"
3. "业务上模型的瓶颈是推理速度还是效果？"

职业发展：
4. "新人加入后前3-6个月的典型成长路径是什么样的？"
5. "团队内部有什么技术分享/论文共读的机制吗？"

---

## 六、最后一天冲刺清单

- [ ] 能默写 RMSNorm、SwiGLU、RoPE 的公式
- [ ] 能讲清楚 MHA/GQA/MQA 的区别和 KV Cache
- [ ] 能讲清楚 SFT loss 为什么 mask prompt
- [ ] 能讲清楚 DPO 公式和直觉
- [ ] 能讲清楚 LoRA 的低秩分解
- [ ] 准备 2-3 个反问问题
- [ ] 把"项目介绍30秒版本"背到脱口而出

**心态**：你做了全链路复现，这个深度已经超过 90% 的简历所说"了解 LLM"的候选人。保持自信，遇到不会的坦诚说不会但给出思路，比乱编更能加分。

祝面试顺利！
