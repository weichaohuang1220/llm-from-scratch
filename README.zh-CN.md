<div align="center">

[English](./README.md) | **简体中文**

</div>

# LLM From Scratch

> 作者：黄威朝 (weichaohuang1220@gmail.com)

从零逐章实现现代 decoder-only 大语言模型的核心组件，使用 PyTorch 编写，附带详细的讲解注释。
每个文件都独立可运行：先讲清楚这个组件解决什么问题，再推导公式，最后动手实现。

同时包含使用 Triton 编写自定义 GPU 算子的入门内容。

## 致谢

这些笔记是在研读 **[@jingyaogong](https://github.com/jingyaogong)** 的
**[MiniMind](https://github.com/jingyaogong/minimind)** 项目过程中写成的——这是一个 64M 参数的中文
大语言模型，完整实现了从预训练到偏好对齐的训练链路。本仓库中分析的架构选型（GQA 的头数配比、
SwiGLU 的 intermediate_size 计算、RoPE 的 base 频率、MoE 路由策略）均参照 MiniMind 的设计，
建议对照原项目一起阅读。

MiniMind 以 Apache License 2.0 开源。本仓库中的代码是我为学习目的独立编写的复现与讲解，
并非 MiniMind 源码的拷贝或 fork。

---

## Transformer 组件

建议按顺序阅读——每一章都建立在上一章的基础上，第 7 章把前面所有积木组装成完整模型。

| # | 文件 | 主题 | 行数 |
|---|------|------|------|
| 1 | [`transformer/01_rmsnorm.py`](transformer/01_rmsnorm.py) | RMSNorm — 现代 LLM 为什么抛弃 LayerNorm，以及混合精度下的升降精度技巧 | 61 |
| 2 | [`transformer/02_rope.py`](transformer/02_rope.py) | RoPE — 旋转位置编码，以及 Attention 为什么需要位置信息 | 67 |
| 3 | [`transformer/03_attention.py`](transformer/03_attention.py) | 多头注意力、因果掩码、GQA 分组查询注意力、KV Cache | 303 |
| 4 | [`transformer/04_feedforward.py`](transformer/04_feedforward.py) | SwiGLU 前馈网络，以及 intermediate_size 的选择 | 141 |
| 5 | [`transformer/05_moe.py`](transformer/05_moe.py) | MoE 混合专家 — 路由、top-k 门控与稀疏激活 | 218 |
| 6 | [`transformer/06_transformer_block.py`](transformer/06_transformer_block.py) | 把前面的组件组装成 Transformer 解码器层 | 197 |
| 7 | [`transformer/07_full_model.py`](transformer/07_full_model.py) | 完整模型 — 词嵌入、堆叠层、最终归一化与因果语言模型头 | 371 |
| 8 | [`transformer/08_generate.py`](transformer/08_generate.py) | 文本生成 — Temperature、Top-K、Top-P、重复惩罚 | 232 |

共 **1,590 行** 实现与讲解。

## Triton 算子

用 [Triton](https://triton-lang.org/) 编写的自定义 GPU 算子，从编程模型讲到真实的 reduction 操作。

| # | 文件 | 主题 | 行数 |
|---|------|------|------|
| 1 | [`triton_kernels/01_vector_add.py`](triton_kernels/01_vector_add.py) | Triton 编程模型 — `program_id`、block、mask | 92 |
| 2 | [`triton_kernels/02_softmax.py`](triton_kernels/02_softmax.py) | Reduction 操作（求最大值、求和）与数值稳定性 | 135 |

## 学习笔记

`notes/` 存放与代码同期整理的学习与准备材料——训练全链路
（预训练 → SFT → LoRA → DPO/GRPO）的梳理，以及架构要点的浓缩复习。

---

## 运行代码

```bash
# 环境要求：Python 3.10+、PyTorch 2.x
pip install torch

# 每一章都可独立运行，并打印一个完整示例
python transformer/01_rmsnorm.py
python transformer/03_attention.py
python transformer/08_generate.py
```

Triton 算子需要 CUDA GPU：

```bash
pip install triton
python triton_kernels/01_vector_add.py
python triton_kernels/02_softmax.py
```

---

## 这个项目涵盖什么

* **归一化** — 为什么 pre-norm 优于 post-norm，以及 RMSNorm 为什么取代了 LayerNorm
* **位置编码** — RoPE 解决的问题，以及旋转如何编码相对位置
* **注意力机制** — 缩放点积注意力、因果掩码、GQA 的 KV 头共享，以及 KV Cache 为什么让推理可行
* **前馈网络** — SwiGLU 门控变体，以及 intermediate_size 的确定方式
* **稀疏化** — MoE 路由，以及为什么每个 token 只激活一小部分参数
* **文本生成** — 采样参数如何真正改变输出分布
* **GPU 算子** — Triton 编程模型，以及如何写出数值稳定的 reduction

这些笔记所参照的项目见 [致谢](#致谢)。
