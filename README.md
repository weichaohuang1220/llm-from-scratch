<div align="center">

**English** | [简体中文](./README.zh-CN.md)

</div>

# LLM From Scratch

> Author: Weichao Huang (weichaohuang1220@gmail.com)

A chapter-by-chapter, from-scratch implementation of the components that make up a modern
decoder-only language model, written in PyTorch with detailed teaching commentary. Each file is
self-contained and runnable: it explains the problem the component solves, derives the formula, and
then implements it.

Also included: an introduction to writing custom GPU kernels in Triton.

> The commentary is written in Chinese. The code, formulas, and structure read the same in any
> language.

---

## Transformer Components

Read them in order — each chapter builds on the previous one, and chapter 7 assembles everything
into a complete model.

| # | File | Topic | Lines |
|---|------|-------|-------|
| 1 | [`transformer/01_rmsnorm.py`](transformer/01_rmsnorm.py) | RMSNorm — why modern LLMs dropped LayerNorm, and the mixed-precision upcast trick | 61 |
| 2 | [`transformer/02_rope.py`](transformer/02_rope.py) | RoPE — rotary position embedding, and why attention needs positional information at all | 67 |
| 3 | [`transformer/03_attention.py`](transformer/03_attention.py) | Multi-head attention, causal masking, GQA (grouped-query attention), and KV caching | 303 |
| 4 | [`transformer/04_feedforward.py`](transformer/04_feedforward.py) | SwiGLU feed-forward networks and intermediate-size selection | 141 |
| 5 | [`transformer/05_moe.py`](transformer/05_moe.py) | Mixture of Experts — routing, top-k gating, and sparse activation | 218 |
| 6 | [`transformer/06_transformer_block.py`](transformer/06_transformer_block.py) | Assembling a decoder block from the pieces above | 197 |
| 7 | [`transformer/07_full_model.py`](transformer/07_full_model.py) | The full model — embeddings, stacked blocks, final norm, and the causal-LM head | 371 |
| 8 | [`transformer/08_generate.py`](transformer/08_generate.py) | Text generation — temperature, top-k, top-p, and repetition penalty | 232 |

**1,590 lines** of implementation and commentary.

## Triton Kernels

Custom GPU kernels written in [Triton](https://triton-lang.org/), building up from the programming
model to a real reduction.

| # | File | Topic | Lines |
|---|------|-------|-------|
| 1 | [`triton_kernels/01_vector_add.py`](triton_kernels/01_vector_add.py) | The Triton programming model — `program_id`, blocks, and masking | 92 |
| 2 | [`triton_kernels/02_softmax.py`](triton_kernels/02_softmax.py) | Reductions (max, sum) and numerical stability in softmax | 135 |

## Notes

`notes/` holds study and preparation material written alongside the code — walkthroughs of the
training pipeline (pretraining → SFT → LoRA → DPO/GRPO) and a condensed review of the architecture.

---

## Running the Code

```bash
# Requirements: Python 3.10+, PyTorch 2.x
pip install torch

# Each chapter runs standalone and prints a worked example
python transformer/01_rmsnorm.py
python transformer/03_attention.py
python transformer/08_generate.py
```

Triton kernels need a CUDA GPU:

```bash
pip install triton
python triton_kernels/01_vector_add.py
python triton_kernels/02_softmax.py
```

---

## What This Covers

* **Normalization** — why pre-norm beats post-norm, and why RMSNorm replaced LayerNorm
* **Positional encoding** — the problem RoPE solves and how rotation encodes relative position
* **Attention** — scaled dot-product attention, causal masks, GQA's KV-head sharing, and why KV
  caching makes inference tractable
* **Feed-forward** — the SwiGLU gated variant and how intermediate size is chosen
* **Sparsity** — MoE routing and why only a fraction of parameters activate per token
* **Generation** — how sampling parameters actually change the output distribution
* **GPU kernels** — the Triton programming model and writing a numerically stable reduction

The architectural choices analyzed throughout follow the design of
[MiniMind](https://github.com/jingyaogong/minimind), a 64M-parameter Chinese LLM, which these notes
were written while studying.
