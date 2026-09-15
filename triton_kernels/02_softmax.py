"""
Triton 算子 #2: Softmax
学习目标: 理解 reduction 操作（求最大值、求和）和数值稳定性
Softmax 是 Transformer 注意力机制的核心组件
公式: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,             # 每行的列数
    input_row_stride,   # 输入矩阵每行的步长
    output_row_stride,  # 输出矩阵每行的步长
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行
    row_idx = tl.program_id(0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 生成列索引 [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # ========== Step 1: 加载一整行数据 ==========
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))

    # ========== Step 2: 数值稳定性 - 减去最大值 ==========
    # 直接算 exp(x) 会溢出，所以先减去 max(x)
    # 这不会改变 softmax 的结果，但避免了数值爆炸
    row_max = tl.max(row, axis=0)
    row_stable = row - row_max

    # ========== Step 3: 计算 exp ==========
    numerator = tl.exp(row_stable)

    # ========== Step 4: 求和 ==========
    denominator = tl.sum(numerator, axis=0)

    # ========== Step 5: 归一化 ==========
    softmax_output = numerator / denominator

    # ========== Step 6: 写回结果 ==========
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton Softmax 的 wrapper"""
    assert x.dim() == 2, "输入必须是 2D 矩阵"
    rows, cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE 必须是 2 的幂且 >= cols
    BLOCK_SIZE = triton.next_power_of_2(cols)

    # 每行一个 program
    grid = (rows,)

    softmax_kernel[grid](
        x, output,
        cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    # ========== 正确性验证 ==========
    print("=" * 50)
    print("Softmax 正确性验证")
    print("=" * 50)

    x = torch.randn(128, 256, device='cuda')

    output_triton = triton_softmax(x)
    output_torch = torch.softmax(x, dim=1)

    print(f"矩阵大小: {x.shape}")
    print(f"最大误差: {(output_triton - output_torch).abs().max().item():.10f}")
    print(f"结果一致: {torch.allclose(output_triton, output_torch, atol=1e-6)}")

    # 验证 softmax 的性质：每行之和 = 1
    row_sums = output_triton.sum(dim=1)
    print(f"行和范围: [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}] (应为 1.0)")

    # ========== 数值稳定性测试 ==========
    print(f"\n{'=' * 50}")
    print("数值稳定性测试")
    print("=" * 50)

    # 用很大的数测试，不做 max 减法的话 exp() 会溢出
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
    output_large = triton_softmax(x_large)
    print(f"输入: {x_large}")
    print(f"Softmax 输出: {output_large}")
    print(f"输出之和: {output_large.sum().item():.6f} (应为 1.0)")

    # ========== 性能对比 ==========
    print(f"\n{'=' * 50}")
    print("性能对比: Triton vs PyTorch")
    print("=" * 50)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # 列数
            x_vals=[128 * i for i in range(2, 50)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'PyTorch'],
            styles=[('blue', '-'), ('red', '-')],
            ylabel='GB/s',
            plot_name='softmax-performance',
            args={'M': 4096},  # 固定 4096 行
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True)
