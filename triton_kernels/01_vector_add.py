"""
Triton 算子入门 #1: 向量加法
学习目标: 理解 Triton 的基本编程模型 - program_id, block, mask
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,      # 输入向量 x 的指针
    y_ptr,      # 输入向量 y 的指针
    output_ptr, # 输出向量的指针
    n,          # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 block 处理的元素数量
):
    # 1. 获取当前 program 的 ID（类似 CUDA 的 blockIdx）
    pid = tl.program_id(0)

    # 2. 计算当前 block 负责处理的元素索引
    #    比如 pid=0 处理 [0,1,...,255], pid=1 处理 [256,257,...,511]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. 边界检查：最后一个 block 可能超出向量长度
    mask = offsets < n

    # 4. 从显存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. 计算并写回显存
    tl.store(output_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Triton 向量加法的 wrapper 函数"""
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024

    # 计算需要多少个 block（grid）
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # 启动 kernel
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    # 测试正确性
    n = 10000
    x = torch.rand(n, device='cuda')
    y = torch.rand(n, device='cuda')

    # Triton 版本
    output_triton = triton_add(x, y)

    # PyTorch 版本（对照）
    output_torch = x + y

    # 验证结果一致
    print(f"向量长度: {n}")
    print(f"最大误差: {(output_triton - output_torch).abs().max().item():.10f}")
    print(f"结果一致: {torch.allclose(output_triton, output_torch)}")

    # 性能对比
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(12, 25)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'PyTorch'],
            styles=[('blue', '-'), ('red', '-')],
            ylabel='GB/s',
            plot_name='vector-add-performance',
            args={},
        )
    )
    def benchmark(size, provider):
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x, y), quantiles=quantiles)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True)
