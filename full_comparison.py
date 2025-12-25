"""
完整的 API 对比：包含所有 shape
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import torch
import triton
import triton.language as tl
import triton.testing
from profiler_utils import KernelProfiler
import time
import numpy as np
import pandas as pd

@triton.jit
def softmax_kernel(output_ptr, input_ptr,
                   input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def test_do_bench(m, n, warmup=100, rep=100):
    """do_bench 测试"""
    kernel_obj = softmax_kernel
    x = torch.randn(m, n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n)
    
    # 预编译
    kernel_obj[(m,)](y, x, x.stride(0), y.stride(0), m, n,
                     BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    
    fn = lambda: kernel_obj[(m,)](y, x, x.stride(0), y.stride(0), m, n,
                                  BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    
    start = time.time()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    elapsed = time.time() - start
    
    return {'avg_time_ms': ms, 'total_time_s': elapsed}

def test_cupti(m, n, warmup=100, rep=100):
    """CUPTI 测试"""
    kernel_obj = softmax_kernel
    x = torch.randn(m, n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n)
    
    # 预编译
    kernel_obj[(m,)](y, x, x.stride(0), y.stride(0), m, n,
                     BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    
    profiler = KernelProfiler()
    
    # Warmup
    for _ in range(warmup):
        kernel_obj[(m,)](y, x, x.stride(0), y.stride(0), m, n,
                        BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    with profiler:
        for _ in range(rep):
            kernel_obj[(m,)](y, x, x.stride(0), y.stride(0), m, n,
                            BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    metrics = profiler.get_metrics()
    times = [k['duration_us'] for k in metrics['kernels']]
    
    # 移除异常值
    if len(times) > 10:
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        times = [t for t in times if lower <= t <= upper]
    
    return {
        'avg_time_ms': np.mean(times) / 1000,
        'std_time_ms': np.std(times) / 1000,
        'median_time_ms': np.median(times) / 1000,
        'total_time_s': elapsed,
    }

def main():
    print("=" * 80)
    print("Complete API Comparison: CUPTI vs do_bench")
    print("=" * 80)
    
    shapes = [4, 8, 16, 32, 512, 1024, 2048, 4096, 5120, 10240]
    m = 1024
    warmup = 100
    rep = 100
    
    print(f"\n{'Shape':<10} {'Method':<15} {'Avg (ms)':<12} {'Median (ms)':<12} {'Std (ms)':<12} {'Total (s)':<10}")
    print("-" * 85)
    
    results = []
    
    for n in shapes:
        print(f"\nShape {n}:")
        
        # do_bench
        result_db = test_do_bench(m, n, warmup, rep)
        results.append({
            'shape': n,
            'method': 'do_bench',
            **result_db
        })
        print(f"{n:<10} {'do_bench':<15} {result_db['avg_time_ms']:<12.6f} {'N/A':<12} {'N/A':<12} {result_db['total_time_s']:<10.3f}")
        
        # CUPTI
        result_cupti = test_cupti(m, n, warmup, rep)
        results.append({
            'shape': n,
            'method': 'CUPTI',
            **result_cupti
        })
        print(f"{n:<10} {'CUPTI':<15} {result_cupti['avg_time_ms']:<12.6f} {result_cupti['median_time_ms']:<12.6f} {result_cupti['std_time_ms']:<12.6f} {result_cupti['total_time_s']:<10.3f}")
    
    print("\n" + "=" * 80)
    
    df = pd.DataFrame(results)
    df.to_csv('full_api_comparison.csv', index=False)
    
    # 统计分析
    cupti_df = df[df['method'] == 'CUPTI']
    print("\n=== CUPTI Statistics ===")
    print(f"Average Std Dev: {cupti_df['std_time_ms'].mean():.6f} ms")
    print(f"Max Std Dev: {cupti_df['std_time_ms'].max():.6f} ms")
    print(f"Avg Measurement Stability: {(cupti_df['std_time_ms'] / cupti_df['avg_time_ms'] * 100).mean():.3f}%")
    
    print(f"\nResults saved to: full_api_comparison.csv")

if __name__ == '__main__':
    main()
