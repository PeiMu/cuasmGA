"""
Benchmark softmax kernel across different shapes for performance table
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import argparse
import torch
import triton
import triton.language as tl
from profiler_utils import benchmark_kernel
import pandas as pd
import numpy as np

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

def run_softmax(x, kernel=None):
    """Run softmax kernel on input x"""
    if kernel is None:
        kernel = softmax_kernel
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4
    
    y = torch.empty_like(x)
    
    kernel[(n_rows,)](
        y, x, x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y

def benchmark_shape_robust(m, n, baseline_kernel=None, warmup=10, repeat=100, 
                          remove_outliers=True):
    """Benchmark with outlier removal"""
    x = torch.randn(m, n, device='cuda', dtype=torch.float32)
    
    def kernel_func():
        return run_softmax(x, baseline_kernel)
    
    result = benchmark_kernel(kernel_func, warmup=warmup, repeat=repeat)
    
    # Get all timing data
    from profiler_utils import KernelProfiler
    profiler = KernelProfiler()
    with profiler:
        for _ in range(repeat):
            kernel_func()
        torch.cuda.synchronize()
    
    metrics = profiler.get_metrics()
    times = [k['duration_us'] for k in metrics['kernels']]
    
    if remove_outliers and len(times) > 10:
        # Remove outliers using IQR method
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        filtered_times = [t for t in times if lower <= t <= upper]
        
        if len(filtered_times) < len(times):
            print(f"  [Removed {len(times) - len(filtered_times)} outliers]", end=' ')
            times = filtered_times
    
    result = {
        'avg_time_us': np.mean(times),
        'min_time_us': np.min(times),
        'max_time_us': np.max(times),
        'std_time_us': np.std(times),
        'median_time_us': np.median(times),
        'num_runs': len(times),
        'm': m,
        'n': n,
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Benchmark softmax across different shapes')
    parser.add_argument('-m', type=int, default=1024, help='Number of rows')
    parser.add_argument('--shapes', type=int, nargs='+', 
                       default=[4, 8, 16, 32, 512, 1024, 2048, 4096],
                       help='List of column sizes (n) to benchmark')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--repeat', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', 
                       help='Output CSV file')
    parser.add_argument('--no-remove-outliers', action='store_true',
                       help='Do not remove outliers')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Softmax Kernel Benchmark")
    print("=" * 80)
    print(f"M (rows): {args.m}")
    print(f"N (cols): {args.shapes}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"Outlier removal: {not args.no_remove_outliers}")
    print("=" * 80)
    print()
    
    results = []
    
    print(f"{'Shape':<10} {'Avg (us)':<12} {'Median (us)':<12} {'Min (us)':<12} {'Max (us)':<12} {'Std (us)':<12}")
    print("-" * 80)
    
    for n in args.shapes:
        result = benchmark_shape_robust(
            args.m, n,
            warmup=args.warmup, 
            repeat=args.repeat,
            remove_outliers=not args.no_remove_outliers
        )
        
        results.append(result)
        
        print(f"{n:<10} {result['avg_time_us']:<12.3f} {result['median_time_us']:<12.3f} {result['min_time_us']:<12.3f} {result['max_time_us']:<12.3f} {result['std_time_us']:<12.3f}")
    
    print("=" * 80)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
