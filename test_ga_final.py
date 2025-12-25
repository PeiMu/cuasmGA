"""  
Fix version: GA kernel is called without passing num_stages and num_warps
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CuAssembler'))  

import argparse
import torch
import triton
import triton.language as tl
import triton.testing
from profiler_utils import KernelProfiler  
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from jit import jit
from autotuner import autotune as fgk_autotune
from gpu_utils import get_gpu_name

GPU = get_gpu_name()

@dataclass
class GAConfig:
    default_out_path: str = "data"
    save_dir: str = ""
    total_flops: float = 0.0
    seed: int = 1337
    n_tests: int = 2

def create_baseline_kernel():
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
    
    return softmax_kernel

def create_ga_kernel(m, n, load_dir):
    BLOCK_SIZE = triton.next_power_of_2(n)
    
    config = GAConfig()
    x = torch.randn(m, n, device='cuda', dtype=torch.float32)
    config.total_flops = 2 * x.nelement() * x.element_size()
    config.save_dir = f'{GPU}/persistent_softmax/{m}_{n}'
    
    @fgk_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': BLOCK_SIZE}, num_stages=4, num_warps=8),
        ],
        key=['n_rows', 'n_cols'],
        ga_config=config,
        ret_ptr=0,
    )
    @jit
    def ga_kernel(output_ptr, input_ptr,
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
    
    return ga_kernel

def benchmark_baseline_dobench(kernel, x, warmup=100, rep=100):
    """Baseline: do_bench"""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    
    kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                     BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    
    fn = lambda: kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                   BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    
    start = time.time()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms, time.time() - start

def benchmark_baseline_cupti(kernel, x, warmup=100, rep=100):
    """Baseline: CUPTI"""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    
    profiler = KernelProfiler()
    
    for _ in range(warmup):
        kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                         BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
    torch.cuda.synchronize()
    
    start = time.time()
    with profiler:
        for _ in range(rep):
            kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                             BLOCK_SIZE=BLOCK_SIZE, num_stages=4, num_warps=8)
        torch.cuda.synchronize()
    total_time = time.time() - start
    
    metrics = profiler.get_metrics()
    times = [k['duration_us'] for k in metrics['kernels']]
    
    if len(times) > 10:
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        times = [t for t in times if q1-1.5*iqr <= t <= q3+1.5*iqr]
    
    return np.mean(times)/1000, np.std(times)/1000, total_time

def benchmark_ga_dobench(kernel, x, load_dir, warmup=100, rep=100):
    """GA: do_bench - 不传递 num_stages 和 num_warps"""
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    
    # When GA kernel is invoked, only data parameters are passed, not num_stages/num_warps  
    kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, load_dir=load_dir)  
    
    fn = lambda: kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, load_dir=load_dir)    
    
    start = time.time()
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms, time.time() - start

def benchmark_ga_cupti(kernel, x, load_dir, warmup=100, rep=100):
    """GA: CUPTI - Does not pass num_stages 和 num_warps"""          
    n_rows, n_cols = x.shape  
    y = torch.empty_like(x)
    
    profiler = KernelProfiler()
    
    for _ in range(warmup):
        kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, load_dir=load_dir)
    torch.cuda.synchronize()
    
    start = time.time()
    with profiler:
        for _ in range(rep):
            kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, load_dir=load_dir)
        torch.cuda.synchronize()
    total_time = time.time() - start
    
    metrics = profiler.get_metrics()
    times = [k['duration_us'] for k in metrics['kernels']]
    
    if len(times) > 10:
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        times = [t for t in times if q1-1.5*iqr <= t <= q3+1.5*iqr]
    
    return np.mean(times)/1000, np.std(times)/1000, total_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('-n', type=int, required=True)
    parser.add_argument('--load-dir', type=str, required=True)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=100)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GA Optimization Test: CUPTI vs do_bench")
    print("=" * 80)
    print(f"Shape: {args.m} x {args.n}")
    print(f"Load from: {args.load_dir}")
    print("=" * 80)
    
    x = torch.randn(args.m, args.n, device='cuda', dtype=torch.float32)
    results = []
    
    # Baseline
    print("\n--- Baseline Kernel ---")
    baseline_kernel = create_baseline_kernel()
    
    print("  do_bench...", end=' ', flush=True)
    ms_db, time_db = benchmark_baseline_dobench(baseline_kernel, x, args.warmup, args.repeat)
    results.append({'type': 'baseline', 'method': 'do_bench', 'avg_ms': ms_db, 'total_s': time_db})
    print(f"{ms_db:.6f}ms")
    
    print("  CUPTI...", end=' ', flush=True)
    ms_cu, std_cu, time_cu = benchmark_baseline_cupti(baseline_kernel, x, args.warmup, args.repeat)
    results.append({'type': 'baseline', 'method': 'CUPTI', 'avg_ms': ms_cu, 'std_ms': std_cu, 'total_s': time_cu})
    print(f"{ms_cu:.6f}ms ±{std_cu:.6f}ms")
    
    # GA Optimized
    print("\n--- GA Optimized Kernel ---")
    print("  Loading optimized kernel...", flush=True)
    ga_kernel = create_ga_kernel(args.m, args.n, args.load_dir)
    
    print("  do_bench...", end=' ', flush=True)
    ms_db_opt, time_db_opt = benchmark_ga_dobench(ga_kernel, x, args.load_dir, args.warmup, args.repeat)
    results.append({'type': 'optimized', 'method': 'do_bench', 'avg_ms': ms_db_opt, 'total_s': time_db_opt})
    print(f"{ms_db_opt:.6f}ms")
    
    print("  CUPTI...", end=' ', flush=True)
    ms_cu_opt, std_cu_opt, time_cu_opt = benchmark_ga_cupti(ga_kernel, x, args.load_dir, args.warmup, args.repeat)
    results.append({'type': 'optimized', 'method': 'CUPTI', 'avg_ms': ms_cu_opt, 'std_ms': std_cu_opt, 'total_s': time_cu_opt})
    print(f"{ms_cu_opt:.6f}ms ±{std_cu_opt:.6f}ms")
    
    # Analysis
    print("\n" + "=" * 80)
    print("=== Results Summary ===")
    print(f"\nBaseline:")
    print(f"  do_bench: {ms_db:.6f}ms")
    print(f"  CUPTI:    {ms_cu:.6f}ms (±{std_cu:.6f}ms)")
    print(f"\nGA Optimized:")
    print(f"  do_bench: {ms_db_opt:.6f}ms")
    print(f"  CUPTI:    {ms_cu_opt:.6f}ms (±{std_cu_opt:.6f}ms)")
    
    print(f"\n=== GA Improvement ===")
    improvement_db = (ms_db - ms_db_opt) / ms_db * 100
    improvement_cu = (ms_cu - ms_cu_opt) / ms_cu * 100
    print(f"  do_bench: {improvement_db:+.2f}%")
    print(f"  CUPTI:    {improvement_cu:+.2f}%")
    
    print(f"\n=== CUPTI Advantages (on GA Optimized) ===")
    print(f"  Measurement precision: ±{std_cu_opt:.6f}ms ({std_cu_opt/ms_cu_opt*100:.3f}%)")
    print(f"  Time difference vs do_bench: {abs(ms_db_opt-ms_cu_opt):.6f}ms ({abs(ms_db_opt-ms_cu_opt)/ms_db_opt*100:.2f}%)")
    
    # Save
    df = pd.DataFrame(results)
    output_file = f'ga_comparison_{args.m}x{args.n}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
