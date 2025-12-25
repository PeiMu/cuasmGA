"""
CUPTI-based performance profiling utilities for cuasmGA
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cupti_profiler'))

import cupti_profiler
import torch
from contextlib import contextmanager

class KernelProfiler:
    """Context manager for profiling CUDA kernels using CUPTI"""
    
    def __init__(self):
        self.profiler = cupti_profiler.CuptiProfiler.get_instance()
        
    def __enter__(self):
        self.profiler.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop()
        return False
        
    def get_metrics(self):
        """Get profiling metrics"""
        return self.profiler.get_metrics()
    
    def clear(self):
        """Clear collected metrics"""
        self.profiler.clear()


def benchmark_kernel(kernel_func, *args, warmup=10, repeat=100):
    """
    Benchmark a kernel function
    
    Args:
        kernel_func: The kernel function to benchmark
        *args: Arguments to pass to the kernel
        warmup: Number of warmup runs
        repeat: Number of benchmark runs
        
    Returns:
        dict: Performance metrics including average time, min time, max time
    """
    # Warmup
    for _ in range(warmup):
        kernel_func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    profiler = KernelProfiler()
    with profiler:
        for _ in range(repeat):
            kernel_func(*args)
        torch.cuda.synchronize()
    
    metrics = profiler.get_metrics()
    
    if metrics['count'] == 0:
        raise RuntimeError("No kernels were profiled")
    
    # Calculate statistics
    times = [k['duration_us'] for k in metrics['kernels']]
    
    result = {
        'avg_time_us': sum(times) / len(times),
        'min_time_us': min(times),
        'max_time_us': max(times),
        'std_time_us': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'total_time_us': sum(times),
        'num_runs': len(times),
        'kernel_name': metrics['kernels'][0]['name'] if metrics['kernels'] else 'unknown',
    }
    
    return result


def compare_kernels(baseline_func, optimized_func, *args, warmup=10, repeat=100):
    """
    Compare performance between baseline and optimized kernels
    
    Returns:
        dict: Comparison results including speedup
    """
    baseline_metrics = benchmark_kernel(baseline_func, *args, warmup=warmup, repeat=repeat)
    optimized_metrics = benchmark_kernel(optimized_func, *args, warmup=warmup, repeat=repeat)
    
    speedup = baseline_metrics['avg_time_us'] / optimized_metrics['avg_time_us']
    improvement = (1 - optimized_metrics['avg_time_us'] / baseline_metrics['avg_time_us']) * 100
    
    return {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'speedup': speedup,
        'improvement_percent': improvement,
    }
