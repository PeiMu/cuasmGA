# CUPTI Profiler for CuAsmGA

### 1. System Requirement
- CUDA 12.1+
- Python 3.11+
- PyTorch 2.1.2+
- pybind11
- CUPTI library (installed with CUDA Toolkit)



### 2. Install

## a. Set environment variables
```bash

# Check CUPTI path  
export CUPTI_INCLUDE=/opt/conda/envs/workflow/include
export CUPTI_LIB=/opt/conda/envs/workflow/lib
export LD_LIBRARY_PATH=$CUPTI_LIB:$LD_LIBRARY_PATH

# Verify
ls $CUPTI_INCLUDE/cupti.h
ls $CUPTI_LIB/libcupti.so*
```

## b. Compile C++ extensions
```bash
cd /mnt/disk/cuasmGA/cupti_profiler

# Install pybind11 (if not installed)
pip install pybind11

# Compile
python setup.py build_ext --inplace  

# Verify
ls -la cupti_profiler*.so
```

## c. Test Installation
```python
from profiler_utils import KernelProfiler
print("✓ CUPTI Profiler has been installed successfully")
```



### 3. QUICK START

## a. Basic Usage
```python
import torch
from profiler_utils import KernelProfiler

# Set profiler
profiler = KernelProfiler()

# Use context manager
with profiler:
    # Run your kernel
    your_kernel[grid](args)
    torch.cuda.synchronize()    

# Get result
metrics = profiler.get_metrics()
print(f"mean time: {metrics['avg_time_us']:.3f} us")
print(f"std: {(metrics['kernels'][0]['duration_us'] - metrics['avg_time_us']):.3f} us")
```

## b.Complete example: Test Triton Kernel
```python
import torch
import triton
import triton.language as tl
from profiler_utils import benchmark_kernel

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Prepare data
size = 1024
x = torch.randn(size, device='cuda')
y = torch.randn(size, device='cuda')
output = torch.empty_like(x)

# Define the kernel call
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

def run_kernel():
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    torch.cuda.synchronize()

# Measure performance
result = benchmark_kernel(run_kernel, warmup=10, repeat=100)
print(f"mean time: {result['avg_time_us']:.3f} us ±{result['std_time_us']:.3f} us")
print(f"std: {(result['std_time_us']/result['avg_time_us']*100):.2f}%")
```



### 4. API

## a. KernelProfiler class
The context manager that uses CUPTI to capture kernel performance data

# Method
**`__enter__()` / `__exit__()`**
- Used as a context manager to automatically start and stop performance monitoring

**`get_metrics() -> dict`**

Return performance metrics：
```python
{
    'kernels': [                    # Detailed information of each kernel
        {
            'name': str,            # Kernel name
            'duration_us': float,   # Execution time (ms)
            'grid': tuple,          # (gridX, gridY, gridZ)
            'block': tuple          # (blockX, blockY, blockZ)
        },
        ...
    ],
    'count': int,                   # Total number of Kernel calls
    'total_time_us': float,         # Total execution time
    'avg_time_us': float            # Average execution time
}
```

**`clear()`**
- Clear the collected performance data

## b. benchmark_kernel function
```python
benchmark_kernel(kernel_func, warmup=10, repeat=100)
```

- Automate kernel performance testing

**parameter：**
- `kernel_func`: The function to be tested (which should include kernel calls and synchronization)
- `warmup`: Preheating times（DEFAULT 10）
- `repeat`: Number of repeated tests（DEFAULT 100）

**return：**
```python
{
    'avg_time_us': float,       # Average time (ms)
    'min_time_us': float,       # Minimum time
    'max_time_us': float,       # Maximum time
    'std_time_us': float,       # Std
    'total_time_us': float,     # Total time
    'num_runs': int,            # Running time
    'kernel_name': str          # Kernel name
}
```

## c. compare_kernels function  
```python
compare_kernels(baseline_func, optimized_func, warmup=10, repeat=100)
```

- Compare the performance of the baseline and the optimized kernel

**return：**
```python
{
    'baseline': dict,           # baseline performance metrics
    'optimized': dict,          # Optimized performance indicators
    'speedup': float,           # speed-up ratio
    'improvement_percent': float # Percentage of performance improvement
}
```



### 5. Testing Script

## a. benchmark_shapes.py - Baseline performance test
Test the baseline performance of multiple shapes and establish performance benchmarks

```bash
# Basic Usage
python benchmark_shapes.py -m 1024 --shapes 512 1024 2048 4096

# Complete parameters
python benchmark_shapes.py \
    -m 1024 \
    --shapes 4 8 16 32 512 1024 2048 4096 5120 10240 \
    --warmup 10 \
    --repeat 100 \
    --output baseline_results.csv
```

**parameter description：**
- `-m`: Number of rows (matrix dimension)
- `--shapes`: Column count list (test shape)
- `--warmup`: Preheating times
- `--repeat`: The number of repeated tests
- `--output`: Output CSV file
- `--no-remove-outliers`: Do not remove outliers

## b. full_comparison.py - Complete API comparison
Compare the performance of 'do_bench' and CUPTI on all shapes

```bash
python full_comparison.py
```

Automatically test 10 shapes (4, 8, 16, 32, 512, 1024, 2048, 4096, 5120, 10240), output:
- `full_api_comparison.csv` - Complete comparison data
- CUPTI statistical information (mean standard deviation, stability)

## c. test_ga_final.py - Test of GA optimization results
Test the performance comparison before and after GA optimization

```bash
python test_ga_final.py \
    -m 1024 \
    -n 16384 \
    --load-dir data/NVIDIA_A100-SXM4-80GB/persistent_softmax/1024_16384 \
    --warmup 100 \
    --repeat 100
```

**output：**
- Baseline vs GA Optimized comparison
- do_bench vs CUPTI comparison
- Percentage improvement in GA performance
- `ga_comparison_<m>x<n>.csv` - detailed data



### 6. Project Integration

## a. Start the complete environment
```bash
cd /mnt/disk/cuasmGA

source .venv/bin/activate

source start_cuasmga.sh

python -c "from profiler_utils import KernelProfiler; print('✓ Ready')"      
```

## b. Run the complete testing process
```bash
# 1. Baseline performance test
python benchmark_shapes.py -m 1024 --shapes 512 1024 2048 4096

# 2. Complete comparison of API
python full_comparison.py

# 3. Run GA optimization
python 04-fused-softmax.py -m 1024 -n 16384

# 4. Test GA results
python test_ga_final.py \
    -m 1024 -n 16384 \
    --load-dir data/NVIDIA_A100-SXM4-80GB/persistent_softmax/1024_16384
```

