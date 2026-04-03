import os
import argparse
from dataclasses import dataclass
from typing import Optional

import torch

import triton
import triton.language as tl

import random
import numpy as np

from jit import jit
from autotuner import autotune
from autotuner import triton_autotune_with_cache
from gpu_utils import get_gpu_name, get_gpu_cc


def w4a16_gen_test_samples(kernel, non_constexpr_arg_values,
                           grid_0, grid_1, grid_2,
                           stream, launch_enter_hook, launch_exit_hook,
                           n_test_samples, ret_ptr):
    """Custom test sample generator for W4A16 dequant GEMM kernel.

    Integer tensors (qweight: int32 packed, qzeros: int32 packed) are cloned
    as-is instead of randomized, since they encode valid packed 4-bit values
    that cannot be randomly generated with torch.randn_like().
    """
    from copy import deepcopy
    test_samples = []
    for t in range(n_test_samples):
        test_list = []
        for i, inp in enumerate(non_constexpr_arg_values):
            if isinstance(inp, torch.Tensor):
                if i == ret_ptr:
                    arg = torch.empty_like(inp)
                elif inp.is_floating_point():
                    arg = torch.randn_like(inp)
                else:
                    # integer tensors (qweight, qzeros): keep valid packed structure
                    arg = inp.clone()
            else:
                arg = deepcopy(inp)
            test_list.append(arg)

        # run reference kernel to produce ground truth output
        kernel.c_wrapper(
            grid_0, grid_1, grid_2,
            kernel.num_warps, kernel.num_ctas,
            kernel.clusterDims[0], kernel.clusterDims[1], kernel.clusterDims[2],
            kernel.shared, stream, kernel.cu_function,
            launch_enter_hook, launch_exit_hook, kernel,
            *kernel.assemble_tensormap_to_arg(test_list),
        )
        test_samples.append(test_list)
    return test_samples


@dataclass
class Config:
    # Kernel
    default_out_path: str = "data"
    seed: int = 1337
    n_tests: int = 2
    load: Optional[str] = None
    bench: bool = False
    tt: bool = False

    # Workload
    M: int = 512          # batch (tokens)
    N: int = 4096         # outfeatures
    K: int = 4096         # infeatures
    groupsize: int = 128  # quantization group size
    bits: int = 4         # weight bit width

    # Custom test sample generator
    gen_test_samples: object = None

    # Set by main()
    total_flops: float = None
    save_dir: str = ""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="W4A16 Dequant GEMM")

    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_tests", type=int, default=10)
    parser.add_argument("--load", type=str)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--groupsize", type=int, default=128)

    args = parser.parse_args()
    config = Config(**vars(args))
    config.gen_test_samples = w4a16_gen_test_samples
    return config


GPU = get_gpu_name()


def call(x, qweight, c, scales, qzeros,
         M, N, K, groupsize, kernel, load_dir):
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    kernel[grid](
        x, qweight, c,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        groupsize,
        NO_GROUPS=groupsize == K,

        # gh512
        load_dir=load_dir,
    )


def triton_w4a16_forward(x, qweight, c, scales, qzeros,
                          M, N, K, groupsize, kernel):
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    kernel[grid](
        x, qweight, c,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        groupsize,
        NO_GROUPS=groupsize == K,
    )


def main():
    config = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # workload
    M = config.M
    N = config.N
    K = config.K
    groupsize = config.groupsize
    bits = config.bits
    G = K // groupsize

    # total_flops: standard GEMM
    config.total_flops = 2.0 * M * N * K
    config.save_dir = f'{GPU}/w4a16_dequant_gemm/{M}_{N}_{K}_{groupsize}'

    # Tensor construction
    # a: (M, K) float16 — input activations
    x = torch.randn((M, K), dtype=torch.float16, device="cuda")
    # qweight: (K//8, N) int32 — packed 4-bit weights (8 x 4bit per int32)
    qweight = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device="cuda")
    # scales: (G, N) float16 — per-group per-outfeature scale
    scales = torch.randn((G, N), dtype=torch.float16, device="cuda")
    # qzeros: (G, N//8) int32 — packed 4-bit zeros (8 x 4bit per int32)
    qzeros = torch.randint(0, 2**31 - 1, (G, N // 8), dtype=torch.int32, device="cuda")
    # c: (M, N) float16 — output
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")

    # All BLOCK_SIZE_M = 64 configs only
    autotune_configs = [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ]

    @autotune(
        configs=autotune_configs,
        key=['M', 'N', 'K'],
        ret_ptr=2,
        ga_config=config,
    )
    @jit
    def ga_kernel(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize,
        NO_GROUPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M)
        b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
        scales_ptrs = scales_ptr + offs_bn * stride_scales_n
        zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

        shifter = (offs_k % 8) * 4
        zeros_shifter = (offs_bn % 8) * 4

        if NO_GROUPS:
            scales = tl.load(scales_ptrs)
            zeros = tl.load(zeros_ptrs)
            zeros = (zeros >> zeros_shifter) & 0xF
            zeros = (zeros + 1) * scales

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_pid_k):
            a = tl.load(a_ptrs, mask=a_mask, other=0.)
            b = tl.load(b_ptrs)

            if not NO_GROUPS:
                g_id = k // (groupsize // BLOCK_SIZE_K)
                ptr = scales_ptrs + g_id * stride_scales_g
                scales = tl.load(ptr)
                ptr = zeros_ptrs + g_id * stride_zeros_g
                zeros = tl.load(ptr)
                zeros = (zeros >> zeros_shifter) & 0xF
                zeros = (zeros + 1) * scales

            b = (b >> shifter[:, None]) & 0xF
            b = b * scales[None, :] - zeros[None, :]

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @triton_autotune_with_cache(
        configs=autotune_configs,
        key=['M', 'N', 'K'],
        ga_config=config,
    )
    @triton.jit
    def tt_kernel(
        a_ptr, b_ptr, c_ptr,
        scales_ptr, zeros_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize,
        NO_GROUPS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M)
        b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
        scales_ptrs = scales_ptr + offs_bn * stride_scales_n
        zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

        shifter = (offs_k % 8) * 4
        zeros_shifter = (offs_bn % 8) * 4

        if NO_GROUPS:
            scales = tl.load(scales_ptrs)
            zeros = tl.load(zeros_ptrs)
            zeros = (zeros >> zeros_shifter) & 0xF
            zeros = (zeros + 1) * scales

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_pid_k):
            a = tl.load(a_ptrs, mask=a_mask, other=0.)
            b = tl.load(b_ptrs)

            if not NO_GROUPS:
                g_id = k // (groupsize // BLOCK_SIZE_K)
                ptr = scales_ptrs + g_id * stride_scales_g
                scales = tl.load(ptr)
                ptr = zeros_ptrs + g_id * stride_zeros_g
                zeros = tl.load(ptr)
                zeros = (zeros >> zeros_shifter) & 0xF
                zeros = (zeros + 1) * scales

            b = (b >> shifter[:, None]) & 0xF
            b = b * scales[None, :] - zeros[None, :]

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    if config.load is None:
        load_dir = None
    elif config.load == "auto":
        load_dir = f'{config.default_out_path}/{GPU}/w4a16_dequant_gemm/{M}_{N}_{K}_{groupsize}'
    else:
        load_dir = config.load

    ## TEST
    fgk_out = c.clone()
    call(x, qweight, fgk_out, scales, qzeros, M, N, K, groupsize, ga_kernel, load_dir)

    if config.tt:
        tri_out = c.clone()
        triton_w4a16_forward(x, qweight, tri_out, scales, qzeros, M, N, K, groupsize, tt_kernel)
        assert torch.allclose(tri_out, fgk_out, atol=1e-2, rtol=0)
    print('TEST PASSED')

    if not config.bench:
        print('SKIP bench...')
        return

    assert config.load is not None
    torch.cuda.synchronize()

    print(f"Benchmarking W4A16 Dequant GEMM: M={M}, N={N}, K={K}, groupsize={groupsize}")

    configs_bench = []
    configs_bench.append(
        triton.testing.Benchmark(
            x_names=["M"],
            x_vals=[config.M],
            line_arg="provider",
            line_vals=["fgk", "triton"],
            line_names=["FGK", "Triton"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"w4a16-dequant-gemm-N{N}-K{K}-g{groupsize}",
            args={
                "N": N,
                "K": K,
                "groupsize": groupsize,
                "dtype": torch.float16,
            },
        ))

    @triton.testing.perf_report(configs_bench)
    def bench_w4a16(M, N, K, groupsize, provider, dtype=torch.float16, device="cuda"):
        print(f'[BENCH]: {provider};; M={M} N={N} K={K} g={groupsize}')
        warmup = 100
        rep = 100
        G = K // groupsize

        a = torch.randn((M, K), dtype=dtype, device=device)
        qw = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)
        sc = torch.randn((G, N), dtype=dtype, device=device)
        qz = torch.randint(0, 2**31 - 1, (G, N // 8), dtype=torch.int32, device=device)
        out = torch.empty((M, N), dtype=dtype, device=device)

        if provider == "fgk":
            fn = lambda: call(a, qw, out, sc, qz, M, N, K, groupsize, ga_kernel, load_dir)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if provider == "triton":
            fn = lambda: triton_w4a16_forward(a, qw, out, sc, qz, M, N, K, groupsize, tt_kernel)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        total_flops = 2.0 * M * N * K
        return total_flops / ms * 1e-9

    df = bench_w4a16.run(print_data=True, return_df=True)


if __name__ == "__main__":
    main()
