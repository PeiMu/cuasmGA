import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch

import triton
import triton.language as tl

import random
import numpy as np

from jit import jit
from autotuner import autotune
from autotuner import triton_autotune_with_cache
from gpu_utils import get_gpu_name, get_gpu_cc


def moe_gen_test_samples(kernel, non_constexpr_arg_values,
                         grid_0, grid_1, grid_2,
                         stream, launch_enter_hook, launch_exit_hook,
                         n_test_samples, ret_ptr):
    """Custom test sample generator for MoE kernel.

    Integer index tensors (sorted_token_ids, expert_ids, num_tokens_post_padded)
    are cloned as-is instead of randomized, since they encode valid routing
    structure that cannot be randomly generated.
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
                    # integer/index tensors: keep valid structure
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
    M: int = 512         # number of tokens
    E: int = 8           # number of experts
    N: int = 2048        # intermediate size (N of w1)
    K: int = 512         # hidden size
    topk: int = 2        # top-k experts

    # Custom test sample generator (for kernels with integer index tensors)
    gen_test_samples: object = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fused MoE")

    parser.add_argument("--default_out_path", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_tests", type=int, default=10)
    parser.add_argument("--load", type=str)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tt', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--topk", type=int, default=2)

    args = parser.parse_args()
    config = Config(**vars(args))
    config.gen_test_samples = moe_gen_test_samples
    return config


GPU = get_gpu_name()


def moe_align_block_size_py(topk_ids: torch.Tensor, block_size: int,
                             num_experts: int):
    """Pure Python version of moe_align_block_size."""
    M_tokens, top_k = topk_ids.shape
    total = M_tokens * top_k

    # Count tokens per expert
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    flat_ids = topk_ids.view(-1)
    for e in range(num_experts):
        tokens_per_expert[e] = (flat_ids == e).sum()

    # Compute padded counts
    padded_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size

    # Build sorted token ids
    num_tokens_post_padded = padded_per_expert.sum().item()
    sorted_token_ids = torch.full((num_tokens_post_padded,), total,
                                  dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(num_tokens_post_padded // block_size,
                             dtype=torch.int32, device=topk_ids.device)

    # For each expert, place its token indices
    offset = 0
    for e in range(num_experts):
        # Find which (token, k) pairs are assigned to expert e
        mask = (topk_ids == e)
        # Get flat indices: token_idx * top_k + k_idx
        indices = mask.nonzero(as_tuple=False)  # [count, 2]
        # The token id in the sorted list is token_idx * top_k + k_idx
        token_indices = indices[:, 0] * top_k + indices[:, 1]
        count = token_indices.shape[0]
        padded = padded_per_expert[e].item()
        sorted_token_ids[offset:offset + count] = token_indices.to(torch.int32)
        # Fill expert_ids for each block
        n_blocks = padded // block_size
        expert_ids[offset // block_size: offset // block_size + n_blocks] = e
        offset += padded

    num_tokens_post_pad = torch.tensor([num_tokens_post_padded],
                                       dtype=torch.int32,
                                       device=topk_ids.device)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def call(A, B, C, topk_weights, topk_ids,
         sorted_token_ids, expert_ids, num_tokens_post_padded,
         mul_routed_weight, top_k, kernel, load_dir):
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,

        # gh512
        load_dir=load_dir,
    )


def triton_moe_forward(A, B, C, topk_weights, topk_ids,
                        sorted_token_ids, expert_ids, num_tokens_post_padded,
                        mul_routed_weight, top_k, kernel):
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
    )


def main():
    config = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # workload
    M_tokens = config.M
    E = config.E
    N = config.N
    K = config.K
    topk = config.topk
    dtype = torch.float16

    # hidden_states: (M, K)
    hidden_states = torch.randn((M_tokens, K), dtype=dtype, device="cuda")
    # w1: (E, N, K)  -- expert weights
    w1 = torch.randn((E, N, K), dtype=dtype, device="cuda")

    # gating: softmax to get topk
    gating_output = torch.randn((M_tokens, E), dtype=torch.float32, device="cuda")
    routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # args
    config.save_dir = f'{GPU}/fused_moe/{M_tokens}_{E}_{N}_{K}_{topk}'

    @autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        ],
        key=['N', 'K'],
        ret_ptr=2,
        ga_config=config,
    )
    @jit
    def ga_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # Strides
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                          offs_k[None, :] * stride_ak)

        off_experts = tl.load(expert_ids_ptr + pid_m)
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                    offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs,
                        mask=token_mask[:, None] &
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                 mask=token_mask,
                                 other=0)
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @triton_autotune_with_cache(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        ],
        key=['N', 'K'],
        ga_config=config,
    )
    @triton.jit
    def tt_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # Strides
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                          offs_k[None, :] * stride_ak)

        off_experts = tl.load(expert_ids_ptr + pid_m)
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                    offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs,
                        mask=token_mask[:, None] &
                        (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token,
                                 mask=token_mask,
                                 other=0)
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    # Prepare MoE alignment data
    default_block_size_m = 128
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_py(
        topk_ids, default_block_size_m, E)

    # Output cache: (M, topk, N)
    intermediate_cache = torch.empty((M_tokens, topk, N),
                                     device=hidden_states.device,
                                     dtype=hidden_states.dtype)
    mul_routed_weight = False

    if config.load is None:
        load_dir = None
    elif config.load == "auto":
        load_dir = f'{config.default_out_path}/{GPU}/fused_moe/{M_tokens}_{E}_{N}_{K}_{topk}'
    else:
        load_dir = config.load

    ## TEST
    fgk_out = intermediate_cache.clone()
    call(hidden_states, w1, fgk_out, topk_weights, topk_ids,
         sorted_token_ids, expert_ids, num_tokens_post_padded,
         mul_routed_weight, topk, ga_kernel, load_dir)

    if config.tt:
        tri_out = intermediate_cache.clone()
        triton_moe_forward(hidden_states, w1, tri_out, topk_weights, topk_ids,
                           sorted_token_ids, expert_ids, num_tokens_post_padded,
                           mul_routed_weight, topk, tt_kernel)
        assert torch.allclose(tri_out, fgk_out, atol=1e-2, rtol=0)
    print('TEST PASSED')

    if not config.bench:
        print('SKIP bench...')
        return

    assert config.load is not None
    torch.cuda.synchronize()

    print(f"Benchmarking Fused MoE: M={M_tokens}, E={E}, N={N}, K={K}, topk={topk}")

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
            plot_name=f"fused-moe-E{E}-N{N}-K{K}-topk{topk}",
            args={
                "E": E,
                "N": N,
                "K": K,
                "topk": topk,
                "dtype": torch.float16,
            },
        ))

    @triton.testing.perf_report(configs_bench)
    def bench_fused_moe(M, E, N, K, topk, provider, dtype=torch.float16, device="cuda"):
        print(f'[BENCH]: {provider};; M={M} E={E} N={N} K={K} topk={topk}')
        warmup = 100
        rep = 100

        hidden = torch.randn((M, K), dtype=dtype, device=device)
        w = torch.randn((E, N, K), dtype=dtype, device=device)
        gating = torch.randn((M, E), dtype=torch.float32, device=device)
        rw = torch.softmax(gating, dim=-1, dtype=torch.float32)
        tw, ti = torch.topk(rw, topk, dim=-1)
        tw = tw / tw.sum(dim=-1, keepdim=True)

        s_ids, e_ids, n_post = moe_align_block_size_py(ti, default_block_size_m, E)
        cache = torch.empty((M, topk, N), device=device, dtype=dtype)

        if provider == "fgk":
            fn = lambda: call(hidden, w, cache, tw, ti, s_ids, e_ids, n_post,
                              False, topk, ga_kernel, load_dir)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if provider == "triton":
            fn = lambda: triton_moe_forward(hidden, w, cache, tw, ti, s_ids, e_ids, n_post,
                                            False, topk, tt_kernel)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        # 2 * M * topk * N * K flops for the matmul
        total_flops = 2.0 * M * topk * N * K
        return total_flops / ms * 1e-9

    df = bench_fused_moe.run(print_data=True, return_df=True)


if __name__ == "__main__":
    main()
