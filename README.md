# cuasmGA — Genetic Algorithm for GPU SASS Instruction Scheduling

BEng Final Year Project, University of Edinburgh, 2026.
Charles Jiang. Supervisor: Prof. Antonio Barbalace. Advisor: Pei Mu.

## Overview

This project replaces the RL agent in CuAsmRL (CGO 2025) with a genetic
algorithm for optimising SASS instruction schedules in Triton-compiled
LLM kernels. See the project report for full details.

## Reproducing results

**Hardware required:** NVIDIA A100-SXM4-80GB GPU, CUDA 12.1.

**Run GA on an operator** (e.g., Fused MoE):
```bash
source start_cuasmga.sh
python 07-fused_moe.py --n_tests 50
```

**Run CUPTI benchmark:**
```bash
python bench_cupti.py
```

Results are saved to the `data/` directory as pickle files.

## Measurement

All reported results use CUPTI hardware profiling (`cupti_profiler/`).
The C++ extension is built via `cd cupti_profiler && pip install -e .`.
`do_bench` results shown in the report are for comparison only.

## GA strategy implementations

See `ga_strategies/README.md` for the five mutation/crossover strategies
described in Chapter 4 of the report.

## Data provenance

- Kernel operators: Triton tutorials (softmax, flash-attention, batch matmul),
  Kernl (mmLeakyReLU, fused feedforward), vLLM 0.4.0 (Fused MoE FP8),
  gptq-triton (W4A16 Dequant GEMM), Triton/LLaMA (RMSNorm)
- CuAsmRL baseline: https://github.com/guolinghui/CuAsmRL (CGO 2025)
- No external datasets; all inputs are randomly generated at runtime



#intsall

```bash
git clone https://github.com/TristeSunrise/cuasmGA.git

 
apt-get install clang -y
apt-get install lld -y
apt-get install ccache -y
apt-get install zlib1g-dev -y

# reinstall cuda version is necessary
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*"
# https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1
sudo apt install cuda-toolkit-12-1 # add PATH etc.
sudo apt install nvidia-gds-12-1

# or use the image: dstackai/base:py3.11-0.4-cuda-12.1-devel
# and soft link the pre-built CUDA bin directory
# ln -s /opt/conda/envs/workflow/bin /usr/local/cuda/bin

python -m venv .venv --prompt triton
source .venv/bin/activate
pip install ninja cmake wheel; # build-time dependencies
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation -r requirement.txt
pip install flash-attn==2.3.3
 
pip uninstall triton -y
pip install -e python --no-build-isolation
export PATH=$PWD/python/triton/third_party/cuda/bin:${PATH}
 
pip install pyelftools tensorboard
pip install nvidia-cutlass==3.5
 
 
git submodule update --init --recursive 
export PATH=${PATH}:$PWD/CuAssembler/bin
export PYTHONPATH=${PYTHONPATH}:$PWD/CuAssembler:$PWD/CuAssembler/bin:$PWD/CuAssembler/CuAsm

cd $PWD/CuAssembler/bin
# from within bin directory
ln -s cuasm.py cuasm
chmod a+x cuasm


tmux new -s cuasmGA
source .venv/bin/activate
rm train.log
bash train.sh 2>&1|tee train.log
rm inference.log
bash inference.sh 2>&1|tee inference.log
```
# structure
`autotuner.py` is the level of tuning the best kernel configuration(grid, block, threads...).

`jit.py` is the level of searching the best optimized kernel in the inference process and will do the `runga` in the training process.

`runga.py` is the entrance of calling the GA algorithm. The defination of `test_correctness` and 
`test_performance`. The best kernel is saved here.

`decoder.py` to decode a single instruction into control code , operation and oprands .

`newga.py` is the currently entire GA algorithm used, including the selection, crossover, mutation, and so on. Add the new function or modified old ones in this file.

`sample.py` is the entrance of calling the stastic_analysis, every step of mutation and crossover will redo the stastic_analysis to uptate the movable instructions conditates.

`gpu_utils.py` is the utils function used in the this projrct to get memory instructions and banne  operations.

`selection.py` is to seltect the best kernel saved in the save path. Modify the path in the config class of every kernel you want.

`sass_kernel.py` is to locate the kernel section in the whole sass file and to update the optimized kernel section, embedding it back into the sass file.

`sassgen.py` is the only file calling the CuAsmbler to convert the sass file into the binary cubin file or reversely.  

`verify.py` is to verify the valiation of a kernel.

currently aborted files: `safe_mem_mover.py` (failed to analysis the computing instruction dependency) and `ga` (original version of the randomly disrupted instruction sequence).

