#!/bin/bash

echo "=========================================="
echo "设置 CuAsmGA 环境..."
echo "=========================================="

# 1. 设置 PYTHONPATH
export PYTHONPATH=/mnt/disk/cuasmGA/python:/mnt/disk/cuasmGA/CuAssembler:$PYTHONPATH

# 2. 设置 PATH（添加 CuAssembler 工具）
export PATH=${PATH}:/mnt/disk/cuasmGA/CuAssembler/bin

# 3. 设置 CUPTI 路径
export CUPTI_INCLUDE=/opt/conda/envs/workflow/include
export CUPTI_LIB=/opt/conda/envs/workflow/lib
export LD_LIBRARY_PATH=$CUPTI_LIB:$LD_LIBRARY_PATH

# 4. 创建 nvdisasm 符号链接（如果不存在）
if [ ! -f /usr/local/cuda/bin/nvdisasm ]; then
    echo "创建 nvdisasm 符号链接..."
    mkdir -p /usr/local/cuda/bin
    ln -sf /opt/conda/envs/workflow/bin/nvdisasm /usr/local/cuda/bin/nvdisasm
    echo "✓ nvdisasm 符号链接已创建"
fi

echo "✓ 环境变量已设置"
echo "Python: $(which python)"
echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'Not found')"
echo "nvdisasm: $(which nvdisasm 2>/dev/null || echo 'Not found')"
echo "=========================================="
