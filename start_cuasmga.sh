#!/bin/bash

echo "=========================================="
echo "Set CuAsmGA environment..."
echo "=========================================="  

# 1. Set PYTHONPATH  
export PYTHONPATH=/mnt/disk/cuasmGA/python:/mnt/disk/cuasmGA/CuAssembler:$PYTHONPATH    

# 2. Set PATH（add CuAssembler tools）  
export PATH=${PATH}:/mnt/disk/cuasmGA/CuAssembler/bin  

# 3. Set CUPTI PATH  
export CUPTI_INCLUDE=/opt/conda/envs/workflow/include    
export CUPTI_LIB=/opt/conda/envs/workflow/lib
export LD_LIBRARY_PATH=$CUPTI_LIB:$LD_LIBRARY_PATH

# 4. Create nvdisasm symbolic link (if not present)  
if [ ! -f /usr/local/cuda/bin/nvdisasm ]; then
    echo "Create nvdisasm symbolic link..."  
    mkdir -p /usr/local/cuda/bin
    ln -sf /opt/conda/envs/workflow/bin/nvdisasm /usr/local/cuda/bin/nvdisasm    
    echo "✓ nvdisasm The symbolic link has been created"
fi

echo "✓ The environment variables have been set"  
echo "Python: $(which python)"
echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'Not found')"
echo "nvdisasm: $(which nvdisasm 2>/dev/null || echo 'Not found')"  
echo "=========================================="
