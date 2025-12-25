from setuptools import setup, Extension
import os
import pybind11

cupti_include = os.environ.get('CUPTI_INCLUDE', '/opt/conda/envs/workflow/include')
cupti_lib = os.environ.get('CUPTI_LIB', '/opt/conda/envs/workflow/lib')

ext_modules = [
    Extension(
        'cupti_profiler',
        ['cupti_profiler.cpp'],
        include_dirs=[
            pybind11.get_include(),
            cupti_include,
            '/usr/local/cuda/include',
        ],
        library_dirs=[
            cupti_lib,
            '/usr/local/cuda/lib64',
        ],
        libraries=['cupti', 'cudart'],
        runtime_library_dirs=[cupti_lib],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='cupti_profiler',
    version='0.1',
    ext_modules=ext_modules,
)
