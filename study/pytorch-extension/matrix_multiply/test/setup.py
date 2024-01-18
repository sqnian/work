# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CppExtension

# setup(
#     name='gemm_test',
#     ext_modules=[
#         CppExtension('cublas_gemm_test', ['test.cpp']),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# cpp_files = glob.glob(os.path.join(CUR_DIR,"*.cpp"))
# cu_files = glob.glob(os.path.join(CUR_DIR,'*.cu'))
# source_files = cpp_files + cu_files
# print("source files:")
# for i in source_files:
#     print(i)
source_files = [
    os.path.join(CUR_DIR,'matrix_multi.cpp'),
]

for i in source_files:
    assert os.path.isfile(i)
    print(i)

setup(
    name="gemm_test",
    ext_modules=[
        CUDAExtension(
            name="cublas_gemm_test",
            sources=source_files,),
    ],
    cmdclass={
        "build_ext": BuildExtension
        }
)