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
    os.path.join(CUR_DIR, 'relu.cpp'),
    os.path.join(CUR_DIR, 'relu_kernel.cu')
]

for i in source_files:
    assert os.path.isfile(i)
    print(i)

setup(
    name="Conv_relu",
    ext_modules=[
        CUDAExtension(
            name="conv_relu",
            sources=source_files,
            libraries=["cudnn"]
            ),
    ],
    cmdclass={
        "build_ext": BuildExtension
        }
)