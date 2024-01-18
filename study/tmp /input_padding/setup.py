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
    os.path.join(CUR_DIR, 'unit_test.cpp'),
    os.path.join(CUR_DIR, 'kernel.cu')
]

for i in source_files:
    assert os.path.isfile(i)
    print(i)

setup(
    name="infer_test",
    ext_modules=[
        CUDAExtension(
            name="conformer_infer_opt",
            sources=source_files,
            libraries=["cuinfer"], # cuinfer
            # extra_compile_args={cuinfer,
            #     "nvcc":[
            #         "-DDYNAMIC_API=ON","-lcuinfer"
                
            #     ],
            #     "cxx":
            #     ["-DDYNAMIC_API=ON","-lcuinfer"],
            #     "x86_64-linux-gnu-g++":
            #     ["-lcuinfer"]
            # }
            ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)