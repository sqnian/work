import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = CUR_DIR.split("/")[:-4]
# ROOT_DIR = "/".join(ROOT_DIR)

# include_dirs = []
# include_dirs.append(os.path.join(ROOT_DIR, "runtime/src/include"))

# cu_file = os.path.join(ROOT_DIR, "runtime/src/impl/backend/bert/bert_embed_kernel.cu")

source_files = [
    os.path.join(CUR_DIR, "unit_test.cpp"),
    os.path.join(CUR_DIR, "kernel.cu"),
    # cu_file,
]

for i in source_files:
    if not os.path.isfile(i):
        print(f"not found: {i} ")
    assert os.path.isfile(i)
# for i in include_dirs:
#     if not os.path.isdir(i):
#         print(f"not found: {i} ")
#     assert os.path.isdir(i)

setup(
    name="test",
    ext_modules=[
        CUDAExtension(name="gemm_test", sources=source_files, libraries=["cuinfer"],),
        
    ],
    cmdclass={"build_ext": BuildExtension},
)