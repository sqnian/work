#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>
#include "ixinfer.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cassert>
#include <iostream>

at::Tensor one_test(at::Tensor input, at::Tensor weight);

at::Tensor test(at::Tensor input, at::Tensor weight) {
    return one_test(input, weight);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_gemm", &test, "test gemm function");
}