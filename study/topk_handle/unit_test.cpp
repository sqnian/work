#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>

at::Tensor one_test(at::Tensor input, at::Tensor output, int K) ;

at::Tensor test_top1(at::Tensor input, at::Tensor output, int K) {
    return one_test(input, output, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_func", &test_top1, "use  top1 op");
}