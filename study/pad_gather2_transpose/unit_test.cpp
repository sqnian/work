#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>

at::Tensor one_test(at::Tensor input, at::Tensor indice1, at::Tensor indice2);

at::Tensor test(at::Tensor input, at::Tensor indice1, at::Tensor indice2) {
    return one_test(input, indice1, indice2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_pg2t", &test, "test pad gather2 transpose function");
}