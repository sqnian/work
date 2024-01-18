#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>
#include <vector>

at::Tensor one_test(at::Tensor input,at::Tensor input_int8, at::Tensor mean,
                             at::Tensor istd, const float amax,bool fp16);

at::Tensor test(at::Tensor input,at::Tensor input_int8,at::Tensor mean,
                             at::Tensor istd, const float amax,bool fp16) {
    return one_test(input,input_int8, mean, istd, amax, fp16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_cmvn", &test, "use  cmvn op");
}