
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



at::Tensor one_test(at::Tensor input, at::Tensor weight,  at::Tensor bias){

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::Half);
    cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
    cuinferOperation_t transa = CUINFER_OP_T;
    cuinferOperation_t transb = CUINFER_OP_N;
    cudaDataType_t Atype = CUDA_R_16F;
    cudaDataType_t Btype = CUDA_R_16F;
    cudaDataType_t Ctype = CUDA_R_16F;
    cudaDataType_t computeType = CUDA_R_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cuinferGEMMCustomOption_t customOption;
    customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cuinferHandle_t handle;
    cuinferCreate(&handle);
    float gemm_alpha = 1.0f;
    float gemm_beta = 0.f;

    int strideA = 0;
    int strideB = 0;
    int strideC = 0;
    at::Tensor output = input.new_empty({input.size(0), input.size(1),weight.size(0)});

    int m = weight.size(0);
    int n = input.size(1)*input.size(0);
    int k = weight.size(1);

    std::cout << "m: " << m << " n: " << n << " k: " << k <<std::endl;
    int lda = k;
    int ldb = k;
    int ldc = m;  

    __half *wei_ptr = (__half *)weight.data_ptr();
    __half *inp_ptr = (__half *)input.data_ptr();
    __half *bias_ptr = (__half *)bias.data_ptr();
    __half *res_ptr = (__half *)output.data_ptr();

    int batch_count = input.size(0);
    std::cout << "batch_count: " << batch_count << std::endl;

    auto status =
        cuinferCustomGemm(handle, stream, cuinfer_ptr_mode, transa, transb, m, n, k, &gemm_alpha, wei_ptr, Atype,
                          lda, strideA, inp_ptr, Btype, ldb, strideB, &gemm_beta, res_ptr, Ctype, ldc,
                          strideC, batch_count, computeType, scaleType, nullptr,  (void *)bias_ptr, customOption);
    if (status != CUINFER_STATUS_SUCCESS) {
        throw std::runtime_error("cuinferCustomGemm error!");
    }

    return output;

}