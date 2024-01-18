
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <time.h>

/* 
gemm_compute_fp32 
入参：A ，B，C
过程：根据入参，获取cublasGemmEx（）需要的所有参数，交给函数执行
出参：得到矩阵相乘的结果
 cublasSgemm(
            cublasHandle_t handle,
            cublasOperation_t transa,   // CUBLAS_OP_T 矩阵A的属性参数，转置，按行优先
            cublasOperation_t transb,   // CUBLAS_OP_T 矩阵B的属性参数，转置，按行优先
            int M,             // 矩阵A行数、矩阵C行数
            int N,             // 矩阵B列数、矩阵C列数
            int K,             // 矩阵A列数、矩阵B行数
            const void *a,            // alpha的值
            const void *d_A,           // 左矩阵，为A
            cudaDataType_t AType,         // A 的类型
            int K,             // A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
            const void *d_B,           // 右矩阵，为B
            cudaDataType_t BType,         // B 的类型
            int N,             // B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
            &b,            // beta的值
            const void *d_C,           // 结果矩阵C
            cudaDataType_t CType,         // C 的类型
            int M,              // C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
            cudaDataType   computeType,,
            cublasGemmAlgo_t algo);

*/
at::Tensor gemm_compute_fp32(
    at::Tensor const d_A,  // m x k
    at::Tensor const d_B   // k X n
     )
{
    /*
    origin:
    A : m x k
    B : k x n
    C : m x n

    convert cublasgemmex ,
    int M,             // 矩阵A行数、矩阵C行数
    int N,             // 矩阵B列数、矩阵C列数
    int K,             // 矩阵A列数、矩阵B行数
    */
    int  m = d_A.size(0) ; //  2
    int  n = d_B.size(0) ;  // 4
    int  k = d_A.size(1) ;  // 3
    float alpha = 1.0;
    float beta = 0.0;
    
    TORCH_CHECK(d_A.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(d_B.scalar_type() == at::ScalarType::Float);

    TORCH_CHECK(d_A.size(1) == k);

    at::Tensor d_C = d_A.new_empty({m,n});  

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // weight 需要转置
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,  
        CUBLAS_OP_N,
        n,
        m,
        k,
        static_cast<const void *>(&alpha),
        static_cast<const void*>(d_B.data_ptr()),
        CUDA_R_32F,
        k,
        static_cast<const void*>(d_A.data_ptr()),
        CUDA_R_32F,
        k,
        static_cast<const void *>(&beta),
        static_cast<void*>(d_C.data_ptr()),
        CUDA_R_32F,
        n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(handle);
    return d_C;
}

at::Tensor gemm_compute_fp16(
    at::Tensor const d_A,  // m x k
    at::Tensor const d_B   // k X n
     )
{
    /*
    origin:
    A : m x k
    B : k x n
    C : m x n

    convert cublasgemmex ,
    int M,             // 矩阵A行数、矩阵C行数
    int N,             // 矩阵B列数、矩阵C列数
    int K,             // 矩阵A列数、矩阵B行数
    */
    int  m = d_A.size(0) ; //  2
    int  n = d_B.size(0) ;  // 4
    int  k = d_A.size(1) ;  // 3
    float alpha = 1.0;
    float beta = 0.0;
    
    TORCH_CHECK(d_A.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(d_B.scalar_type() == at::ScalarType::Half);

    TORCH_CHECK(d_B.size(1) == k);

    at::Tensor d_C = d_A.new_empty({m,n});  

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // weight 需要转置
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,  
        CUBLAS_OP_N,
        n,
        m,
        k,
        static_cast<const void *>(&alpha),
        static_cast<const void*>(d_B.data_ptr()),
        CUDA_R_16F,
        k,
        static_cast<const void*>(d_A.data_ptr()),
        CUDA_R_16F,
        k,
        static_cast<const void *>(&beta),
        static_cast<void*>(d_C.data_ptr()),
        CUDA_R_16F,
        n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(handle);
    return d_C;
}

at::Tensor gemm_compute_fp16_AT(at::Tensor const weight,
                                at::Tensor const input) {
  // weight: inner_size, hidden_size --> hidden_size, inner_size
  // input: batch_size, hidden_size  --> hidden_size, batch_size
  float alpha = 1.0;
  float beta = 0.0;

  int batch_size = input.size(0);
  int hidden_size = input.size(1);
  int inner_size = weight.size(0);

  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

  TORCH_CHECK(weight.size(1) == hidden_size);

  at::Tensor res = weight.new_empty({batch_size, inner_size});

  __half *weight_ptr = (__half *)weight.data_ptr();
  __half *input_ptr = (__half *)input.data_ptr();
  __half *res_ptr = (__half *)res.data_ptr();

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inner_size, batch_size,
               hidden_size, &alpha, weight_ptr, CUDA_R_16F, hidden_size,
               input_ptr, CUDA_R_16F, hidden_size, &beta, res_ptr, CUDA_R_16F,
               inner_size, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cublasDestroy(handle);
  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_fp32", &gemm_compute_fp32,
          "fast depthwise conv1d forward (cuda)");
    m.def("test_fp16", &gemm_compute_fp16,
          "fast depthwise conv1d forward (cuda)");
    m.def("test_fp16_at", &gemm_compute_fp16_AT,
          "fast depthwise conv1d forward (cuda)");
}