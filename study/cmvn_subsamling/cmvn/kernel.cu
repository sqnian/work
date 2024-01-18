#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>

// #include "backend/bert/bert_embed_kernel.h"
/*
GlobalCMVN
x: batch, max_len, feat_dim
mean: feat_dim
istd: feat_dim

x = x - mean
x = x * istd
*/

__forceinline__ __device__ int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  // int32_t i8 = floorf(i8_f);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

template <typename T>
__global__ void ker_global_cmvn(T* f_inputs, T* inputs, int8_t* inputs_,const T* mean,
                                const T* istd,const float amax) {
  const float quant_scale = 127.0 / amax ;
  inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  inputs[threadIdx.x] =
      (f_inputs[threadIdx.x] - mean[threadIdx.x]) * istd[threadIdx.x];
  inputs_[threadIdx.x] =float2int8(inputs[threadIdx.x] ,quant_scale);
}

template <>
__global__ void ker_global_cmvn<__half>(__half* f_inputs, __half* inputs,int8_t* inputs_,
                                        const __half* mean,
                                        const __half* istd,
                                        const float amax) {
  const float quant_scale = 127 / amax ;
  // printf("quant_scale:%f\n",quant_scale);

  inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
 
  inputs[threadIdx.x] =__hmul(__hsub((f_inputs[threadIdx.x]), mean[threadIdx.x]),
             istd[threadIdx.x]);

  inputs_[threadIdx.x] = float2int8(__half2float(inputs[threadIdx.x] ),quant_scale);
  
}


template <typename T>
void ker_global_cmvn_ix_launcher(T* f_inputs, T* inputs, int8_t* inputs_, const T* mean,
                                 const T* istd, const int batch_size,
                                 const int seq_len, const int hidden_size,
                                 cudaStream_t stream,const float amax) {
  if (hidden_size > 4096) {
    throw std::runtime_error("hidden_size should <= 4096");
  }
  dim3 grid(batch_size, seq_len);
  dim3 block(hidden_size);
  
  ker_global_cmvn<<<grid, block, 0, stream>>>(f_inputs, inputs, inputs_, mean, istd,amax);
}

at::Tensor one_test(at::Tensor input,  at::Tensor input_int8, at::Tensor mean,
                    at::Tensor istd, const float amax, bool fp16) {
  // input: fp16 [batch, max_len, feat_dim]
  TORCH_CHECK(input.dim() == 3);
  int batch_size = input.size(0);
  int max_len = input.size(1);
  int feat_dim = input.size(2);

  if(fp16){
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(istd.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(mean.dim() == 1);
    TORCH_CHECK(istd.dim() == 1);

  }else{
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(istd.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(mean.dim() == 1);
    TORCH_CHECK(istd.dim() == 1);
  }
 
  at::Tensor output = input.new_empty({batch_size, max_len, feat_dim});
  at::Tensor output2 = input_int8.new_empty({batch_size, max_len, feat_dim});

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if(fp16){
    printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();
    
    __half* mean_ = (__half*) mean.data_ptr();
    __half* istd_ = (__half*) istd.data_ptr();

    int8_t * output_2 = (int8_t *) output2.data_ptr();

    ker_global_cmvn_ix_launcher(input_,output_,output_2,mean_,istd_,batch_size,max_len,feat_dim,stream,amax);
  
    return output2;

  }else{
    printf(" not use fp16\n");
    float* input_ = (float*) input.data_ptr();
    float* output_ = (float*) output.data_ptr();
    int8_t * output_2 = (int8_t *) output2.data_ptr();
    float* mean_ = (float*) mean.data_ptr();
    float* istd_ = (float*) istd.data_ptr();
  
    ker_global_cmvn_ix_launcher(input_,output_,output_2,mean_,istd_,batch_size,max_len,feat_dim,stream,amax);
    return output2;
  }
}