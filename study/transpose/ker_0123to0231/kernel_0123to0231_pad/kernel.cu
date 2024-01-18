
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>




template <typename T>
__global__ void ker_0123to0231_pad(T *f_inputs, T *inputs) {
    // inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
    f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
    int batch_idx = blockIdx.x;  
    int seq_idx = blockIdx.y;   
    int hidden_idx = threadIdx.x;  
    // int out_index = batch_idx * gridDim.y * blockDim.x * 64 + hidden_idx * gridDim.y * 64  ;
    int out_index = batch_idx * blockDim.x * gridDim.y* 4 +  blockDim.x * seq_idx * 4 + 4 * hidden_idx ;
    inputs[out_index] = f_inputs[threadIdx.x];
}

template <>
__global__ void ker_0123to0231_pad<__half>(__half *f_inputs, __half *inputs) {
    // inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
    f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;

    int batch_idx = blockIdx.x;  
    int seq_idx = blockIdx.y;   
    int hidden_idx = threadIdx.x;  
    // int out_index = batch_idx * gridDim.y * blockDim.x * 64 + hidden_idx * gridDim.y * 64  ;
    int out_index = batch_idx * blockDim.x * gridDim.y* 4 +  blockDim.x * seq_idx * 4 + 4 * hidden_idx ;

    inputs[out_index] = f_inputs[threadIdx.x];
}

template <typename T>
void ker_0123to0231_pad_ix_launcher(T *f_inputs, T *inputs, const int batch_size,
                                 const int seq_len, const int hidden_size, cudaStream_t stream) {
    if (hidden_size > 4096) {
        throw std::runtime_error("hidden_size should <= 4096");
    }
    dim3 grid(batch_size, seq_len);
    dim3 block(hidden_size);
    ker_0123to0231_pad<<<grid, block, 0, stream>>>(f_inputs, inputs);
}



at::Tensor one_test(at::Tensor input) {

  // input: fp16 [batch_size, cin, seq_len, hidden_size]

    TORCH_CHECK(input.dim() == 4);
    int batch_size = input.size(0);  // 256
    int cin = input.size(1);  // 1
    int seq_len = input.size(2);   //3
    int hidden_size =  input.size(3);

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

 
    at::Tensor output = input.new_zeros({batch_size,seq_len,hidden_size,4});

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

 
    // printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    ker_0123to0231_pad_ix_launcher(input_,output_,batch_size,seq_len,hidden_size,stream);
  
    return output;

  
  }