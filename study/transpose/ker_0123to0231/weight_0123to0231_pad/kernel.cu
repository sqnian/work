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
__global__ void ker_weight_pad_trans(T* inputs_, T* outputs_) {

  // dim3 grid(cout, kernel);
  // dim3 block(kernel,cin);
  inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  int cout = blockIdx.x;  // 
  int kernel_h = blockIdx.y;   // 
  int kernel = threadIdx.x;  // 

  int out_index = cout * blockDim.x * gridDim.y *4  +  blockDim.x * kernel_h  *4 +  kernel*4  +  threadIdx.y *4  ;
  outputs_[out_index] = inputs_[threadIdx.x];


  // dim3 grid(cout, cin);
    // dim3 block(kernel,kernel);
  // inputs_ += blockIdx.x * gridDim.y * blockDim.x * blockDim.y + blockIdx.y * blockDim.x  * blockDim.y + threadIdx.x *blockDim.x;

  // int cout = blockIdx.x;  // 
  // int cin = blockIdx.y;
  // int kernel_h =  threadIdx.x;   // 
  // int kernel = threadIdx.y;  // 
  
  // int out_index = cout * blockDim.x * gridDim.y * blockDim.y* 64 +  cin * 64 +  kernel_h * gridDim.y * blockDim.x* 64 +  kernel * gridDim.y * 64;
  // outputs_[out_index] = inputs_[threadIdx.y];
}

template <>
__global__ void ker_weight_pad_trans<__half>(__half* inputs_, __half* outputs_) {


  // dim3 grid(cout, kernel);
  // dim3 block(kernel,cin);
  inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  
  int cout = blockIdx.x;  // 
  int kernel_h = blockIdx.y;   // 
  int kernel = threadIdx.x;  // 
  
  int out_index = cout * blockDim.x * gridDim.y *4 +  blockDim.x * kernel_h *4  +  kernel *4 +  threadIdx.y*4  ;
  outputs_[out_index] = inputs_[threadIdx.x];

  // dim3 grid(cout, cin);
    // dim3 block(kernel,kernel);
  // inputs_ += blockIdx.x * gridDim.y * blockDim.x * blockDim.y + blockIdx.y * blockDim.x  * blockDim.y + threadIdx.x *blockDim.x;
  
  // int cout = blockIdx.x;  // 
  // int cin = blockIdx.y;
  // int kernel_h =  threadIdx.x;   // 
  // int kernel = threadIdx.y;  // 
  
  // int out_index = cout * blockDim.x * gridDim.y * blockDim.y * 64 +  cin * 64 +  kernel_h * gridDim.y * blockDim.x* 64 +  kernel * gridDim.y * 64;
  // outputs_[out_index] = inputs_[threadIdx.y];
}


template <typename T>
void  ker_weight_pad_trans_ix_launcher(T *input_, T *output_, 
                                        const int cout,const int  cin, const int kernel,
                                        cudaStream_t stream){

    dim3 grid(cout, kernel);
    dim3 block(kernel,cin);
    // dim3 grid(cout, cin);
    // dim3 block(kernel,kernel);
    
    ker_weight_pad_trans<<<grid, block, 0, stream>>>(input_, output_);

}


at::Tensor one_test(at::Tensor input) {

  // input: fp16 [cout, cin, kernel,kernel]

    TORCH_CHECK(input.dim() == 4);
    int cout = input.size(0);  // 256
    int cin = input.size(1);  // 1
    int kernel = input.size(2);   //3

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

 
    at::Tensor output = input.new_zeros({cout,kernel,kernel,4});

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

 
    // printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    ker_weight_pad_trans_ix_launcher(input_,output_, cout, cin, kernel,stream);
  
    return output;

  
  }