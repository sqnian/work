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
__global__ void ker_0123to0132(T *input_, T *output_,
                                const int N,  const int C,const int H, const int  W){

    /*
    dim3 grid( N*C*H*W / 4096 + 1 );  
    dim3 block(4096); 
    */
    
    // 一维数据进行处理方式
    int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
    // // 顺序读取数据
    // output_[input_index] = input_[input_index];
    int n_index = input_index / (H*W*C) ; // 24
    int h_index = input_index / (W*C) % H ;  // 17
    int w_index = input_index / (C) % W ;   // /25 , 19
    int c_index = input_index  % (C);   // %25 ,25
    int output_index = n_index * H * W * C + h_index * W * C + W * c_index + w_index ;
    output_[output_index] = input_[input_index];
}

template <>
__global__ void ker_0123to0132<__half>(__half *input_, __half *output_,
                                const int N,  const int C,const int H, const int  W){

    /*
    dim3 grid( N*C*H*W / 4096 + 1 );  
    dim3 block(4096); 
    */
    
    // 一维数据进行处理方式
    int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
    // // 顺序读取数据
    // output_[input_index] = input_[input_index];
    int n_index = input_index / (H*W*C) ; // 24
    int h_index = input_index / (W*C) % H ;  // 17
    int w_index = input_index / (C) % W ;   // /25 , 19
    int c_index = input_index  % (C);   // %25 ,25
    int output_index = n_index * H * W * C + h_index * W * C + W * c_index + w_index ;
    output_[output_index] = input_[input_index];
}

template <typename T>
void ker_0123to0132_ix_launcher(T *input_, T *output_,
                                const int N,  const int C,const int H, const int  W,
                                cudaStream_t stream)  {
    int grid_size = N*C*H*W / 4096 + 1 ;
    dim3 grid(grid_size);  
    // dim3 block(4096);   
    // dim3 grid(N,H);  // 
    dim3 block(4096); 
    // dim3 block(W,C);   
    
    ker_0123to0132<<<grid, block, 0, stream>>>(input_, output_, N, C, H, W);
}

at::Tensor one_test(at::Tensor input) {

    TORCH_CHECK(input.dim() == 4);
    int n = input.size(0);
    int h = input.size(1);
    int w = input.size(2);
    int c = input.size(3);

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

 
    at::Tensor output = input.new_zeros({n,h,c,w});

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

 
    // printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    ker_0123to0132_ix_launcher(input_,output_, n, c, h, w,stream);
  
    return output;

  
  }