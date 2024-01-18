#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>


#ifndef C10_WARP_SIZE

#ifdef __ILUVATAR__
#define C10_WARP_SIZE 64
#else
#define C10_WARP_SIZE 32
#endif

#endif

#ifndef WARP_SIZE
#ifdef __ILUVATAR__
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif


__global__ void ker_0123to0132(int8_t *input_, __half *output_,
                                const int N,  const int C,const int H, const int  W,const float amax){

    /*
    dim3 grid( N*C*H*W / 4096 + 1 );  
    dim3 block(4096); 
    */
    const float descale = amax / 127;
    
    // 一维数据进行处理方式
    int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
    // 顺序读取数据
    // output_[input_index] = input_[input_index];
    int n_index = input_index / (H*W*C) ; // 24
    int h_index = input_index / (W*C) % H ;  // 17
    int w_index = input_index / (C) % W ;   // /25 , 19
    int c_index = input_index  % (C);   // %25 ,25
    int output_index = n_index * H * W * C + h_index * W * C + W * c_index + w_index ;
    output_[output_index] = __float2half(float(input_[input_index])*descale);

    /*
    grid(N,H)
    block(W,C)
    */
    
    // int n_index = blockIdx.x;
    // int h_index =blockIdx.y ;
    // int w_index = threadIdx.x;
    // int c_index = threadIdx.y;

    // int input_index = n_index * H * W * C + h_index*W*C + w_index * C  + c_index;
    // int output_index = n_index * H * W * C + h_index*W*C + w_index  + c_index *W;


    // output_[output_index] = input_[input_index];

    /*
    grid(N,H)
    block(4096)
    */
    // int input_index = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x ;
    // int n_index = input_index / (H*W*C) ; //blockIdx.x;  // input_index / (H*W*C)  ;  
    // int h_index =input_index / (W*C) % H; // blockIdx.y ; //  input_index / (W*C) % H ;
    // int w_index = input_index / C % W ; // 
    // int c_index = input_index % C ;

    // // int input_index = n_index * H * W * C + h_index*W*C + w_index * C  + c_index;
    // int output_index = n_index * H * W * C + h_index * W * C + c_index * W + w_index  ;

    // output_[output_index] = __float2half(float(input_[input_index])*descale);

 
    
}

void ker_0123to0132_ix_launcher(int8_t *input_, __half *output_,
                                const int N,  const int C,const int H, const int  W,
                                cudaStream_t stream, const float amax)  {
    int grid_size = N*C*H*W / 4096 + 1 ;
    dim3 grid(grid_size);  
    dim3 block(4096);   
    // dim3 grid(N,H);  // 
    // dim3 block(W,C);   
    // dim3 block(4096);
    ker_0123to0132<<<grid, block, 0, stream>>>(input_, output_, N, C, H, W, amax);
}


at::Tensor one_test(at::Tensor input, at::Tensor output, const float amax ){

    // TORCH_CHECK(input.scalar_type() == at::ScalarType::Int8_t);
    TORCH_CHECK(input.dim() == 4);
    // input shape: N H W C 
    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int C = input.size(3);

    at::Tensor res = output.new_empty({N, H, C, W});
    int8_t* input_ = (int8_t*) input.data_ptr();
    __half* res_ = (__half*) res.data_ptr();

    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ker_0123to0132_ix_launcher(input_, res_, N, C, H, W, stream, amax);

    return res;
   

}