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


__global__ void ker_0123to0231(__half *input, float *output){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = __half2float(input[idx]);
}

void ker_0123to0231_ix_launcher(__half *input, float *output,
                                cudaStream_t stream)  {
    int max_thread = 4096;
    int num_blocks = 256;  
    
    ker_0123to0231<<<num_blocks, max_thread, 0, stream>>>(input, output);
}


at::Tensor one_test(at::Tensor input, at::Tensor output ){

    // TORCH_CHECK(input.scalar_type() == at::ScalarType::Int8_t);
    // TORCH_CHECK(input.dim() == 4);
    // // input shape: N C  H W 
    // int N = input.size(0);
    // int C = input.size(1);
    // int H = input.size(2);
    // int W = input.size(3);
    

    at::Tensor res = output.new_empty({256});
    __half* input_ = (__half*) input.data_ptr();
    float* res_ = (float*) res.data_ptr();

    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ker_0123to0231_ix_launcher(input_, res_, stream);
    printf("run in here!\n");

    return res;

}