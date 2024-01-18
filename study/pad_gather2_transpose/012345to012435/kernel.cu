#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>


__global__ void ker_012345to012435(__half *input, __half *output, int d0, int d1, int d2, int d3, int d4, int d5){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d0_index = idx / (d1*d2*d3*d4*d5);
    int d1_index = idx / (d2*d3*d4*d5) % d1;
    int d2_index = idx / (d3*d4*d5) % d2;
    int d3_index = idx /(d4*d5)% d3 ;
    int d4_index = idx / (d5) % d4;
    int d5_index = idx % d5;
    // 012345 -->012435
    int out_index = d0_index*d1*d2*d3*d4*d5 + d1_index*d2*d3*d4*d5 + d2_index*d3*d4*d5 + d4_index*d3*d5 + d3_index*d5 + d5_index;
    output[out_index] = input[idx];
}

void ker_012345to012435_ix_launcher(__half *input, __half *output, int d0, int d1, int d2, int d3, int d4, int d5,
                                cudaStream_t stream)  {
    int num_blocks = (d0*d1*d2*d3*d4*d5)/4096 + 1; 
    int max_thread = 4096;
    ker_012345to012435<<<num_blocks, max_thread, 0, stream>>>(input, output, d0, d1, d2, d3, d4, d5);
}

at::Tensor one_test(at::Tensor input ){
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(input.dim() == 6);
    // 
    int d0 = input.size(0);
    int d1 = input.size(1);
    int d2 = input.size(2);
    int d3 = input.size(3);
    int d4 = input.size(4);
    int d5 = input.size(5);

    at::Tensor res = input.new_empty({d0, d1, d2, d4, d3, d5});
    __half* input_ = (__half*) input.data_ptr();
    __half* res_ = (__half*) res.data_ptr();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ker_012345to012435_ix_launcher(input_, res_, d0, d1, d2, d3, d4, d5, stream);
    printf("run in here!\n");

    return res;

}