#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>


__global__ void ker_concat(__half *input, __half *input2, __half *output, int d0, int d1, int d2 ){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx];
    for(int i=0; i < d2; i++){
        output[d1*d2 +i] = input2[i];
    }
}

void ker_concat_ix_launcher(__half *input, __half *input2, __half *output, int d0, int d1, int d2, 
                                cudaStream_t stream)  {
    int num_blocks = (d0*(d1)*d2)/4096 + 1; 
    int max_thread = 4096;
    ker_concat<<<num_blocks, max_thread, 0, stream>>>(input, input2,output, d0, d1, d2);
}


at::Tensor one_test(at::Tensor input, at::Tensor input2){

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(input2.dim() == 3);

    int d0 = input.size(0);
    int d1 = input.size(1);
    int d2 = input.size(2);
    int d0_2 = input2.size(0);
    int d1_2 = input2.size(1);
    int d2_2 = input2.size(2);

    at::Tensor res = input.new_zeros({d0, d1+d1_2, d2});
    __half* input_ = (__half*) input.data_ptr();
    __half* input_2 = (__half*) input2.data_ptr();
    __half* res_ = (__half*) res.data_ptr();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ker_concat_ix_launcher(input_, input_2, res_, d0,d1,d2, stream);
    printf("run in here!\n");

    return res;

}