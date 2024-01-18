#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>


__global__ void ker_gather(__half *input, int32_t *indice, __half *output, int d0, int d1, int indice_d0, int indice_d1, int d3){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d0_index = idx / (d1*indice_d0*indice_d1*d3);
    int d1_index = idx / (indice_d0*indice_d1*d3) % d1;
    int indice_d0_index = idx / (indice_d1*d3) % indice_d0;
    int indice_d1_index = idx /d3 % indice_d1 ;
    int d3_index = idx % d3;

    int indice_index = indice_d1*indice_d0_index+ indice_d1_index;
    int indice_data = indice[indice_index];
    // printf("indice_index: %d indice_data:%d \n",indice_index, indice_data);
    int in_index = d0_index*d1*d3*d3 + d1_index * d3*d3 + d3*indice_data + d3_index;

    int out_index = d0_index * d1*indice_d0*indice_d1*d3 + d1_index*indice_d0*indice_d1*d3 + indice_d0_index*indice_d1*d3 + indice_d1_index*d3 + d3_index;
    output[idx] = input[in_index];
}

__global__ void ker_gather(__half *input, int32_t *indice, __half *output, int d0, int d1, int d2,  int d3, int indice_d0,int indice_d1){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d0_index = idx / (d1*indice_d0*indice_d1*d3*d2);
    int d1_index = idx / (indice_d0*indice_d1*d3*d2) % d1;
    int d2_index = idx / (indice_d0*indice_d1*d3) % d2;
    int d3_index = idx / (indice_d0*indice_d1)% d3;
    int indice_d0_index = idx / (indice_d1) % indice_d0;
    int indice_d1_index = idx % indice_d1 ;
    

    int indice_index = indice_d1*indice_d0_index+ indice_d1_index;
    int indice_data = indice[indice_index];

    int in_index = d0_index*d1*d2*d3*228 + d1_index * d2*d3*228 + d2_index*d3*228 + d3_index*228 + indice_data;

    int out_index = d0_index * d1*indice_d0*indice_d1*d3 + d1_index*indice_d0*indice_d1*d3 + indice_d0_index*indice_d1*d3 + indice_d1_index*d3 + d3_index;
    output[idx] = input[in_index];
}

void ker_gather_ix_launcher(__half *input, int32_t *indice, __half *output, int d0, int d1, int indice_d0, int indice_d1, int d3,
                                cudaStream_t stream)  {
    int num_blocks = (d0*d1*indice_d0*indice_d1*d3)/4096 + 1; 
    int max_thread = 4096;
    ker_gather<<<num_blocks, max_thread, 0, stream>>>(input, indice, output, d0, d1, indice_d0, indice_d1, d3);
}

void ker_gather_ix_launcher(__half *input, int32_t *indice, __half *output, int d0, int d1, int d2,  int d3, int indice_d0,int indice_d1,
                                cudaStream_t stream)  {
    int num_blocks = (d0*d1*indice_d0*indice_d1*d3*d2)/4096 + 1; 
    int max_thread = 4096;
    ker_gather<<<num_blocks, max_thread, 0, stream>>>(input, indice, output, d0, d1, d2, d3, indice_d0, indice_d1);
}


at::Tensor one_test(at::Tensor input, at::Tensor indice ){

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    // TORCH_CHECK(indice.scalar_type() == at::ScalarType::Int);
    int indice_d0 = indice.size(0);
    int indice_d1 = indice.size(1);
    
    if(input.dim() == 4){
        int d0 = input.size(0);
        int d1 = input.size(1);
        int d2 = input.size(2);
        int d3 = input.size(3);

        at::Tensor res = input.new_empty({d0, d1, indice_d0, indice_d1, d3});
        __half* input_ = (__half*) input.data_ptr();
        int32_t* indice_ = (int32_t*) indice.data_ptr();
        __half* res_ = (__half*) res.data_ptr();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        ker_gather_ix_launcher(input_, indice_, res_, d0, d1, indice_d0, indice_d1, d3, stream);
        printf("run in here!\n");
        return res;
    }else{
        int d0 = input.size(0);
        int d1 = input.size(1);
        int d2 = input.size(2);
        int d3 = input.size(3);
        int d4 = input.size(4);

        at::Tensor res = input.new_empty({d0, d1, d2, d3, indice_d0, indice_d1});
        __half* input_ = (__half*) input.data_ptr();
        int32_t* indice_ = (int32_t*) indice.data_ptr();
        __half* res_ = (__half*) res.data_ptr();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        ker_gather_ix_launcher(input_, indice_, res_, d0, d1, d2, d3, indice_d0, indice_d1, stream);
        printf("run in here!\n");
        return res;
      
    

    }
    
    

}