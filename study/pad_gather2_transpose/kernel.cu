#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>


__global__ void ker_pad(__half *input, __half *output, int n, int c, int h, int w, int flag){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_index = idx / (h*c*w);
    int c_index = idx / (h*w) % c;
    int h_index = idx / w % h;
    int w_index = idx % w ;
    int out_index = n_index*(flag+flag+w)*(flag+flag+h)*c + c_index*((flag+flag+w))*((flag+flag+h)) + h_index*((flag+flag+w)) + w_index + flag*((flag+flag+w)) + flag;
    // int out_index = n_index*228*228*c + c_index*228*228 + h_index*228 + w_index + 2*228 + 2;
    output[out_index] = input[idx];
}

void ker_pad_ix_launcher(__half *input, __half *output, int n, int c, int h, int w, int flag,
                                cudaStream_t stream)  {
    int num_blocks = (n*c*h*w)/4096 + 1; 
    int max_thread = 4096;
    ker_pad<<<num_blocks, max_thread, 0, stream>>>(input, output, n, c, h, w, flag);
}



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

__global__ void ker_gather(__half *input, int32_t *indice, __half *output, int d0, int d1, int d2,  int d3, int indice_d0,int indice_d1, int dd){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d0_index = idx / (d1*indice_d0*indice_d1*d3*d2);
    int d1_index = idx / (indice_d0*indice_d1*d3*d2) % d1;
    int d2_index = idx / (indice_d0*indice_d1*d3) % d2;
    int d3_index = idx / (indice_d0*indice_d1)% d3;
    int indice_d0_index = idx / (indice_d1) % indice_d0;
    int indice_d1_index = idx % indice_d1 ;
    

    int indice_index = indice_d1*indice_d0_index+ indice_d1_index;
    int indice_data = indice[indice_index];

    int in_index = d0_index*d1*d2*d3*dd + d1_index * d2*d3*dd + d2_index*d3*dd + d3_index*dd + indice_data;

    int out_index = d0_index * d1*indice_d0*indice_d1*d3 + d1_index*indice_d0*indice_d1*d3 + indice_d0_index*indice_d1*d3 + indice_d1_index*d3 + d3_index;
    output[idx] = input[in_index];
}

void ker_gather_ix_launcher(__half *input, int32_t *indice, __half *output, int d0, int d1, int indice_d0, int indice_d1, int d3,
                                cudaStream_t stream)  {
    int num_blocks = (d0*d1*indice_d0*indice_d1*d3)/4096 + 1; 
    int max_thread = 4096;
    ker_gather<<<num_blocks, max_thread, 0, stream>>>(input, indice, output, d0, d1, indice_d0, indice_d1, d3);
}

void ker_gather_ix_launcher(__half *input, int32_t *indice, __half *output, int d0, int d1, int d2,  int d3, int indice_d0,int indice_d1, int dd,
                                cudaStream_t stream)  {
    int num_blocks = (d0*d1*indice_d0*indice_d1*d3*d2)/4096 + 1; 
    int max_thread = 4096;
    ker_gather<<<num_blocks, max_thread, 0, stream>>>(input, indice, output, d0, d1, d2, d3, indice_d0, indice_d1,dd);
}

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

at::Tensor one_test(at::Tensor input, at::Tensor indice1, at::Tensor indice2 ){

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(input.dim() == 4);
    // 
    int d0 = input.size(0);
    int d1 = input.size(1);
    int d2 = input.size(2);
    int d3 = input.size(3);
    int d0_in = indice1.size(0);
    int d1_in = indice1.size(1);

    int flag;
    if (d1 == 3){
        flag = 2;
    }else{
        flag = 1;
    }
    printf("flag:%d\n",flag);
    int h_size = d2 + flag*2;
    int w_size = d3 + flag*2;
   
    at::Tensor res_pad = input.new_zeros({d0, d1, h_size, w_size});
    __half* input_ = (__half*) input.data_ptr();
    __half* res_pad_ = (__half*) res_pad.data_ptr();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ker_pad_ix_launcher(input_, res_pad_, d0, d1, d2, d3, flag, stream);
    
    at::Tensor res_gather1 = input.new_empty({d0, d1, d0_in, d1_in, w_size});
    int32_t* indice_1 = (int32_t*) indice1.data_ptr();
    __half* res_gather1_ = (__half*) res_gather1.data_ptr();
    ker_gather_ix_launcher(res_pad_, indice_1, res_gather1_, d0, d1, d0_in, d1_in, w_size, stream);
    
    at::Tensor res_gather2 = input.new_empty({d0, d1, d0_in, d1_in, d0_in, d1_in});
    int32_t* indice_2 = (int32_t*) indice2.data_ptr();
    __half* res_gather2_ = (__half*) res_gather2.data_ptr();
    ker_gather_ix_launcher(res_gather1_, indice_2, res_gather2_, d0, d1, d0_in, d1_in, d0_in, d1_in,w_size, stream);

    at::Tensor res = input.new_empty({d0, d1, d0_in, d0_in, d1_in, d1_in});
    __half* res_ = (__half*) res.data_ptr();
    
    ker_012345to012435_ix_launcher(res_gather2_, res_, d0, d1, d0_in, d1_in, d0_in, d1_in, stream);
    printf("run in here!\n");
    return res;

}