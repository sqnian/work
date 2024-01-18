#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>


__global__ void ker_top1(__half *input, int64_t *output, const int batch_size, 
                          const int max_len, const int vocab_size){
    /*
    dim3 grid(batch_size);
    dim3 block(max_len);
    */
    int input_index = blockIdx.x * blockDim.x * vocab_size + threadIdx.x * vocab_size;
    int batch_idx = input_index / (vocab_size * max_len);
    int seq_idx =  input_index / vocab_size % max_len;
    int output_index = batch_idx * max_len + seq_idx;

    int prev_index = -1; 
    int cur_len = 0;  // 写入的长度

    float val = -std::numeric_limits<float>::infinity();
    long  index = -1;
    for(long  i = input_index; i < input_index + vocab_size; ++i ){

        if (val < __half2float(input[i])){
            val = __half2float(input[i]);
            index = i;
        }
    }
    output[output_index] = index % vocab_size;
    // printf("batch_size: %d, max_len: %d, output_index: %d, output[output_index]: %ld  \n",batch_idx, seq_idx,output_index, output[output_index] );
}


__global__ void ker_remove_blank_duplicate(int64_t *input, int64_t *output, const int batch_size, const int max_len){
    /*
    dim3 grid(batch_size);
    dim3 block(1);
    */
    int input_index = blockIdx.x * max_len + threadIdx.x ;
    int prev_index = -1; 
    int cur_len = input_index;  // 写入的长度
    
    for(int i = input_index; i < input_index + max_len; ++i ){
        if ( input[i] != 0 && prev_index != input[i]) {
            cur_len += 1;
            output[cur_len - 1] = input[i];
        }
        prev_index = input[i];
        if (input[i] == 4232) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        for (int i = cur_len; i < input_index + max_len; ++i) {
            output[i] = 4232;
        }
    }  
}

void ker_top1_ix_launcher(__half *input, int64_t *output, const int batch_size, 
                          const int max_len, const int vocab_size, cudaStream_t stream){
    dim3 grid(batch_size); // 24
    dim3 block(max_len);  // 176
    // ker_top1<<<grid, block, 0, stream>>>(input,output, batch_size, max_len, vocab_size);
    ker_top1<<<grid, block, 0, stream>>>(input,output, batch_size, max_len, vocab_size);
    cudaDeviceSynchronize();
    
}

void ker_remove_blank_duplicate_ix_launcher(int64_t *input, int64_t *output, const int batch_size, 
                          const int max_len, cudaStream_t stream){
    dim3 grid(batch_size); // 24
    dim3 block(1); 
    ker_remove_blank_duplicate<<<grid, block, 0, stream>>>(input,output, batch_size, max_len);
    cudaDeviceSynchronize();
    
}


at::Tensor one_test(at::Tensor input, at::Tensor output, int K) {
    //  input : batch_size max_len vocab_size
    TORCH_CHECK(input.dim() == 3); // 24, 176 4233
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    int batch_size = input.size(0);
    int max_len = input.size(1);
    int vocab_size = input.size(2);

    at::Tensor output_res1 = output.new_zeros({batch_size,max_len});
    at::Tensor output_res2 = output.new_zeros({batch_size,max_len});

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    __half* input_ptr = (__half*) input.data_ptr();
    int64_t *output_ptr1 = (int64_t *) output_res1.data_ptr();
    int64_t *output_ptr2 = (int64_t *) output_res2.data_ptr();

    ker_top1_ix_launcher(input_ptr, output_ptr1, batch_size, max_len, vocab_size, stream);
    cudaDeviceSynchronize();
    ker_remove_blank_duplicate_ix_launcher(output_ptr1, output_ptr2, batch_size, max_len, stream);

    return output_res2;

  }