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


at::Tensor one_test(at::Tensor input ){

    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(input.dim() == 4);
    // // input shape: N C  H W 
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int flag;
    if (H == 224){
        flag = 2;
    }else{
        flag = 1;
    }
    printf("flag:%d\n",flag);
    int h_size = H + flag*2;
    int w_size = W + flag*2;
    printf("h_size:%d\n",h_size);
    printf("w_size:%d\n",w_size);

    at::Tensor res = input.new_zeros({N, C, h_size, w_size});
    __half* input_ = (__half*) input.data_ptr();
    __half* res_ = (__half*) res.data_ptr();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ker_pad_ix_launcher(input_, res_, N, C, H, W, flag, stream);
    printf("run in here!\n");

    return res;

}