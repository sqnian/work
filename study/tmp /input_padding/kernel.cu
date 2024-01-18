#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>

// template <typename T>
__global__ void ker_input_pad(half *f_inputs, half *inputs, const int batch_size,
                                 const int seq_len, const int hidden_size,const int padding) {
    // dim3 grid(batch_size,seq_len);
    // dim3 block(hidden_size,cin);
    // f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
    // int batch_idx = blockIdx.x;  
    // int seq_idx = blockIdx.y;   
    // int hidden_idx = threadIdx.x;  
    // int out_index = batch_idx * blockDim.x * gridDim.y *32 +  blockDim.x * seq_idx*32  +  hidden_idx*32  +  threadIdx.y *32 ;
    // inputs[out_index] = f_inputs[threadIdx.x];
  
    // dim3 grid(batch_size*seq_len*hidden_size/ 4096 + 1);
    // dim3 block(4096);
    int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
    int batch_idx = input_index / (seq_len*hidden_size);
    int seq_idx = input_index / hidden_size % seq_len;
    int hidden_idx = input_index % hidden_size;
    int out_index = batch_idx * seq_len * hidden_size  * padding  +  hidden_size * seq_idx * padding  +  hidden_idx * padding ;
    int out_index_ = out_index + padding;
    inputs[out_index] = f_inputs[input_index];
    

}

__global__ void ker_input_pad_half2(half *f_inputs, half *inputs, const int batch_size,
                                 const int seq_len, const int hidden_size,const int padding) {
    // dim3 grid(batch_size,seq_len);
    // dim3 block(hidden_size,cin);
    // f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
    // int batch_idx = blockIdx.x;  
    // int seq_idx = blockIdx.y;   
    // int hidden_idx = threadIdx.x;  
    // int out_index = batch_idx * blockDim.x * gridDim.y *32 +  blockDim.x * seq_idx*32  +  hidden_idx*32  +  threadIdx.y *32 ;
    // inputs[out_index] = f_inputs[threadIdx.x];
  
    // dim3 grid(batch_size*seq_len*hidden_size/ 4096 + 1);
    // dim3 block(4096);
    // half2* p_out = (half2*)inputs;
    half2* p_in = (half2*)f_inputs; 
    int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
    int batch_idx = input_index / (seq_len*hidden_size);
    int seq_idx = input_index / hidden_size % seq_len;
    int hidden_idx = input_index % hidden_size;
    int out_index = batch_idx * seq_len * hidden_size  * padding  +  hidden_size * seq_idx * padding  +  hidden_idx * padding ;
    int out_index_ = out_index + padding/2;
    // inputs[out_index] = f_inputs[input_index];
    // half2 in_value = p_in[input_index];
    // half2 value;
    // value.x = __float2half(__half2float(in_value.x));
    // value.y = __float2half(__half2float(in_value.y));
    // value.y = __float2half(0.0);
    // p_out[out_index] = value;

    inputs[out_index] =  __float2half(__half2float(p_in[input_index].x));

    // half2 value1;
    // value1.x = __float2half(__half2float(in_value.y));
    // value1.y = __float2half(0.0);
    // p_out[out_index_] = value1;

    inputs[out_index_] = __float2half(__half2float(p_in[input_index].y));

}

// template <>
// __global__ void ker_input_pad_32_32<__half>(__half *f_inputs, __half *inputs, const int batch_size,
//                                  const int seq_len, const int hidden_size) {
//     // dim3 grid(batch_size,seq_len);
//     // dim3 block(hidden_size,cin);
    
//     // // inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
//     // f_inputs += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
//     // int batch_idx = blockIdx.x;  
//     // int seq_idx = blockIdx.y;   
//     // int hidden_idx = threadIdx.x;  
//     // int out_index = batch_idx * blockDim.x * gridDim.y*32  +  blockDim.x * seq_idx*32   +  hidden_idx *32  +  threadIdx.y *32 ;
//     // inputs[out_index] = f_inputs[threadIdx.x];

//     // dim3 grid(batch_size*seq_len*hidden_size/ 4096 + 1);
//     // dim3 block(4096);
//     int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
//     int batch_idx = input_index / (seq_len*hidden_size);
//     int seq_idx = input_index / hidden_size % seq_len;
//     int hidden_idx = input_index % hidden_size;

// }

template <typename T>
void ker_input_pad_ix_launcher(T *f_inputs, T *inputs, const int batch_size,
                                 const int seq_len, const int hidden_size, const int cin, cudaStream_t stream, const int padding) {

    // dim3 grid(176);
    // dim3 block(80*24);
    // dim3 grid(batch_size,seq_len);
    // dim3 block(hidden_size,cin);
    dim3 grid(batch_size*seq_len*hidden_size/ 4096 + 1);
    dim3 block(4096/2);
    // ker_input_pad<<<grid, block, 0, stream>>>(f_inputs, inputs,batch_size,seq_len,hidden_size,padding*2);
    ker_input_pad_half2<<<grid, block, 0, stream>>>(f_inputs, inputs,batch_size,seq_len,hidden_size,padding*2);
    cudaDeviceSynchronize();

}


template <typename T>
__global__ void ker_weight_pad_trans(T* inputs_, T* outputs_, const int padding) {
  // dim3 grid(cout, kernel);
  // dim3 block(kernel,cin);
  inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  int cout = blockIdx.x;  // 
  int kernel_h = blockIdx.y;   // 
  int kernel = threadIdx.x;  // 
  int out_index = cout * blockDim.x * gridDim.y *padding  +  blockDim.x * kernel_h  *padding +  kernel*padding  +  threadIdx.y *padding  ;
  outputs_[out_index] = inputs_[threadIdx.x];
}

// template <>
// __global__ void ker_weight_pad_trans<__half>(__half* inputs_, __half* outputs_) {
//   // dim3 grid(cout, kernel);
//   // dim3 block(kernel,cin);
//   inputs_ += blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
//   int cout = blockIdx.x;  // 
//   int kernel_h = blockIdx.y;   // 
//   int kernel = threadIdx.x;  // 
//   int out_index = cout * blockDim.x * gridDim.y *32 +  blockDim.x * kernel_h *32  +  kernel *32 +  threadIdx.y*32  ;
//   outputs_[out_index] = inputs_[threadIdx.x];
// }
template <typename T>
void  ker_weight_pad_trans_ix_launcher(T *input_, T *output_, 
                                        const int cout,const int  cin, const int kernel,
                                        cudaStream_t stream, const int padding){

    dim3 grid(cout, kernel);
    dim3 block(kernel,cin);
    ker_weight_pad_trans<<<grid, block, 0, stream>>>(input_, output_, padding);
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void ker_output_data(T *input_, T *output_){
  int input_index = blockDim.x * blockIdx.x + threadIdx.x ;
  output_[input_index] = input_[input_index];
}

template <typename T>
void  ker_output_data_ix_launcher(T *input_, T *output_, 
                                  const int batch_size, const int  h_out, 
                                  const int w_out, const int c_out,
                                   cudaStream_t stream){

    dim3 grid(batch_size*h_out*w_out*c_out / 4096 + 1);
    dim3 block(4096);
    ker_output_data<<<grid, block, 0, stream>>>(input_, output_);
    cudaDeviceSynchronize();

}

at::Tensor weight_padding(at::Tensor weight) {
  // weight shape : cout,,cin,ker-h,ker-w
    TORCH_CHECK(weight.dim() == 4);
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half);
    int c_out = weight.size(0);  // 256
    int c_in = weight.size(1);   // 1
    int kernel =  weight.size(2); // 3
    int padding = 32;
    at::Tensor output = weight.new_zeros({c_out,kernel,kernel,padding}); // 256,3,3,32
    // printf("use fp16\n");
    __half* input_ = (__half*) weight.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ker_weight_pad_trans_ix_launcher(input_, output_, c_out, c_in, kernel, stream, padding);
    return output;
  }

at::Tensor input_padding(at::Tensor input) {

  // input: fp16 [batch_size, cin,seq_len, hidden_size]
    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    int batch_size = input.size(0);  // 24
    int seq_len = input.size(2);   // 710
    int hidden_size = input.size(3); // 80
    int cin = input.size(1);  // 1
    int padding = 32;
    at::Tensor output = input.new_zeros({32,seq_len,hidden_size,padding}); // 32,710,80,32
    // printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ker_input_pad_ix_launcher(input_,output_,batch_size,seq_len,hidden_size,cin,stream,padding);
    return output;
  }

at::Tensor output_data(at::Tensor input){
    // output shape: 32,176,19,256 nhwc
    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);
    int batch_size = 24;  // 
    int h_out = input.size(1);   // 
    int w_out = input.size(2); // 
    int c_out = input.size(3); 

    at::Tensor output = input.new_zeros({batch_size,h_out,w_out,c_out}); // 24,176,19,256
    // printf("use fp16\n");
    __half* input_ = (__half*) input.data_ptr();
    __half* output_ = (__half*) output.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ker_output_data_ix_launcher(input_,output_,batch_size,h_out,w_out,c_out,stream);
    return output;

  }