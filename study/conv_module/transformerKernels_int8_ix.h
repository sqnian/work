#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <cublasLt.h>
#include <cuda_runtime.h>
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
#define BLOCKSIZE 1024
namespace lightseq_opt {
namespace cuda {

/*transpose native ;warning :need optimize*/
/*
transpose 0123 0213
nhcw ->nchw
*/

__forceinline__ __device__ int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  // int32_t i8 = floorf(i8_f);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

template <typename T>
__global__ void ker_transpose_0123to0213(T* outputs, const T* inputs,
                                         const int n,
                                         const int c, const int h,
                                         const int w) {
  
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
    int block_ele_num = batch_idx * c * h * w + blockIdx.x * h * w;
    T* p_outputs = (T*)outputs + block_ele_num;
    int ele_index = threadIdx.x;
    while (ele_index < h * w) {
      
      int h_index = ele_index / w;
      int w_index = ele_index % w;
      T value =inputs[batch_idx * c * h * w + h_index * c * w + blockIdx.x * w + w_index] ;     
      p_outputs[ele_index] = value;
      ele_index += blockDim.x;
    }
  }
}
template <typename T>
void ker_transpose_0123to0213_ix_launcher(T* outputs, const T* inputs,
                                          const int n,
                                          const int c, const int h, const int w,
                                          cudaStream_t stream) {
  dim3 grid(c);
  dim3 block(BLOCKSIZE);
  ker_transpose_0123to0213<<<grid, block, 0, stream>>>(outputs, inputs, n,
                                                       c, h, w);
}

/*bias glu
inputs:n,c,h,w
outputs:n,c/2，h,w*/
//glu = inputs(:,:c/2,:,:)点乘sigmoid(inputs(:,c/2:,:,:))
__device__ __forceinline__ float sigmoid(float x){
    return(1.f/(1.f+__expf(0.f -x)));  
}

// template <typename T>
__global__ void ker_bias_glu(int8_t* outputs,const int8_t* inputs, const float amax_in, const float amax_out, const int n,
                              const int c, const int h, const int w) {
  // T block_bias1 =(bias==nullptr)?0.0f: bias[blockIdx.x];
  // T block_bias2 =(bias==nullptr)?0.0f:bias[blockIdx.x+c];
  const float descale = amax_in / 127;
  const float  scale = 127 /amax_out;
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) 
  {
    int in_block_ele_num = batch_idx * 2*c * h * w + blockIdx.x * h * w;    
    int in_block_ele_num_sig = batch_idx * 2* c * h * w + (blockIdx.x+c) * h * w;
    int out_block_ele_num = batch_idx *c * h * w + blockIdx.x * h * w;

    

    int ele_index = threadIdx.x;

    while (ele_index < h * w) {
      // 反量化 ，int8 -> float
      float fout_inputs1 = float(inputs[in_block_ele_num + ele_index]) * descale;
      float fout_inputs2 = float(inputs[in_block_ele_num_sig + ele_index]) * descale;
      //bias glu
      float value = (fout_inputs1 )*sigmoid(fout_inputs2);
      
      // 量化，float-> int8
      int8_t int_value  = float2int8(value, scale);
      outputs[out_block_ele_num+ele_index] = int_value;

      ele_index += blockDim.x;
    }
  }
}
// template <>
// __global__ void ker_bias_glu<__half>(__half* outputs, const __half* inputs, const int n,
//                               const int c, const int h, const int w) {
          
  // __half block_bias1 =(bias==nullptr)?__float2half(0.0f):(bias[blockIdx.x]);
  // __half block_bias2 =(bias==nullptr)?__float2half(0.0f):(bias[blockIdx.x+c]);
//   for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
//     int in_block_ele_num = batch_idx * 2*c * h * w + blockIdx.x * h * w;    
//     int in_block_ele_num_sig = batch_idx * 2* c * h * w + (blockIdx.x+c) * h * w;
//     int out_block_ele_num = batch_idx *c * h * w + blockIdx.x * h * w;
//     __half* p_inputs1 = (__half*)inputs + in_block_ele_num;
//     __half* p_inputs2 = (__half*)inputs + in_block_ele_num_sig;
//     int ele_index = threadIdx.x;
//     while (ele_index < h * w) {
//       //bias glu
//       __half value1 = p_inputs1[ele_index] ;
//       __half value2 = p_inputs2[ele_index] ;
//       __half value = __hmul(value1,__float2half(sigmoid(__half2float(value2))));
//       // float value = (__half2float(p_inputs1[ele_index]) + block_bias1)+(__half2float(p_inputs2[ele_index]) + block_bias2);;
//       // float value = sigmoid(__half2float(p_inputs2[ele_index]) + block_bias2);
      
//       outputs[out_block_ele_num+ele_index] =(value) ;
//       ele_index += blockDim.x;
//     }
//   }
//   printf(" half  run here\n");
// }
//c :output channel 
// template <typename T>
void ker_bias_glu_ix_launcher(int8_t* outputs,const int8_t* inputs,const float amax_in, const float amax_out, const int n,
                               const int c, const int h, const int w,
                               cudaStream_t stream) {
  dim3 grid(c);
  dim3 block(BLOCKSIZE);
  ker_bias_glu<<<grid, block, 0, stream>>>(outputs,inputs,amax_in,amax_out, n, c, h, w);
  // printf(".h run here\n");
}
/*
bias_relu
inputs: n, c, h, w
bias: c

relu(inputs+c)
*/
template <typename T>
__global__ void ker_bias_relu(T* inputs, const T* bias, const int n,
                              const int c, const int h, const int w) {
  T block_bias = bias[blockIdx.x];
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
    int block_ele_num = batch_idx * c * h * w + blockIdx.x * h * w;
    T* p_inputs = inputs + block_ele_num;
    int ele_index = threadIdx.x;
    while (ele_index < h * w) {
      T value = p_inputs[ele_index] + block_bias;
      // relu
      value = value > 0.f ? value : 0.f;
      p_inputs[ele_index] = value;
      ele_index += blockDim.x;
    }
  }
}

template <>
__global__ void ker_bias_relu<__half>(__half* inputs, const __half* bias,
                                      const int n, const int c, const int h,
                                      const int w) {
  float block_bias = __half2float(bias[blockIdx.x]);
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
    int block_ele_num = batch_idx * c * h * w + blockIdx.x * h * w;
    __half* p_inputs = inputs + block_ele_num;
    int ele_index = threadIdx.x;
    while (ele_index < h * w) {
      float value = __half2float(p_inputs[ele_index]) + block_bias;
      // relu
      value = value > 0.f ? value : 0.f;
      p_inputs[ele_index] = __float2half(value);
      ele_index += blockDim.x;
    }
  }
}

template <typename T>
void ker_bias_relu_ix_launcher(T* inputs, const T* bias, const int n,
                               const int c, const int h, const int w,
                               cudaStream_t stream) {
  dim3 grid(c);
  dim3 block(BLOCKSIZE);
  ker_bias_relu<<<grid, block, 0, stream>>>(inputs, bias, n, c, h, w);
}

/*
bias_relu_0123to0213
inputs: n, c, h, w
bias: c

relu(inputs+c)-> transpose(1,2) n,h,c,w
*/
template <typename T>
__global__ void ker_bias_relu_0123to0213(T* outputs, const T* inputs,
                                         const T* bias, const int n,
                                         const int c, const int h,
                                         const int w) {
  T block_bias = bias[blockIdx.x];
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
    int block_ele_num = batch_idx * c * h * w + blockIdx.x * h * w;
    T* p_inputs = inputs + block_ele_num;
    int ele_index = threadIdx.x;
    while (ele_index < h * w) {
      T value = p_inputs[ele_index] + block_bias;
      // relu
      value = value > 0.f ? value : 0.f;

      int h_index = ele_index / w;
      int w_index = ele_index % w;

      outputs[batch_idx * c * h * w + h_index * c * w + blockIdx.x * w +
              w_index] = value;
      ele_index += blockDim.x;
    }
  }
}

template <>
__global__ void ker_bias_relu_0123to0213<__half>(__half* outputs,
                                                 const __half* inputs,
                                                 const __half* bias,
                                                 const int n, const int c,
                                                 const int h, const int w) {
  float block_bias = __half2float(bias[blockIdx.x]);
  for (int batch_idx = 0; batch_idx < n; ++batch_idx) {
    int block_ele_num = batch_idx * c * h * w + blockIdx.x * h * w;
    const __half* p_inputs = inputs + block_ele_num;
    int ele_index = threadIdx.x;
    while (ele_index < h * w) {
      float value = __half2float(p_inputs[ele_index]) + block_bias;
      // relu
      value = value > 0.f ? value : 0.f;

      int h_index = ele_index / w;

      int w_index = ele_index % w;

      outputs[batch_idx * c * h * w + h_index * c * w + blockIdx.x * w +
              w_index] = __float2half(value);

      ele_index += blockDim.x;
    }
  }
}

template <typename T>
void ker_bias_relu_0123to0213_ix_launcher(T* outputs, const T* inputs,
                                          const T* bias, const int n,
                                          const int c, const int h, const int w,
                                          cudaStream_t stream) {
  dim3 grid(c);
  dim3 block(BLOCKSIZE);
  ker_bias_relu_0123to0213<<<grid, block, 0, stream>>>(outputs, inputs, bias, n,
                                                       c, h, w);
}
//bias bn1d silu
const float epsilon = 0.000000000001;
const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const float CUDA_FLOAT_INF_NEG = -100000000.f;
const float CUDA_FLOAT_INF_POS = 100000000.f;

/*

input: batch_size,channels, h,w
bias: channels
scale: channels
residual_bias: channels

input = norm2(input+residual_bias)
*/
template <typename T>
__global__ void ker_enc_bias_bn_silu(
    T *input, const T *scale, const T *bias,const T *running_mean, const T *running_var, const T *residual_bias,
    int batch_size,int channels,int h,int w) {

  int feat_dim = channels*h *w;
  int feat_dim_hw = h *w;


#pragma unroll
for(int batchid =0;batchid<batch_size;batchid++){
  int batch_start =batchid*feat_dim+  blockIdx.x *feat_dim_hw;
  
    int feat_dim_hw_idx =threadIdx.x;
    int element_index = batch_start+feat_dim_hw_idx;
    
      float norm_value=0.f;
      if(residual_bias==nullptr)
      {
         norm_value = (input[element_index] - running_mean[blockIdx.x]) * rsqrtf(running_var[blockIdx.x] + epsilon) *
                              (float)scale[blockIdx.x] + (float)bias[blockIdx.x];          
      }
      else
      {
        norm_value  = (input[element_index]+residual_bias[blockIdx.x] - running_mean[blockIdx.x]) * rsqrtf(running_var[blockIdx.x] + epsilon) *
                             (float)scale[blockIdx.x] + (float)bias[blockIdx.x];       
        // if(batchid ==0 && blockIdx.x==0 && threadIdx.x==0 && element_index==0)
        // printf("cuda checking input[element_index] %f residual_bias[blockIdx.x]  %f running_mean[blockIdx.x] %f running_var[blockIdx.x] %f\
        // bias[blockIdx.x] %f norm_value %f\n",input[element_index],residual_bias[blockIdx.x],running_mean[blockIdx.x],running_var[blockIdx.x],bias[blockIdx.x],norm_value);
         
      }
      float fout = norm_value/(__expf(-norm_value)+1.0f);
      input[element_index] = fout;
      
  
}
}
/*

input: batch_size,channels, h,w
bias: channels
scale: channels
residual_bias: channels

input = norm2(input+residual_bias)
*/
template <>
__global__ void ker_enc_bias_bn_silu<__half>(
    __half *input, const __half *scale, const __half *bias,const __half *running_mean, const __half *running_var, const __half *residual_bias,
    int batch_size,int channels,int h,int w) {
 
  int feat_dim = channels*h *w;
  int feat_dim_hw = h *w; 
#pragma unroll
for(int batchid =0;batchid<batch_size;batchid++){
  int batch_start =batchid*feat_dim+  blockIdx.x *feat_dim_hw;
  
    int feat_dim_hw_idx =threadIdx.x ;
    int element_index = batch_start+feat_dim_hw_idx;
   
      float norm_value =0.f;
      if(residual_bias==nullptr)
      {
         
       norm_value= (__half2float(input[element_index]) - __half2float(running_mean[blockIdx.x])) * rsqrtf(__half2float(running_var[blockIdx.x]) + epsilon) *
                            __half2float(scale[blockIdx.x]) +                        __half2float(bias[blockIdx.x]);  
      }
      else
      {
        norm_value= (__half2float(input[element_index]) + __half2float(residual_bias[blockIdx.x])  -__half2float(running_mean[blockIdx.x]) ) * rsqrtf(__half2float(running_var[blockIdx.x]) + epsilon) *
                            __half2float(scale[blockIdx.x]) +                        __half2float(bias[blockIdx.x]);  
      }
      float fout = norm_value/(__expf(-norm_value)+1.0f);          
      input[element_index] = __float2half(fout);
      
      
    
    
  }
}




template <typename T>
void ker_enc_bias_bn_silu_ix_launcher(
    T *input, const T *scale, const T *bias, const T *running_mean, const T *running_var,const T *residual_bias,int batch_size,
    int channels, int h,int w,cudaStream_t stream) {
  int feat_dim_hw = h* w;
  if (feat_dim_hw > 4096) {
    throw std::runtime_error("hidden_size should <= 4096");
  }
  
  dim3 gridSize(channels);
  dim3 blockSize(feat_dim_hw);


  ker_enc_bias_bn_silu<T><<<gridSize, blockSize, 0, stream>>>(
              input, scale, bias,running_mean,running_var, residual_bias,batch_size,channels,h,w);
   
  
}

template <>
void ker_enc_bias_bn_silu_ix_launcher<__half>(
    __half *input, const __half *scale, const __half *bias,const __half *running_mean, const __half *running_var,
    const __half *residual_bias,int batch_size,
    int channels, int h,int w,cudaStream_t stream) {
 
  int feat_dim_hw = h* w;
  if (feat_dim_hw > 4096) {
    throw std::runtime_error("hidden_size should <= 4096");
  }
  
  dim3 gridSize(channels);
  dim3 blockSize(feat_dim_hw);


  ker_enc_bias_bn_silu<__half><<<gridSize, blockSize, 0, stream>>>(
              input, scale, bias,running_mean,running_var, residual_bias,batch_size,channels,h,w);
}
template <typename T>
void ker_enc_bias_bn_silu_ix_launcher(
    T *input, const T *scale, const T *bias, const T *running_mean, const T *running_var,const T *residual_bias,
    int batch_size,int channels, int h,int w,cudaStream_t stream);
template <>
void ker_enc_bias_bn_silu_ix_launcher<__half>(
    __half *input, const __half *scale, const __half *bias,const __half *running_mean, const __half *running_var, const __half *residual_bias,
    int batch_size,int channels, int h,int w,cudaStream_t stream);
}  // namespace cuda
}  // namespace lightseq_opt