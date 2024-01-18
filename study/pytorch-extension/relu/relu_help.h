#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>

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

namespace relu_opt {
namespace cuda {

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
  dim3 block(1024);
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
  dim3 block(1024);
  ker_bias_relu_0123to0213<<<grid, block, 0, stream>>>(outputs, inputs, bias, n,
                                                       c, h, w);
}

}  // namespace cuda
}  // namespace relu_opt