#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudnn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "relu_help.h"

namespace relu {
namespace cuda {

template <typename T>
void ker_bias_relu_launcher(T *inputs, const T *bias, const int n, const int c,
                            const int h, const int w, cudaStream_t stream) {
  relu_opt::cuda::ker_bias_relu_ix_launcher(inputs, bias, n, c, h, w,
                                                stream);
}

template <typename T>
void ker_bias_relu_0123to0213_launcher(T *outputs, const T *inputs,
                                       const T *bias, const int n, const int c,
                                       const int h, const int w,
                                       cudaStream_t stream) {
  relu_opt::cuda::ker_bias_relu_0123to0213_ix_launcher(
      outputs, inputs, bias, n, c, h, w, stream);
}

void Conv2d(const int n, const int c_in, const int h_in, const int w_in,
            const int c_out, const int kernel_size, const int stride,
            __half *input, __half *weight, __half *work_space, __half *output,
            cudnnHandle_t cu_handle) {
  int h_out = (h_in - (kernel_size - 1) - 1) / stride + 1;
  int w_out = (w_in - (kernel_size - 1) - 1) / stride + 1;

  cudnnTensorDescriptor_t cu_x_desc;
  cudnnConvolutionDescriptor_t cu_conv_desc;
  cudnnFilterDescriptor_t cu_w_desc;
  cudnnTensorDescriptor_t cu_y_desc;

  cudnnCreateTensorDescriptor(&cu_x_desc);
  cudnnCreateTensorDescriptor(&cu_y_desc);
  cudnnCreateConvolutionDescriptor(&cu_conv_desc);
  cudnnCreateFilterDescriptor(&cu_w_desc);

  cudnnSetTensor4dDescriptor(cu_x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n,
                             c_in, h_in, w_in);
  cudnnSetTensor4dDescriptor(cu_y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n,
                             c_out, h_out, w_out);
  cudnnSetConvolution2dDescriptor(cu_conv_desc, 0, 0, stride, stride, 1, 1,
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnSetFilter4dDescriptor(cu_w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                             c_out, c_in, kernel_size, kernel_size);

  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
  cudnnConvolutionFwdAlgo_t algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);

  int cu_workspace_size = c_in * kernel_size * kernel_size * n * h_out * w_out;

  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
  float alpha1 = 1.f;
  float alpha2 = 0.f;

  cudnnConvolutionForward(cu_handle, &alpha1, cu_x_desc, input, cu_w_desc,
                          weight, cu_conv_desc, algo, work_space,
                          cu_workspace_size * sizeof(__half), &alpha2,
                          cu_y_desc, output);

  cudnnDestroyTensorDescriptor(cu_x_desc);
  cudnnDestroyTensorDescriptor(cu_y_desc);
  cudnnDestroyConvolutionDescriptor(cu_conv_desc);
  cudnnDestroyFilterDescriptor(cu_w_desc);
}

at::Tensor convrelu(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1) {
  //  inputs: nchw
  // weight1: c_out, c_in, kernel_size, kernel_size
  const int n = inputs.size(0);
  const int c_in = inputs.size(1);
  const int h_in = inputs.size(2);
  const int w_in = inputs.size(3);
  const int c_out = weight1.size(0);

  const int kernel_size = 3;
  const int stride = 2;
  int h_out = (h_in - (kernel_size - 1) - 1) / stride + 1;
  int w_out = (w_in - (kernel_size - 1) - 1) / stride + 1;

  int cu_workspace_size = c_out * kernel_size * kernel_size * n * h_out * w_out;
  at::Tensor work_space = inputs.new_empty({cu_workspace_size});
  at::Tensor out = inputs.new_empty({n, c_out, h_out, w_out});

  cudnnHandle_t cudnn_handle;
  cudnnCreate(&cudnn_handle);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  __half *inputs_ptr = (__half *)inputs.data_ptr();
  __half *weight1_ptr = (__half *)weight1.data_ptr();
  __half *work_space_ptr = (__half *)work_space.data_ptr();
  __half *out_ptr = (__half *)out.data_ptr();
  __half *bias1_ptr = (__half *)bias1.data_ptr();

  // conv
  Conv2d(n, c_in, h_in, w_in, c_out, kernel_size, stride, inputs_ptr,
         weight1_ptr, work_space_ptr, out_ptr, cudnn_handle);
  // bias relu
  ker_bias_relu_launcher(out_ptr, bias1_ptr, n, c_out, h_out, w_out, stream);
  // conv
  // const int kernel_size_2 = 5;
  // const int stride_2 = 3;
  // int h_out_2 = (h_out - (kernel_size_2 - 1) - 1) / stride_2 + 1;
  // int w_out_2 = (w_out - (kernel_size_2 - 1) - 1) / stride_2 + 1;
  // at::Tensor out2 = inputs.new_empty({n, c_out, h_out_2, w_out_2});

  // __half *out2_ptr = (__half *)out2.data_ptr();
  // __half *weight2_ptr = (__half *)weight2.data_ptr();

  // Conv2d(n, c_out, h_out, w_out, c_out, kernel_size_2,stride_2,out_ptr, weight2_ptr, work_space_ptr,
  //        out2_ptr, cudnn_handle);
  // __half *bias2_ptr = (__half *)bias2.data_ptr();
  // ker_bias_relu_0123to0213_launcher(out_ptr, out2_ptr, bias2_ptr, n, c_out,
  //                                   h_out_2, w_out_2, stream);
  cudnnDestroy(cudnn_handle);

  return out;
}

at::Tensor basic_conv(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1,
                      at::Tensor weight2, at::Tensor bias2) {
  //  inputs: nchw
  // weight1: c_out, c_in, kernel_size, kernel_size
  const int n = inputs.size(0);
  const int c_in = inputs.size(1);
  const int h_in = inputs.size(2);
  const int w_in = inputs.size(3);
  const int c_out = weight1.size(0);

  const int kernel_size = 3;
  const int stride = 2;
  int h_out = (h_in - (kernel_size - 1) - 1) / stride + 1;
  int w_out = (w_in - (kernel_size - 1) - 1) / stride + 1;

  int cu_workspace_size = c_out * kernel_size * kernel_size * n * h_out * w_out;
  at::Tensor work_space = inputs.new_empty({cu_workspace_size});
  at::Tensor out = inputs.new_empty({n, c_out, h_out, w_out});

  cudnnHandle_t cudnn_handle;
  cudnnCreate(&cudnn_handle);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  __half *inputs_ptr = (__half *)inputs.data_ptr();
  __half *weight1_ptr = (__half *)weight1.data_ptr();
  __half *work_space_ptr = (__half *)work_space.data_ptr();
  __half *out_ptr = (__half *)out.data_ptr();
  __half *bias1_ptr = (__half *)bias1.data_ptr();

  // conv
  Conv2d(n, c_in, h_in, w_in, c_out, kernel_size, stride, inputs_ptr,
         weight1_ptr, work_space_ptr, out_ptr, cudnn_handle);
  // bias relu
  ker_bias_relu_launcher(out_ptr, bias1_ptr, n, c_out, h_out, w_out, stream);
  // conv
  const int kernel_size_2 = 5;
  const int stride_2 = 3;
  int h_out_2 = (h_out - (kernel_size_2 - 1) - 1) / stride_2 + 1;
  int w_out_2 = (w_out - (kernel_size_2 - 1) - 1) / stride_2 + 1;
  at::Tensor out2 = inputs.new_empty({n, c_out, h_out_2, w_out_2});

  __half *out2_ptr = (__half *)out2.data_ptr();
  __half *weight2_ptr = (__half *)weight2.data_ptr();

  Conv2d(n, c_out, h_out, w_out, c_out, kernel_size_2,stride_2,out_ptr, weight2_ptr, work_space_ptr,
         out2_ptr, cudnn_handle);
  __half *bias2_ptr = (__half *)bias2.data_ptr();
  ker_bias_relu_0123to0213_launcher(out_ptr, out2_ptr, bias2_ptr, n, c_out,
                                    h_out_2, w_out_2, stream);
  cudnnDestroy(cudnn_handle);

  return out;
}

}  // namespace cuda
}  // namespace relu