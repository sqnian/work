#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "ixinfer.h"

#undef CUDA_CHECK
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#undef CUINFER_CHECK
#define CUINFER_CHECK(func)                                                              \
    do {                                                                                 \
        cuinferStatus_t status = (func);                                                 \
        if (status != CUINFER_STATUS_SUCCESS) {                                          \
            std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                      << cuinferGetErrorString(status) << std::endl;                     \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

at::Tensor one_test(at::Tensor input,at::Tensor input_int8, at::Tensor mean,
                             at::Tensor istd, const float amax,bool fp16);

at::Tensor test(at::Tensor input,at::Tensor input_int8,at::Tensor mean,
                             at::Tensor istd, const float amax,bool fp16) {
    return one_test(input,input_int8, mean, istd, amax, fp16);
}



// 构建一个con2d函数
// 两次调用这个函数，得到最终结果
void  Conv2d_int8(  
    const int batchsize_,
    const int chls_in_,
    const int height_in_,
    const int width_in_,
    const int chls_out_,
    const int pad_h_,
    const int pad_w_ ,
    const int stride_h_,
    const int stride_w_,
    const int kernel_h_,
    const int kernel_w_ ,
    const int height_out,
    const int width_out,
    float *cu_bias,   // bias nullptr
    float *cu_alpha, // perchannel alpha
    int8_t *wei_ptr,
    int8_t *inp_ptr,
    int8_t *res_ptr  
)
{
    cuinferHandle_t cu_handle;
    cuinferTensorDescriptor_t cu_x_desc;
    cuinferConvolutionDescriptor_t cu_conv_desc;
    cuinferFilterDescriptor_t cu_w_desc;
    cuinferTensorDescriptor_t cu_y_desc;
    cuinferTensorDescriptor_t cu_bias_desc;
    cuinferActivationDescriptor_t cu_act_desc;

    const int group_count_ = 1;
    cuinferConvolutionFwdAlgo_t cu_algo;
    cu_algo = static_cast<cuinferConvolutionFwdAlgo_t>(1); //DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM shape无限制
    /** create handle and descriptor */
    CUINFER_CHECK(cuinferCreate(&cu_handle));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&cu_x_desc));  // input data 
    CUINFER_CHECK(cuinferCreateConvolutionDescriptor(&cu_conv_desc));
    CUINFER_CHECK(cuinferCreateFilterDescriptor(&cu_w_desc));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&cu_y_desc));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&cu_bias_desc));
    CUINFER_CHECK(cuinferCreateActivationDescriptor(&cu_act_desc));

    CUINFER_CHECK(cuinferSetTensor4dDescriptor(
        cu_x_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_, chls_in_, height_in_, width_in_));

    CUINFER_CHECK(cuinferSetTensor4dDescriptor(
        cu_y_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_,chls_out_, height_out, width_out));

    CUINFER_CHECK(cuinferSetConvolution2dDescriptor(
        cu_conv_desc,pad_h_, pad_w_,stride_h_, stride_w_, 1, 1, CUINFER_CROSS_CORRELATION, CUINFER_DATA_INT32));

    CUINFER_CHECK(cuinferSetFilter4dDescriptor(
        cu_w_desc, CUINFER_DATA_INT8, CUINFER_TENSOR_NHWC, chls_out_, chls_in_ / group_count_, kernel_h_, kernel_w_));

    CUINFER_CHECK(cuinferSetTensor4dDescriptor(
        cu_bias_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_FLOAT, 1, chls_out_, 1, 1));

    CUINFER_CHECK(cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
        // cu_act_desc, CUINFER_ACTIVATION_IDENTITY, CUINFER_NOT_PROPAGATE_NAN, 0));//no activate

     // bias format 
    // float *cu_bias = ( float *)bias.data_ptr() ;   // bias nullptr
    // float *cu_alpha = (float *)alpha.data_ptr();  // perchannel alpha
    int *cu_d_workspace;
    const float alpha_ = 0.01;  // alpha
    const int beta_ = 0;
    float gamma = 0.1;
    bool use_pechannel = true;  // false

    
    CUINFER_CHECK(cuinferSetConvolutionGroupCount(cu_conv_desc, group_count_));
    size_t cuworkspace_size = 0;
    auto status = static_cast<int>(cuinferGetConvolutionForwardWorkspaceSize(
      cu_handle, cu_x_desc, cu_w_desc, cu_conv_desc, cu_y_desc, cu_algo,
      &cuworkspace_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cu_d_workspace), cuworkspace_size));
 
    CUINFER_CHECK(cuinferQDConvolutionForward(
      cu_handle,  &alpha_, cu_alpha,&beta_,&gamma,
      cu_x_desc,inp_ptr,cu_w_desc, wei_ptr ,cu_conv_desc,cu_algo, 
      cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, res_ptr, 
      cu_bias_desc, cu_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, res_ptr));

    cudaDeviceSynchronize();
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(cu_x_desc));
    CUINFER_CHECK(cuinferDestroyConvolutionDescriptor(cu_conv_desc));
    CUINFER_CHECK(cuinferDestroyFilterDescriptor(cu_w_desc));
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(cu_y_desc));
    CUINFER_CHECK(cuinferDestroy(cu_handle));
    CUDA_CHECK(cudaFree(cu_d_workspace));


    // return cu_d_y;
}

// 两个 conv2d + relu 的函数
at::Tensor int8_conv2d(
    at::Tensor inputs,  // inputs  shape :nhwc 
    at::Tensor weight1,  // 
    at::Tensor alpha1,
    at::Tensor bias1,
    at::Tensor weight2,
    at::Tensor alpha2,
    at::Tensor bias2
)
{   
    // 检查dim，cuda，
    TORCH_CHECK(inputs.dim()==4);
    TORCH_CHECK(inputs.is_contiguous());
    TORCH_CHECK(inputs.is_cuda());

    TORCH_CHECK(weight1.dim()==4);
    TORCH_CHECK(weight1.is_contiguous());
    TORCH_CHECK(weight1.is_cuda());

    TORCH_CHECK(alpha1.dim()==1);
    TORCH_CHECK(alpha1.is_cuda());

    TORCH_CHECK(bias1.dim()==1);
    TORCH_CHECK(bias1.is_cuda());

    TORCH_CHECK(weight2.dim()==4);
    TORCH_CHECK(weight2.is_contiguous());
    TORCH_CHECK(weight2.is_cuda());

    TORCH_CHECK(alpha2.dim()==1);
    TORCH_CHECK(alpha2.is_cuda());

    TORCH_CHECK(bias2.dim()==1);
    TORCH_CHECK(bias2.is_cuda());


    // input:nhwc
    const int batchsize_ = inputs.size(0);
    const int chls_in_ = inputs.size(3);
    const int height_in_ = inputs.size(1);
    const int width_in_ = inputs.size(2); 

    // weight: out_channels, kernel_size, kernel_size, in_channels
    const int chls_out_ = weight1.size(0);

    // conv 
    const int pad_h_ = 0;
    const int pad_w_ = 0;
    const int stride_h_ = 2;
    const int stride_w_ = 2;
    const int kernel_h_ =  weight1.size(1);
    const int kernel_w_ =  weight1.size(2);
    
    const int height_out = (height_in_ + 2 * pad_h_ - (kernel_h_ -1) - 1) / stride_h_ + 1;
    const int width_out =  (width_in_ + 2 * pad_w_ - (kernel_w_ -1) - 1) / stride_w_ + 1;

    at::Tensor cu_out = inputs.new_empty({batchsize_, height_out, width_out, chls_out_});


    float *cu_bias1 = ( float *)bias1.data_ptr() ;   // bias nullptr
    float *cu_alpha1 = (float *)alpha1.data_ptr();  // perchannel alpha
    int8_t *wei_ptr = (int8_t *)weight1.data_ptr();
    int8_t *inp_ptr = (int8_t *)inputs.data_ptr();
    int8_t *res_ptr = (int8_t *)cu_out.data_ptr();
    
    Conv2d_int8(batchsize_, chls_in_, height_in_, width_in_, chls_out_,
                pad_h_,pad_w_,stride_h_,stride_w_,kernel_h_,kernel_w_,height_out,width_out,
                cu_bias1,cu_alpha1,wei_ptr,inp_ptr,res_ptr);
    
    // return  cu_out;

    // conv
    const int chls_out_2 = weight2.size(0);
    const int kernel_h_2 =  weight2.size(1);
    const int kernel_w_2 =  weight2.size(2);
    const int pad_h_2 = 0;
    const int pad_w_2 = 0;
    const int stride_h_2 = 2;
    const int stride_w_2= 2;
    const int h_out2 = (height_out + 2 * pad_h_2 - (kernel_h_2 -1) - 1) / stride_h_2 + 1;
    const int w_out2 =  (width_out + 2 * pad_w_2 - (kernel_w_2 -1) - 1) / stride_w_2 + 1;

    at::Tensor cu_out2 = cu_out.new_empty({batchsize_, h_out2, w_out2, chls_out_2});

    float *cu_bias2 = ( float *)bias2.data_ptr() ;   // bias nullptr
    float *cu_alpha2 = (float *)alpha2.data_ptr();  // perchannel alpha
    int8_t *wei_ptr2 = (int8_t *)weight2.data_ptr();
    int8_t *inp_ptr2 = (int8_t *)cu_out.data_ptr();
    int8_t *res_ptr2 = (int8_t *)cu_out2.data_ptr();

    Conv2d_int8(batchsize_, chls_out_, height_out, width_out, chls_out_2,
                pad_h_2,pad_w_2,stride_h_2,stride_w_2,kernel_h_2,kernel_w_2,h_out2,w_out2,
                cu_bias2,cu_alpha2,wei_ptr2,inp_ptr2,res_ptr2);

    return cu_out2;  

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_cmvn", &test, "use  cmvn op");
    m.def("test_func", &int8_conv2d,"gpu test int8_conv2d cpp");
}