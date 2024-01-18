#include <torch/extension.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <time.h>
#include "ixinfer.h"
// #include "transformerKernels_int8_ix.h"

#ifndef CHECK
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
#endif


at ::Tensor conv_qd(
    at::Tensor inputs,  // NHWC
    at::Tensor input,
    at::Tensor weight,
    at::Tensor weigh,
    at::Tensor alpha,
    at::Tensor bias,
    const int  act_num,
    const float amax_out
)
{
    
    // input:nhwc
    const int batchsize_ = inputs.size(0);
    const int chls_in_ = inputs.size(3);
    const int height_in_ = inputs.size(1);
    const int width_in_ = inputs.size(2);  //1

    // weight: out_channels, kernel_size, kernel_size, in_channels
    const int chls_out_ = weight.size(0);

    // conv 
    const int pad_h_ = 0;  // 7 不支持
    const int pad_w_ = 0;
    const int stride_h_ = 1;
    const int stride_w_ = 1;
    const int kernel_h_ =  weight.size(1);
    const int kernel_w_ =  weight.size(2);


    const int group_count_ = 1;
    
    
    const int height_out = (height_in_ + 2 * pad_h_ - (kernel_h_ -1) - 1) / stride_h_ + 1;
    const int width_out =  (width_in_ + 2 * pad_w_ - (kernel_w_ -1) - 1) / stride_w_ + 1;

    cuinferHandle_t cu_handle;
    cuinferTensorDescriptor_t cu_x_desc;
    cuinferConvolutionDescriptor_t cu_conv_desc;
    cuinferFilterDescriptor_t cu_w_desc;
    cuinferTensorDescriptor_t cu_y_desc;
    cuinferTensorDescriptor_t cu_bias_desc;
    cuinferActivationDescriptor_t cu_act_desc;


    cuinferConvolutionFwdAlgo_t cu_algo;
    cu_algo = static_cast<cuinferConvolutionFwdAlgo_t>(1);  //DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM shape无限制
    /** create handle and descriptor */
    (cuinferCreate(&cu_handle));
    (cuinferCreateTensorDescriptor(&cu_x_desc));  // input data 
    (cuinferCreateConvolutionDescriptor(&cu_conv_desc));
    (cuinferCreateFilterDescriptor(&cu_w_desc));
    (cuinferCreateTensorDescriptor(&cu_y_desc));
    (cuinferCreateTensorDescriptor(&cu_bias_desc));
    (cuinferCreateActivationDescriptor(&cu_act_desc));

    (cuinferSetTensor4dDescriptor(
        cu_x_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_, chls_in_, height_in_, width_in_));

    (cuinferSetTensor4dDescriptor(
        cu_y_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_,chls_out_, height_out, width_out));

    (cuinferSetConvolution2dDescriptor(
        cu_conv_desc,pad_h_, pad_w_,stride_h_, stride_w_, 1, 1, CUINFER_CROSS_CORRELATION, CUINFER_DATA_INT32));

    (cuinferSetFilter4dDescriptor(
        cu_w_desc, CUINFER_DATA_INT8, CUINFER_TENSOR_NHWC, chls_out_, chls_in_ / group_count_, kernel_h_, kernel_w_));

    (cuinferSetTensor4dDescriptor(
        cu_bias_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_FLOAT, 1, chls_out_, 1, 1));
    
    // activation type
    /*
    typedef enum {
        CUINFER_ACTIVATION_SIGMOID = 0,
        CUINFER_ACTIVATION_RELU = 1,
        CUINFER_ACTIVATION_TANH = 2,
        CUINFER_ACTIVATION_CLIPPED_RELU = 3,
        CUINFER_ACTIVATION_ELU = 4,
        CUINFER_ACTIVATION_IDENTITY = 5,
        CUINFER_ACTIVATION_LEAKY_RELU = 6,
        CUINFER_ACTIVATION_SILU = 7,
        CUINFER_ACTIVATION_HARD_SWISH = 8,
        CUINFER_ACTIVATION_HARD_SIGMOID = 9,
        } cuinferActivationMode_t;
    */
    if (act_num == 0){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_SIGMOID, CUINFER_NOT_PROPAGATE_NAN, 0));
        // printf("here==========================\n");

    }else if(act_num == 1){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
        // printf("here==========================\n");
    }else if(act_num == 2){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_TANH, CUINFER_NOT_PROPAGATE_NAN, 0));
    }else if(act_num == 3){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_CLIPPED_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
    }else if(act_num == 4){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_ELU, CUINFER_NOT_PROPAGATE_NAN, 0));
    }else if(act_num == 5){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_IDENTITY, CUINFER_NOT_PROPAGATE_NAN, 0)); // no activate
    }else if(act_num == 6){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_LEAKY_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
    }else if(act_num == 7){
        (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_SILU, CUINFER_NOT_PROPAGATE_NAN, 0));
        // printf("==================\n");
    }
    // else if(act_num == 8){
    //     (cuinferSetActivationDescriptor(
    //     cu_act_desc, CUINFER_ACTIVATION_HARD_SWISH, CUINFER_NOT_PROPAGATE_NAN, 0));
    // }else if(act_num == 9){
    //     (cuinferSetActivationDescriptor(
    //     cu_act_desc, CUINFER_ACTIVATION_HARD_SIGMOID, CUINFER_NOT_PROPAGATE_NAN, 0));
    // }
    

     // bias format 
    float *cu_d_bias = ( float *)bias.data_ptr() ; //  nullptr; // bias nullptr
    float *cu_alpha = (float *)alpha.data_ptr();  // perchannel alpha
    int *cu_d_workspace;
    const float alpha_ = 0.01;  // alpha
    float beta_;
    if(act_num == 0 || act_num == 7){
        beta_ = 127 / amax_out; 
    }else{
        beta_ = 0.0; 
        // printf("c++ run in here\n");
    }
    float gamma = 0.1;
    bool use_pechannel = true;  // false


    (cuinferSetConvolutionGroupCount(cu_conv_desc, group_count_));
    size_t cuworkspace_size = 0;
    auto status = static_cast<int>(cuinferGetConvolutionForwardWorkspaceSize(
      cu_handle, cu_x_desc, cu_w_desc, cu_conv_desc, cu_y_desc, cu_algo,
      &cuworkspace_size));
    cudaMalloc(reinterpret_cast<void **>(&cu_d_workspace), cuworkspace_size);
    // printf("cuworkspace_size %d\n",cuworkspace_size);
    // printf("cu_d_workspace %d\n",cu_d_workspace);
    
    at::Tensor cu_d_y = inputs.new_empty({batchsize_, height_out, width_out, chls_out_});
    
    int8_t *wei_ptr = (int8_t *)weigh.data_ptr();
    int8_t *inp_ptr = (int8_t *)input.data_ptr();
    int8_t *res_ptr = (int8_t *)cu_d_y.data_ptr();
    // printf("cu_d_y size %d\n",batchsize_ * height_out * width_out * chls_out_  * sizeof(int));

    // int8_t *h_input =  (int8_t *) malloc(input.size(0) * sizeof(int8_t));
    // CHECK(cudaMemcpy(h_input, input.data_ptr(), input.size(0) * sizeof(int8_t), cudaMemcpyDeviceToHost));
    // input data information
    // for(int i = 0; i < 100; i++){
    //     printf("h_inputs: %d \n",h_input[i]);
    // }

    cuinferQDConvolutionForward(
      cu_handle,  &alpha_, cu_alpha,&beta_,&gamma,
      cu_x_desc,inp_ptr,cu_w_desc, wei_ptr ,cu_conv_desc,cu_algo, 
      cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, res_ptr, 
      cu_bias_desc, cu_d_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, res_ptr);

    (cudaDeviceSynchronize());
    (cuinferDestroyTensorDescriptor(cu_x_desc));
    (cuinferDestroyConvolutionDescriptor(cu_conv_desc));
    (cuinferDestroyFilterDescriptor(cu_w_desc));
    (cuinferDestroyTensorDescriptor(cu_y_desc));
    (cuinferDestroy(cu_handle));
    cudaFree(cu_d_workspace);

    // printf("========= print output data ======\n");
    // int8_t *cu_h_y = (int8_t *) malloc(input.size(0) * sizeof(int8_t));
    // int out_size = cu_d_y.size(0) * cu_d_y.size(1) * cu_d_y.size(2) * cu_d_y.size(3);

    // CHECK(cudaMemcpy(cu_h_y, cu_d_y.data_ptr(), out_size * sizeof(int8_t), cudaMemcpyDeviceToHost));

    // for (int i=0; i<100;i++){
    //     printf("cu_h_y :%d\n",cu_h_y[i]);
    // }

    return cu_d_y;
}

namespace lightseq {
namespace cuda {
// at::Tensor basic_conv(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1,
//                       at::Tensor weight2, at::Tensor bias2);

    at::Tensor convGlu(at::Tensor outputs, at::Tensor inputs,const float amax_in,const float amax_out);

}
}  // namespace lightseq

at::Tensor bias_glu_ix_launcher(
    at::Tensor outputs,
    at::Tensor inputs,
    const float amax_in,
    const float amax_out)
{
    
    return lightseq::cuda::convGlu(outputs, inputs,amax_in,amax_out);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_conv", &conv_qd,"test conv1d quant cpp");
    m.def("test_glu", &bias_glu_ix_launcher,"test conv glu quant cpp");


}