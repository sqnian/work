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

at ::Tensor conv2d_relu(
    at::Tensor inputs,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor weigh,
    at::Tensor alpha,
    at::Tensor bias   
)
{
    
    // input:nhwc
    const int batchsize_ = inputs.size(0);
    const int chls_in_ = inputs.size(3);
    const int height_in_ = inputs.size(1);
    const int width_in_ = inputs.size(2); 

    // weight: out_channels, kernel_size, kernel_size, in_channels
    const int chls_out_ = weight.size(0);

    // conv 
    const int pad_h_ = 0;
    const int pad_w_ = 0;
    const int stride_h_ = 2;
    const int stride_w_ = 2;
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
    cu_algo = static_cast<cuinferConvolutionFwdAlgo_t>(0);  //DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
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

    (cuinferSetActivationDescriptor(
        cu_act_desc, CUINFER_ACTIVATION_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
        // cu_act_desc, CUINFER_ACTIVATION_IDENTITY, CUINFER_NOT_PROPAGATE_NAN, 0));//no activate

     // bias format 
    float *cu_d_bias = ( float *)bias.data_ptr() ;   // bias nullptr
    float *cu_alpha = (float *)alpha.data_ptr();  // perchannel alpha
    int *cu_d_workspace;
    const float alpha_ = 0.01;  // alpha
    const int beta_ = 0;
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
    
    // // // input:nhwc
    // const int batchsize_ = inputs.size(0);
    // const int chls_in_ = inputs.size(3);
    // const int height_in_ = inputs.size(1);
    // const int width_in_ = inputs.size(2); 

    // // weight: out_channels, kernel_size, kernel_size, in_channels
    // const int chls_out_ = weight.size(0);

    // // conv 
    // const int pad_h_ = 0;
    // const int pad_w_ = 0;
    // const int stride_h_ = 2;
    // const int stride_w_ = 2;
    // const int kernel_h_ =  weight.size(1);
    // const int kernel_w_ =  weight.size(2);

    // const int height_out = (height_in_ + 2 * pad_h_ - (kernel_h_ -1) - 1) / stride_h_ + 1;
    // const int width_out =  (width_in_ + 2 * pad_w_ - (kernel_w_ -1) - 1) / stride_w_ + 1;

    cuinferHandle_t cu_handle;
    cuinferTensorDescriptor_t cu_x_desc;
    cuinferConvolutionDescriptor_t cu_conv_desc;
    cuinferFilterDescriptor_t cu_w_desc;
    cuinferTensorDescriptor_t cu_y_desc;
    cuinferTensorDescriptor_t cu_bias_desc;
    cuinferActivationDescriptor_t cu_act_desc;

    const int group_count_ = 1;
    cuinferConvolutionFwdAlgo_t cu_algo;
    cu_algo = static_cast<cuinferConvolutionFwdAlgo_t>(0);  //DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
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

    (cuinferSetActivationDescriptor(
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

    
    (cuinferSetConvolutionGroupCount(cu_conv_desc, group_count_));
    size_t cuworkspace_size = 0;
    auto status = static_cast<int>(cuinferGetConvolutionForwardWorkspaceSize(
      cu_handle, cu_x_desc, cu_w_desc, cu_conv_desc, cu_y_desc, cu_algo,
      &cuworkspace_size));
    cudaMalloc(reinterpret_cast<void **>(&cu_d_workspace), cuworkspace_size);
  
    // at::Tensor cu_d_y = input.new_empty({batchsize_, height_out, width_out, chls_out_});
    
    // int8_t *wei_ptr = (int8_t *)weigh.data_ptr();
    // int8_t *inp_ptr = (int8_t *)input.data_ptr();
    // int8_t *res_ptr = (int8_t *)cu_d_y.data_ptr();
    // printf("===========here is ok ===============\n");

 
    cuinferQDConvolutionForward(
      cu_handle,  &alpha_, cu_alpha,&beta_,&gamma,
      cu_x_desc,inp_ptr,cu_w_desc, wei_ptr ,cu_conv_desc,cu_algo, 
      cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, res_ptr, 
      cu_bias_desc, cu_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, res_ptr);

    (cudaDeviceSynchronize());
    (cuinferDestroyTensorDescriptor(cu_x_desc));
    (cuinferDestroyConvolutionDescriptor(cu_conv_desc));
    (cuinferDestroyFilterDescriptor(cu_w_desc));
    (cuinferDestroyTensorDescriptor(cu_y_desc));
    (cuinferDestroy(cu_handle));
    cudaFree(cu_d_workspace);


    // return cu_d_y;
}

at::Tensor int8_conv2d(
    at::Tensor inputs,  // inputs  shape :nhwc 
    at::Tensor input,   // inputs  一维数据
    at::Tensor weight1,  // 
    at::Tensor weigh1,   // 一维数据
    at::Tensor alpha1,
    at::Tensor bias1,
    at::Tensor weight2,
    at::Tensor weigh2,  // 一维数据
    at::Tensor alpha2,
    at::Tensor bias2
)
{
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
    int8_t *wei_ptr = (int8_t *)weigh1.data_ptr();
    int8_t *inp_ptr = (int8_t *)input.data_ptr();
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
    int8_t *wei_ptr2 = (int8_t *)weigh2.data_ptr();
    int8_t *inp_ptr2 = (int8_t *)cu_out.data_ptr();
    int8_t *res_ptr2 = (int8_t *)cu_out2.data_ptr();

    Conv2d_int8(batchsize_, chls_out_, height_out, width_out, chls_out_2,
                pad_h_2,pad_w_2,stride_h_2,stride_w_2,kernel_h_2,kernel_w_2,h_out2,w_out2,
                cu_bias2,cu_alpha2,wei_ptr2,inp_ptr2,res_ptr2);

    return cu_out2;  

}


// cpu compute
struct LayoutNHWC {
    int N, H, W, C;

    LayoutNHWC() {}
    LayoutNHWC(int N, int H, int W, int C) : N(N), H(H), W(W), C(C) {}

    int operator()(int n, int h, int w, int c) { return n * H * W * C + h * W * C + w * C + c; }

    size_t count() { return (size_t)N * H * W * C; }
};

at::Tensor  cpuQDConvolutionForward(
                        at::Tensor inputs,
                        at::Tensor weight,
                        at::Tensor alpha,
                        at::Tensor input,
                        at::Tensor weigh ,
                        at::Tensor bias
                          ) {
    
    // inputs : NHWC
    const int l_batch = inputs.size(0) ;
    const int l_h = inputs.size(1) ;
    const int l_w = inputs.size(2) ;
    const int l_padded_c = inputs.size(3) ;

    // for(int i=0;i < 10;i++){
    //     printf("input_ %d\n", ( (int8_t *)input.data_ptr())[i]);
    // }

    // weight : outchannel kerner_h kernel_w inchannel
    const int l_size = weight.size(1) ;
    const int l_padded_outc = weight.size(0);

    LayoutNHWC layoutInput(l_batch,l_h,l_w,l_padded_c);
    LayoutNHWC layoutWeight(l_padded_outc,l_size,l_size,l_padded_c);

    // conv
    const int l_pad = 0;
    const int l_stride = 2;
    
    int P = ((l_h + l_pad * 2 - l_size) / l_stride) + 1;
    int Q = ((l_w + l_pad * 2 - l_size) / l_stride) + 1;

    const int l_padded_outputs = l_padded_outc * P * Q ;

    int32_t* output_q = new int32_t[l_batch*l_padded_outputs];

    // common set
    bool per_channel = true;
    bool use_leaky = false;
    const float  gamma = 0.1;
    const float *bias_ = (float *)bias.data_ptr();  //nullptr

    int output_size = l_batch * P * Q * l_padded_outc;
    // int8_t *y = new int8_t[output_size];
    
    at::Tensor y = input.new_empty({output_size});

    const int8_t* input_ = (int8_t *)input.data_ptr();
    const int8_t* weigh_ = (int8_t *)weigh.data_ptr();
    float *alpha_ = (float *)alpha.data_ptr();

    for (int n = 0; n < l_batch; ++n){
        for (int p = 0; p < P; ++p){
            for (int q = 0; q < Q; ++q){
                for (int k = 0; k < l_padded_outc; ++k) {
                    int32_t acc = 0;
                    int idx = n*P*Q*l_padded_outc + p*Q*l_padded_outc + q*l_padded_outc + k;// nhwc
                    // int idx = n*P*Q*l.padded_outc + k*P*Q + p*Q + q;// nchw
                    for (int r = 0; r < l_size; ++r){
                        int h = p * l_stride - l_pad + r;
                        for (int s = 0; s <l_size; ++s){
                            int w = q * l_stride - l_pad+ s;
                            for (int c = 0; c < l_padded_c; ++c) {
                                // int filter_r = r;
                                // int filter_s = s;
                                if (h >= 0 && h < l_h && w >= 0 && w < l_w) {
                                    int a = input_[layoutInput(n, h, w, c)];
                                    int b = weigh_[layoutWeight(k, r, s, c)];
                                    acc += a * b;
                                }
                            }
                        }
                    }
                    output_q[idx] = acc;
                    float tmp_output;
                    if(per_channel)
                        tmp_output = output_q[idx] * alpha_[k];
                    else
                        tmp_output = output_q[idx] * alpha_[0];
                    if(bias_ != nullptr)
                        tmp_output += bias_[k];
                    if(tmp_output < 0){
                      if(use_leaky){
                        tmp_output = tmp_output*gamma;
                      }
                      else{ // use relu
                        tmp_output = 0;
                        // tmp_output = tmp_output;//no activate
                      }
                    }
                    int output_val = std::round(tmp_output);
                    if(output_val < -128)
                        y[idx] = -128;
                    else if(output_val > 127)
                        y[idx] = 127;
                    else
                        y[idx] = output_val;
                }
            }
        }
    }
    delete [] output_q;
    
    return y;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_conv2d_relu", &conv2d_relu,"test conv2d_relu quant cpp");
    m.def("test_conv_cpu", &cpuQDConvolutionForward," cpu test conv quant cpp");
    // int8_conv2d
    m.def("test_basic", &int8_conv2d," gpu test int8_conv2d cpp");

}