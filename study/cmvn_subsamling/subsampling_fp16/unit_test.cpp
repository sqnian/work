// #include <cudnn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <iostream>
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


void print_element_conf(__half* data_ptr, int size) {
  std::vector<__half> h_data(size);
  CUDA_CHECK(cudaMemcpy(h_data.data(), data_ptr, size * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  for(int i=0;i<size;++i){
      std::cout << "  " << (float)h_data[i] << "  ";
  }
  std::cout << "" << std::endl;
}

// test ixinfer cuinferConvolutionBiasActivationForward
at::Tensor conv2d_fp16(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias
)
{
    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.is_cuda());

    TORCH_CHECK(weight.dim() == 4);
    TORCH_CHECK(weight.is_contiguous());
    TORCH_CHECK(weight.is_cuda());
    TORCH_CHECK(bias.dim() == 1);
    // TORCH_CHECK(bias.is_contiguous());
    TORCH_CHECK(bias.is_cuda());

    // input n h  w c 
    const int n = input.size(0);
    const int c_in = input.size(3);
    const int h_in = input.size(1);
    const int w_in = input.size(2);

    // weight: out_channels, kernel_size, kernel_size, in_channels
    const int c_out = weight.size(0);

    // conv 
    const int pad_h = 0;
    const int pad_w = 0;
    const int stride_h = 2;
    const int stride_w = 2;
    const int kernel_h =  weight.size(1);
    const int kernel_w =  weight.size(2);

    // std::cout << "kernel_h: " << kernel_h << std::endl;
    // std::cout << "kernel_w: " << kernel_w << std::endl;


    const int h_out = ( h_in + 2 * pad_h - (kernel_h -1) - 1) / stride_h + 1;
    const int w_out =  ( w_in + 2 * pad_w - (kernel_w -1) - 1) / stride_w + 1;

    const int group_count_ = 1 ;


    cuinferHandle_t handle;
    CUINFER_CHECK(cuinferCreate(&handle));

    cuinferTensorDescriptor_t input_descriptor;
    cuinferConvolutionDescriptor_t convolution_descriptor;
    cuinferFilterDescriptor_t kernel_descriptor;
    cuinferTensorDescriptor_t bias_descriptor;
    cuinferActivationDescriptor_t activation_descriptor;
    cuinferTensorDescriptor_t output_descriptor;


    CUINFER_CHECK(cuinferCreateTensorDescriptor(&input_descriptor));
    CUINFER_CHECK(cuinferCreateConvolutionDescriptor(&convolution_descriptor));
    CUINFER_CHECK(cuinferCreateFilterDescriptor(&kernel_descriptor));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&bias_descriptor));
    CUINFER_CHECK(cuinferCreateActivationDescriptor(&activation_descriptor));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&output_descriptor));

    CUINFER_CHECK(cuinferSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,  //  CUINFER_TENSOR_NHWC
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*batch_size=*/n,
                                      /*channels=*/c_in,
                                      /*image_height=*/h_in,
                                      /*image_width=*/w_in));
    CUINFER_CHECK(cuinferSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/pad_h,
                                           /*pad_width=*/pad_w,
                                           /*vertical_stride=*/stride_h,
                                           /*horizontal_stride=*/stride_w,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUINFER_CROSS_CORRELATION,
                                           /*computeType=*/CUINFER_DATA_FLOAT));                                  
    CUINFER_CHECK(cuinferSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*out_channels=*/c_out,
                                      /*in_channels=*/c_in / group_count_,
                                      /*kernel_height=*/kernel_h,
                                      /*kernel_width=*/kernel_w));   
    CUINFER_CHECK(cuinferSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*dataType=*/CUINFER_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/c_out,
                                      /*image_height=*/1,
                                      /*image_width=*/1));
    
    // cuinferActivationMode_t activation_model = static_cast<cuinferActivationMode_t>(0) ;
    CUINFER_CHECK(cuinferSetActivationDescriptor(activation_descriptor,
                                        /*mode=*/    CUINFER_ACTIVATION_RELU,//CUINFER_ACTIVATION_IDENTITY,   // no activate  // CUDNN_ACTIVATION_RELU,  //  CUDNN_ACTIVATION_SIGMOID
                                        /*reluNanOpt=*/CUINFER_NOT_PROPAGATE_NAN,
                                        /*relu_coef=*/0));                                 
    // printf("n = %d\n",n);
    // printf("c_out = %d\n",c_out);
    // printf("h_out = %d\n",h_out);
    // printf("w_out = %d\n",w_out);
    CUINFER_CHECK(cuinferSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*batch_size=*/n,
                                      /*channels=*/c_out,
                                      /*image_height=*/h_out,
                                      /*image_width=*/w_out));
    
    CUINFER_CHECK(cuinferSetConvolutionGroupCount(convolution_descriptor, group_count_));

    
    // fp16 只有 0 和 2 
    cuinferConvolutionFwdAlgo_t convolution_algorithm = static_cast<cuinferConvolutionFwdAlgo_t>(0) ;  //CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

    size_t workspace_bytes = 0;
    CUINFER_CHECK(cuinferGetConvolutionForwardWorkspaceSize(handle,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));
    // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    //         << std::endl;

    //  bias != nullptr:   
    //    y = Act( alpha1 * conv(x) + bias)
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
    float *cu_bias = (float *)bias.data_ptr();  //nullptr ; // (float *)bias.data_ptr(); // nullptr;
    int *work_space;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work_space), workspace_bytes));

    __half *inp_ptr = (__half *)input.data_ptr();
    __half *wei_ptr = (__half *)weight.data_ptr();

    // print_element_conf(inp_ptr,10); 

    float gamma = 0.0f;
    float beta = 0.0f;
    bool connectionBeforeActivation = false ;
    cuinferTensorConnectionMode_t connectionDesc = static_cast<cuinferTensorConnectionMode_t>(0) ;
    // std::cerr << "connectionDesc :" << connectionDesc << std::endl;

    at::Tensor output = input.new_empty({n, h_out, w_out, c_out});
    __half *out_ptr = (__half *)output.data_ptr();

    CUINFER_CHECK(cuinferHalfConvolution2dForward(
                handle, 
                &alpha1, 
                &beta,
                &gamma,
                input_descriptor, 
                inp_ptr, 
                kernel_descriptor, 
                wei_ptr, 
                convolution_descriptor,
                convolution_algorithm, 
                work_space,   
                workspace_bytes, 
                &alpha2, 
                output_descriptor, 
                out_ptr,
                bias_descriptor,
                cu_bias,
                activation_descriptor,
                connectionBeforeActivation,
                connectionDesc,
                output_descriptor,
                out_ptr));

    CUINFER_CHECK(cuinferDestroyTensorDescriptor(input_descriptor));
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(output_descriptor));
    CUINFER_CHECK(cuinferDestroyConvolutionDescriptor(convolution_descriptor));
    CUINFER_CHECK(cuinferDestroyFilterDescriptor(kernel_descriptor));
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(bias_descriptor));
    CUINFER_CHECK(cuinferDestroyActivationDescriptor(activation_descriptor));
    CUDA_CHECK(cudaFree(work_space));

    return output;

}


// conv2d relu 

void conv2d_relu(
    const int n,
    const int c_in,
    const int h_in,
    const int w_in,
    const int c_out,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int kernel_h,
    const int kernel_w,
    const int h_out,
    const int w_out,
    float *cu_bias,   
    __half *wei_ptr,
    __half *inp_ptr,
    __half *out_ptr  
)
{
    const int group_count_ = 1 ;

    cuinferHandle_t handle;
    CUINFER_CHECK(cuinferCreate(&handle));

    cuinferTensorDescriptor_t input_descriptor;
    cuinferConvolutionDescriptor_t convolution_descriptor;
    cuinferFilterDescriptor_t kernel_descriptor;
    cuinferTensorDescriptor_t bias_descriptor;
    cuinferActivationDescriptor_t activation_descriptor;
    cuinferTensorDescriptor_t output_descriptor;


    CUINFER_CHECK(cuinferCreateTensorDescriptor(&input_descriptor));
    CUINFER_CHECK(cuinferCreateConvolutionDescriptor(&convolution_descriptor));
    CUINFER_CHECK(cuinferCreateFilterDescriptor(&kernel_descriptor));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&bias_descriptor));
    CUINFER_CHECK(cuinferCreateActivationDescriptor(&activation_descriptor));
    CUINFER_CHECK(cuinferCreateTensorDescriptor(&output_descriptor));

    CUINFER_CHECK(cuinferSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,  //  CUINFER_TENSOR_NHWC
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*batch_size=*/n,
                                      /*channels=*/c_in,
                                      /*image_height=*/h_in,
                                      /*image_width=*/w_in));
    CUINFER_CHECK(cuinferSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/pad_h,
                                           /*pad_width=*/pad_w,
                                           /*vertical_stride=*/stride_h,
                                           /*horizontal_stride=*/stride_w,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUINFER_CROSS_CORRELATION,
                                           /*computeType=*/CUINFER_DATA_FLOAT));                                  
    CUINFER_CHECK(cuinferSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*out_channels=*/c_out,
                                      /*in_channels=*/c_in / group_count_,
                                      /*kernel_height=*/kernel_h,
                                      /*kernel_width=*/kernel_w));   
    CUINFER_CHECK(cuinferSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*dataType=*/CUINFER_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/c_out,
                                      /*image_height=*/1,
                                      /*image_width=*/1));
    
    // cuinferActivationMode_t activation_model = static_cast<cuinferActivationMode_t>(0) ;
    CUINFER_CHECK(cuinferSetActivationDescriptor(activation_descriptor,
                                        /*mode=*/    CUINFER_ACTIVATION_RELU,//CUINFER_ACTIVATION_IDENTITY,   // no activate  // CUDNN_ACTIVATION_RELU,  //  CUDNN_ACTIVATION_SIGMOID
                                        /*reluNanOpt=*/CUINFER_NOT_PROPAGATE_NAN,
                                        /*relu_coef=*/0));                                 
    // printf("n = %d\n",n);
    // printf("c_out = %d\n",c_out);
    // printf("h_out = %d\n",h_out);
    // printf("w_out = %d\n",w_out);
    CUINFER_CHECK(cuinferSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUINFER_TENSOR_NHWC,
                                      /*dataType=*/CUINFER_DATA_HALF,
                                      /*batch_size=*/n,
                                      /*channels=*/c_out,
                                      /*image_height=*/h_out,
                                      /*image_width=*/w_out));
    
    CUINFER_CHECK(cuinferSetConvolutionGroupCount(convolution_descriptor, group_count_));

    
    // fp16 只有 0 和 2 
    cuinferConvolutionFwdAlgo_t convolution_algorithm = static_cast<cuinferConvolutionFwdAlgo_t>(0) ;  //CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

    size_t workspace_bytes = 0;
    CUINFER_CHECK(cuinferGetConvolutionForwardWorkspaceSize(handle,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));
                                        
    //  bias != nullptr:   
    //    y = Act( alpha1 * conv(x) + bias)
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
    // float *cu_bias = (float *)bias.data_ptr();  //nullptr ; // (float *)bias.data_ptr(); // nullptr;
    int *work_space;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work_space), workspace_bytes));

    float gamma = 0.0f;
    float beta = 0.0f;
    bool connectionBeforeActivation = false ;
    cuinferTensorConnectionMode_t connectionDesc = static_cast<cuinferTensorConnectionMode_t>(0) ;
    
    CUINFER_CHECK(cuinferHalfConvolution2dForward(
                handle, 
                &alpha1, 
                &beta,
                &gamma,
                input_descriptor, 
                inp_ptr, 
                kernel_descriptor, 
                wei_ptr, 
                convolution_descriptor,
                convolution_algorithm, 
                work_space,   
                workspace_bytes, 
                &alpha2, 
                output_descriptor, 
                out_ptr,
                bias_descriptor,
                cu_bias,
                activation_descriptor,
                connectionBeforeActivation,
                connectionDesc,
                output_descriptor,
                out_ptr));

    CUINFER_CHECK(cuinferDestroyTensorDescriptor(input_descriptor));
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(output_descriptor));
    CUINFER_CHECK(cuinferDestroyConvolutionDescriptor(convolution_descriptor));
    CUINFER_CHECK(cuinferDestroyFilterDescriptor(kernel_descriptor));
    CUINFER_CHECK(cuinferDestroyTensorDescriptor(bias_descriptor));
    CUINFER_CHECK(cuinferDestroyActivationDescriptor(activation_descriptor));
    CUDA_CHECK(cudaFree(work_space));

} 
// 两层 conv2d+ relu 
at::Tensor conv2d_relu_2(
    at::Tensor input,
    at::Tensor weight1,
    at::Tensor bias1,
    at::Tensor weight2,
    at::Tensor bias2

){
    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

    TORCH_CHECK(weight1.dim() == 4);
    TORCH_CHECK(weight1.is_contiguous());
    TORCH_CHECK(weight1.is_cuda());
    TORCH_CHECK(weight1.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(bias1.dim() == 1);
    TORCH_CHECK(bias1.is_cuda());
    TORCH_CHECK(bias1.scalar_type() == at::ScalarType::Float);

    TORCH_CHECK(weight2.dim() == 4);
    TORCH_CHECK(weight2.is_contiguous());
    TORCH_CHECK(weight2.is_cuda());
    TORCH_CHECK(weight2.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(bias2.dim() == 1);
    TORCH_CHECK(bias2.is_cuda());
    TORCH_CHECK(bias2.scalar_type() == at::ScalarType::Float);

    // input n h  w c 
    const int n = input.size(0);
    const int c_in = input.size(3);
    const int h_in = input.size(1);
    const int w_in = input.size(2);

    // weight: out_channels, kernel_size, kernel_size, in_channels
    const int c_out = weight1.size(0);

    // conv 
    const int pad_h = 0;
    const int pad_w = 0;
    const int stride_h = 2;
    const int stride_w = 2;
    const int kernel_h =  weight1.size(1);
    const int kernel_w =  weight1.size(2);

    // std::cout << "kernel_h: " << kernel_h << std::endl;
    // std::cout << "kernel_w: " << kernel_w << std::endl;


    const int h_out = ( h_in + 2 * pad_h - (kernel_h -1) - 1) / stride_h + 1;
    const int w_out =  ( w_in + 2 * pad_w - (kernel_w -1) - 1) / stride_w + 1;

    float *cu_bias1 = (float *)bias1.data_ptr();
    __half *inp_ptr = (__half *)input.data_ptr();
    __half *wei_ptr = (__half *)weight1.data_ptr();

    at::Tensor output = input.new_empty({n, h_out, w_out, c_out});
    __half *out_ptr = (__half *)output.data_ptr();

    conv2d_relu(n, c_in, h_in, w_in, c_out,
                pad_h,pad_w,stride_h,stride_w,kernel_h,kernel_w,h_out,w_out,
                cu_bias1,wei_ptr,inp_ptr,out_ptr);

    // weight2: out_channels, kernel_size, kernel_size, in_channels
    const int c_out_2 = weight2.size(0);

    // conv 
    const int pad_h_2 = 0;
    const int pad_w_2 = 0;
    const int stride_h_2 = 2;
    const int stride_w_2 = 2;
    const int kernel_h_2 =  weight2.size(1);
    const int kernel_w_2 =  weight2.size(2);

    const int h_out_2 = ( h_out + 2 * pad_h_2 - (kernel_h_2 -1) - 1) / stride_h_2 + 1;
    const int w_out_2 =  ( w_out + 2 * pad_w_2 - (kernel_w_2 -1) - 1) / stride_w_2 + 1;

    float *cu_bias2 = (float *)bias2.data_ptr();
    // __half *inp_ptr = (__half *)input.data_ptr();
    __half *wei_ptr_2 = (__half *)weight2.data_ptr();

    at::Tensor output_2 = output.new_empty({n, h_out_2, w_out_2, c_out_2});
    __half *out_ptr_2 = (__half *)output_2.data_ptr();

    conv2d_relu(n, c_out, h_out, w_out, c_out_2,
                pad_h_2,pad_w_2,stride_h_2,stride_w_2,kernel_h_2,kernel_w_2,h_out_2,w_out_2,
                cu_bias2,wei_ptr_2,out_ptr,out_ptr_2);

    return output_2;

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
   // test conv2d fp16 api
    m.def("test_conv", &conv2d_fp16,"test conv2d fp16 api");
    m.def("test_func", &conv2d_relu_2,"test double conv2d and relu fp16 api");


}