#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "gtest/gtest.h"
#include "ixinfer.h"

/// helper macros
#define ASSERT_CUDA(ret) ASSERT_EQ(cudaSuccess, ret)
#define EXPECT_CUDA(ret) EXPECT_EQ(cudaSuccess, ret)

#define ASSERT_CUINFER(ret) ASSERT_EQ(CUINFER_STATUS_SUCCESS, ret)
#define EXPECT_CUINFER(ret) EXPECT_EQ(CUINFER_STATUS_SUCCESS, ret)

double calcuate_time_span(std::chrono::time_point<std::chrono::steady_clock> beforeTime, std::chrono::time_point<std::chrono::steady_clock> afterTime){
    double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
    auto timespan = duration_millsecond;
    return timespan;
}


int GetTensorFormat(std::string tensor_format) {
  if (!tensor_format.compare("DNN_TENSOR_NCHW")) {
    return 0;
  } else if (!tensor_format.compare("DNN_TENSOR_NHWC")) {
    return 1;
  } else if (!tensor_format.compare("DNN_TENSOR_NCHW_VECT_C")) {
    return 2;
  } else {
    std::cout << "Tensor Format Error" << std::endl;
    return -1;
  }
}

int GetConvolutionFwdAlgo(std::string conv_fwd_algo) {
  if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM")) {
    return 0;
  } else if (!conv_fwd_algo.compare(
                 "DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM")) {
    return 1;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_GEMM")) {
    return 2;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_DIRECT")) {
    return 3;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_FFT")) {
    return 4;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_FFT_TILING")) {
    return 5;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_WINOGRAD")) {
    return 6;
  } else if (!conv_fwd_algo.compare(
                 "DNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED")) {
    return 7;
  } else if (!conv_fwd_algo.compare("DNN_CONVOLUTION_FWD_ALGO_COUNT")) {
    return 8;
  } else {
    std::cout << "ConvolutionForward Algo Error" << std::endl;
    return -1;
  }
}

int GetConvMode(std::string conv_mode) {
  if (!conv_mode.compare("DNN_CONVOLUTION")) {
    return 0;
  } else if (!conv_mode.compare("DNN_CROSS_CORRELATION")) {
    return 1;
  } else {
    std::cout << "Convolution Mode Error" << std::endl;
    return -1;
  }
}

int GetActiveMode(std::string active_mode) {
  if (!active_mode.compare("DNN_ACTIVATION_SIGMOID")) {
    return 0;
  } else if (!active_mode.compare("DNN_ACTIVATION_RELU")) {
    return 1;
  } else if (!active_mode.compare("DNN_ACTIVATION_TANH")) {
    return 2;
  } else if (!active_mode.compare("DNN_ACTIVATION_CLIPPED_RELU")) {
    return 3;
  } else if (!active_mode.compare("DNN_ACTIVATION_ELU")) {
    return 4;
  } else if (!active_mode.compare("DNN_ACTIVATION_IDENTITY")) {
    return 5;
  } else if (!active_mode.compare("DNN_ACTIVATION_LEAKY_RELU")) {
    return 6;
  } else if (!active_mode.compare("DNN_ACTIVATION_SILU")) {
    return 7;
  } else {
    std::cout << "Convolution Activation Mode Error" << std::endl;
    return -1;
  }
}

int GetDataType(std::string data_type) {
  if (!data_type.compare("DNN_DATA_FLOAT")) {
    return 0;
  } else if (!data_type.compare("DNN_DATA_DOUBLE")) {
    return 1;
  } else if (!data_type.compare("DNN_DATA_HALF")) {
    return 2;
  } else if (!data_type.compare("DNN_DATA_INT8")) {
    return 3;
  } else if (!data_type.compare("DNN_DATA_INT32")) {
    return 4;
  } else if (!data_type.compare("DNN_DATA_INT8x4")) {
    return 5;
  } else if (!data_type.compare("DNN_DATA_UINT8")) {
    return 6;
  } else if (!data_type.compare("DNN_DATA_UINT8x4")) {
    return 7;
  } else {
    std::cout << "Convolution Mode Error" << std::endl;
    return -1;
  }
}

struct LayoutNHWC {
    int N, H, W, C;

    LayoutNHWC() {}
    LayoutNHWC(int N, int H, int W, int C) : N(N), H(H), W(W), C(C) {}

    int operator()(int n, int h, int w, int c) { return n * H * W * C + h * W * C + w * C + c; }

    size_t count() { return (size_t)N * H * W * C; }
};

struct layer{
    int batch;
    int h;
    int w;
    int padded_c;
    int pad;
    int size;
    int stride;
    int padded_outc;
    int padded_outputs;
};

void run_cuinferQDConvolutionForward(layer l,
    const int8_t *cu_h_x, const int8_t *cu_h_w, const float *cu_h_bias, const float* cu_h_alpha,const int8_t* cu_h_element,
    const float* element_scale, const float* alpha2, const float* gamma, int8_t *cu_h_y,
    int group_count_, float *exec_time,bool use_pechannel,
    std::string conv_fwd_algo_, std::string conv_mode_,
    std::string x_tensor_format_, std::string y_tensor_format_,
    std::string filter_tensor_format_, std::string activate_mode_) {
  int batchsize_ = l.batch;
  int chls_in_ = l.padded_c;
  int height_in_ = l.h;
  int width_in_ = l.w;
  int chls_out_ = l.padded_outc;
  int kernel_h_ = l.size;
  int kernel_w_ = l.size;
  int pad_h_ = l.pad;
  int pad_w_ = l.pad;
  int stride_h_ = l.stride;
  int stride_w_ = l.stride;
  int dilation_h_ = 1;
  int dilation_w_ = 1;
  float alpha_ = cu_h_alpha[0];
  int beta_ = 0;

  // conv2d params
  int dilation_d_ = 1;
  int depth_in_ = 1;
  int kernel_d_ = 1;
  int stride_d_ = 1;
  int pad_d_ = 1;

  int nb_dims_ = depth_in_ == 1 ? 4 : 5;

  int depth_out =
      (depth_in_ + 2 * pad_d_ - (dilation_d_ * (kernel_d_ - 1) + 1)) /
          stride_d_ +
      1;
  int height_out =
      (height_in_ + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) /
          stride_h_ +
      1;
  int width_out =
      (width_in_ + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) /
          stride_w_ +
      1;

  int x_w_padding = 0;
  int x_h_padding = 0;
  int x_d_padding = 0;
  int x_c_padding = 0;

  int x_w_stride = 0;
  int x_h_stride = 0;
  int x_d_stride = 0;
  int x_c_stride = 0;
  int x_n_stride = 0;

  if ((!x_tensor_format_.compare("DNN_TENSOR_NCHW")) ||
      (!x_tensor_format_.compare("DNN_TENSOR_NCDHW"))) {
    x_w_stride = 1;
    x_h_stride = x_w_stride * width_in_ + x_w_padding;
    x_d_stride = x_h_stride * height_in_ + x_h_padding;
    x_c_stride = (nb_dims_ == 4 ? 1 : (depth_in_ + x_d_padding)) * x_d_stride;
    x_n_stride = chls_in_ * x_c_stride + x_c_padding;
  } else if ((!x_tensor_format_.compare("DNN_TENSOR_NHWC")) ||
             (!x_tensor_format_.compare("DNN_TENSOR_NDHWC"))) {
    x_c_stride = 1;
    x_w_stride = chls_in_ * x_c_stride + x_c_padding;
    x_h_stride = width_in_ * x_w_stride + x_w_padding;
    x_d_stride = height_in_ * x_h_stride + x_d_padding;
    x_n_stride = (nb_dims_ == 4 ? 1 : (depth_in_ + x_d_padding)) * x_d_stride;
  }

  int y_w_padding = 0;
  int y_h_padding = 0;
  int y_d_padding = 0;
  int y_c_padding = 0;

  int y_w_stride = 0;
  int y_h_stride = 0;
  int y_d_stride = 0;
  int y_c_stride = 0;
  int y_n_stride = 0;

  if ((!y_tensor_format_.compare("DNN_TENSOR_NCHW")) ||
      (!y_tensor_format_.compare("DNN_TENSOR_NCDHW"))) {
    y_w_stride = 1;
    y_h_stride = (y_w_stride * width_out) + y_w_padding;
    y_d_stride = (y_h_stride * height_out) + y_h_padding;
    y_c_stride =
        (nb_dims_ == 4 ? y_d_stride : (depth_out * y_d_stride + y_d_padding));
    y_n_stride = chls_out_ * y_c_stride + y_c_padding;
  } else if ((!y_tensor_format_.compare("DNN_TENSOR_NHWC")) ||
             (!y_tensor_format_.compare("DNN_TENSOR_NDHWC"))) {
    y_c_stride = 1;
    y_w_stride = (y_c_stride * chls_out_) + y_c_padding;
    y_h_stride = (y_w_stride * width_out) + y_w_padding;
    y_d_stride = (y_h_stride * height_out) + y_h_padding;
    y_n_stride = (nb_dims_ <= 4 ? 1 : (depth_out + y_d_padding)) * y_d_stride;
  }

  int data_type_size0 = sizeof(int8_t);
  int data_type_size2 = sizeof(int8_t);

  /** input mem size */
  size_t mem_size_input = (size_t)batchsize_ * x_n_stride * data_type_size0;
  /** output mem size */
  size_t mem_size_out = (size_t)batchsize_ * y_n_stride * data_type_size2;
  /** weight mem size */
  size_t mem_size_weight = (size_t)(chls_in_ / group_count_) *
                           (nb_dims_ > 4 ? kernel_d_ : 1) * kernel_h_ *
                           kernel_w_ * chls_out_ * data_type_size0;
  size_t mem_size_bias = (size_t)batchsize_ * chls_out_;

  cuinferHandle_t cu_handle;
  cuinferTensorDescriptor_t cu_x_desc;
  cuinferConvolutionDescriptor_t cu_conv_desc;
  cuinferFilterDescriptor_t cu_w_desc;
  cuinferTensorDescriptor_t cu_y_desc;
  cuinferTensorDescriptor_t cu_element_desc;
  cuinferTensorDescriptor_t cu_bias_desc;
  cuinferActivationDescriptor_t cu_act_desc;

  /** device mem ptr */
  int8_t *cu_d_x;
  int8_t *cu_d_y;
  int8_t *cu_d_element;
  int8_t *cu_d_w;
  float *cu_d_bias = nullptr;
  float *cu_d_alpha;
  int *cu_d_workspace;

  cuinferConvolutionFwdAlgo_t cu_algo;
  /** create handle and descriptor */
  ASSERT_CUINFER(cuinferCreate(&cu_handle));
  ASSERT_CUINFER(cuinferCreateTensorDescriptor(&cu_x_desc));
  ASSERT_CUINFER(cuinferCreateConvolutionDescriptor(&cu_conv_desc));
  ASSERT_CUINFER(cuinferCreateFilterDescriptor(&cu_w_desc));
  ASSERT_CUINFER(cuinferCreateTensorDescriptor(&cu_y_desc));
  ASSERT_CUINFER(cuinferCreateTensorDescriptor(&cu_element_desc));
  ASSERT_CUINFER(cuinferCreateTensorDescriptor(&cu_bias_desc));
  ASSERT_CUINFER(cuinferCreateActivationDescriptor(&cu_act_desc));

  cuinferTensorFormat_t cu_filter_tensor_format =
      static_cast<cuinferTensorFormat_t>(
          GetTensorFormat(filter_tensor_format_));
  cuinferConvolutionMode_t cu_conv_mode =
      static_cast<cuinferConvolutionMode_t>(GetConvMode(conv_mode_));
  cuinferDataType_t cu_data_type0 =
      static_cast<cuinferDataType_t>(CUINFER_DATA_INT8);
  cuinferDataType_t cu_data_type1 =
      static_cast<cuinferDataType_t>(CUINFER_DATA_INT32);
  cuinferDataType_t cu_data_type2 =
      static_cast<cuinferDataType_t>(CUINFER_DATA_FLOAT);
  cu_algo = static_cast<cuinferConvolutionFwdAlgo_t>(
      GetConvolutionFwdAlgo(conv_fwd_algo_));

  if (4 == nb_dims_) {

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_x_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_, chls_in_, height_in_, width_in_));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_y_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_,chls_out_, height_out, width_out));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_element_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_,chls_out_, height_out, width_out));

    ASSERT_CUINFER(cuinferSetConvolution2dDescriptor(
        cu_conv_desc,pad_h_, pad_w_,stride_h_, stride_w_, 1, 1, CUINFER_CROSS_CORRELATION, CUINFER_DATA_INT32));

    ASSERT_CUINFER(cuinferSetFilter4dDescriptor(
        cu_w_desc, CUINFER_DATA_INT8, CUINFER_TENSOR_NHWC, chls_out_, chls_in_ / group_count_, kernel_h_, kernel_w_));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_bias_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_FLOAT, 1, chls_out_, 1, 1));

    cuinferActivationMode_t cu_act_mode =
      static_cast<cuinferActivationMode_t>(GetActiveMode(activate_mode_));
    ASSERT_CUINFER(cuinferSetActivationDescriptor(
        cu_act_desc, cu_act_mode, CUINFER_NOT_PROPAGATE_NAN, 0));
    
  }
  /** set descriptor */
  ASSERT_CUINFER(cuinferSetConvolutionGroupCount(cu_conv_desc, group_count_));

  size_t cuworkspace_size = 0;
  auto status = static_cast<int>(cuinferGetConvolutionForwardWorkspaceSize(
      cu_handle, cu_x_desc, cu_w_desc, cu_conv_desc, cu_y_desc, cu_algo,
      &cuworkspace_size));
  EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_x), mem_size_input));
  EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_y), mem_size_out));
  EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_w), mem_size_weight));
  EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_alpha), mem_size_bias*sizeof(float)));
  EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_element), mem_size_out));
  EXPECT_CUDA(
      cudaMemcpy(cu_d_x, cu_h_x, mem_size_input, cudaMemcpyHostToDevice));
  EXPECT_CUDA(
      cudaMemcpy(cu_d_w, cu_h_w, mem_size_weight, cudaMemcpyHostToDevice));
  EXPECT_CUDA(cudaMemcpy(cu_d_element, cu_h_element, mem_size_out, cudaMemcpyHostToDevice));
  EXPECT_CUDA(cudaMemcpy(cu_d_y, cu_h_y, mem_size_out, cudaMemcpyHostToDevice));
  EXPECT_CUDA(cudaMemcpy(cu_d_alpha, cu_h_alpha, mem_size_bias*sizeof(float), cudaMemcpyHostToDevice));

  EXPECT_CUDA(
      cudaMalloc(reinterpret_cast<void **>(&cu_d_workspace), cuworkspace_size));

  if(cu_h_bias != nullptr){
    EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_bias), mem_size_bias));
    EXPECT_CUDA(cudaMemcpy(cu_d_bias, cu_h_bias, mem_size_bias, cudaMemcpyHostToDevice));
  }

  bool connectionBeforeActivation = false;
  /** exec API */
  EXPECT_CUINFER(cuinferQDConvolutionForward(
      cu_handle,  &alpha_, cu_d_alpha,&beta_,gamma,
      cu_x_desc,cu_d_x,cu_w_desc, cu_d_w, cu_conv_desc,cu_algo, 
      cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, cu_d_y, 
      cu_bias_desc, cu_d_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, cu_d_y));

  //warm up
  for(int i=0; i<5; i++)
    EXPECT_CUINFER(cuinferQDConvolutionForward(
        cu_handle,  &alpha_, cu_d_alpha,&beta_,gamma,
        cu_x_desc,cu_d_x,cu_w_desc, cu_d_w, cu_conv_desc,cu_algo, 
        cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, cu_d_y, 
        cu_bias_desc, cu_d_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, cu_d_y));

  auto before_run = std::chrono::steady_clock::now();
  int loop_count = 100;
  for(int i=0; i<loop_count; i++)
    EXPECT_CUINFER(cuinferQDConvolutionForward(
        cu_handle,  &alpha_, cu_d_alpha,&beta_,gamma,
        cu_x_desc,cu_d_x,cu_w_desc, cu_d_w, cu_conv_desc,cu_algo, 
        cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, cu_d_y, 
        cu_bias_desc, cu_d_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, cu_d_y));
  auto after_run = std::chrono::steady_clock::now();
  *exec_time = calcuate_time_span(before_run, after_run)/loop_count;

  // EXPECT_CUINFER(cuinferQDEConvolutionForward(
  //     cu_handle,  &alpha_, cu_d_alpha,&beta_,gamma,
  //     cu_x_desc,cu_d_x,cu_w_desc, cu_d_w, cu_conv_desc,cu_algo, 
  //     cu_d_workspace, cuworkspace_size,alpha2, element_scale,cu_element_desc, cu_d_element,cu_bias_desc, cu_d_bias,use_pechannel, cu_act_desc,connectionBeforeActivation,CUINFER_CONNECTION_ADD, cu_y_desc, cu_d_y));

  EXPECT_CUDA(cudaDeviceSynchronize());
  EXPECT_CUDA(cudaMemcpy(cu_h_y, cu_d_y, mem_size_out, cudaMemcpyDeviceToHost));

  EXPECT_CUDA(cudaFree(cu_d_workspace));
  EXPECT_CUDA(cudaFree(cu_d_x));
  EXPECT_CUDA(cudaFree(cu_d_w));
  EXPECT_CUDA(cudaFree(cu_d_y));
  EXPECT_CUDA(cudaFree(cu_d_alpha));
  if(cu_h_bias != nullptr){
    EXPECT_CUDA(cudaFree(cu_d_bias));
  }
  ASSERT_CUINFER(cuinferDestroyTensorDescriptor(cu_x_desc));
  ASSERT_CUINFER(cuinferDestroyConvolutionDescriptor(cu_conv_desc));
  ASSERT_CUINFER(cuinferDestroyFilterDescriptor(cu_w_desc));
  ASSERT_CUINFER(cuinferDestroyTensorDescriptor(cu_y_desc));
  ASSERT_CUINFER(cuinferDestroy(cu_handle));
}

int compare_qdconv(int batch_size, int input_h, int input_w, int input_c, int kernel_h, int kernel_w, int  output_c, int output_h, int output_w, int input_pad, int input_stride, 
        bool use_pechannel, bool use_bias,float leaky_gamma){
    int group_num = 1;
    int input_size = batch_size*input_h*input_w*input_c;
    int weight_size = output_c*input_c*kernel_h*kernel_w;
    int output_size = batch_size*output_h*output_w*output_c;
    int8_t *input_x = new int8_t[input_size];
    int8_t *input_weight = new int8_t[weight_size];
    int8_t *pre_output_y = new int8_t[output_size];
    int8_t *cuinfer_output_y = new int8_t[output_size];
    float *input_bias = new float[output_c];
    float *quant_alpha = new float[output_c];
    float pre_exec_time, exec_time;

    for(int i=0; i<input_size; ++i){
      // input_x[i] = rand()%256 -128;
      input_x[i] = rand()%30 -15;
      // input_x[i] = 25;
    }
    for(int i=0; i<weight_size; ++i){
      // input_weight[i] = rand()%256 -128;
      input_weight[i] = rand()%16 -8;
      // input_weight[i] = 5;
    }
    for(int i=0; i<output_c; ++i){
        // input_bias[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        // quant_alpha[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        input_bias[i] = 3 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        quant_alpha[i] = 0.1 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        // input_bias[i] = 3.9573;
        // quant_alpha[i] = 0.00132;
    }
    
    layer cur_l;
    cur_l.batch = batch_size;
    cur_l.h = input_h;
    cur_l.w = input_w;
    cur_l.padded_c = input_c;
    cur_l.pad = input_pad;
    cur_l.size = kernel_h;
    cur_l.stride = input_stride;
    cur_l.padded_outc = output_c;
    cur_l.padded_outputs = output_c*output_h*output_w;
    if(!use_bias){
      delete [] input_bias;
      input_bias = nullptr;
    }
    run_cuinferQDConvolutionForward(cur_l, input_x,input_weight,input_bias,quant_alpha,&leaky_gamma,pre_output_y,group_num,&pre_exec_time,
        use_pechannel,"DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM","DNN_CROSS_CORRELATION","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC","DNN_ACTIVATION_SILU");
    printf("pre_output_y: %d %d %d\n",pre_output_y[0],pre_output_y[1],pre_output_y[2]);
    run_cuinferQDConvolutionForward(cur_l, input_x,input_weight,input_bias,quant_alpha,&leaky_gamma,cuinfer_output_y,group_num,&exec_time,
        use_pechannel,"DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM","DNN_CROSS_CORRELATION","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC","DNN_ACTIVATION_SILU");
    printf("cuinfer_output_y: %d %d %d\n",cuinfer_output_y[0],cuinfer_output_y[1],cuinfer_output_y[2]);
    int max_abs_diff=0;
    int diff_num = 0;
    int cpu_sum=0, cuinfer_sum=0, nonzero_num=0;
    
    for(int i=0; i<output_size; ++i){
      int cur_abs_diff = std::abs(pre_output_y[i] - cuinfer_output_y[i]);
      cpu_sum += pre_output_y[i];
      cuinfer_sum += cuinfer_output_y[i];
      if(cur_abs_diff > max_abs_diff)
        max_abs_diff = cur_abs_diff;
      if(cur_abs_diff > 0){
        diff_num++;
        // std::cout<<"infer:"<<(int)(pre_output_y[i])<<" "<< (int)(cuinfer_output_y[i])<<std::endl;
      } 
      if(cuinfer_output_y[i] != 0) nonzero_num++;
    }
    std::cout<<"max diff: "<< max_abs_diff <<"  diff_num: "<<diff_num<<" total_num: "<<output_size<<std::endl;
    std::cout<<"sum: "<<cpu_sum<<" "<<cuinfer_sum<<" nonzero:"<<nonzero_num<<std::endl;
    std::cout<<"precom_time: "<<pre_exec_time<<"  imp_time: "<<exec_time<<std::endl;

    delete [] input_x;
    delete [] input_weight;
    delete [] pre_output_y;
    delete [] cuinfer_output_y;
    if(use_bias){
      delete [] input_bias;
    }
    delete [] quant_alpha;
    return 0;
}

//int compare_qdconv(int batch_size, int input_h, int input_w, int input_c, int kernel_h, int kernel_w, int  output_c, int output_h, int output_w, int input_pad, int input_stride,bool use_pechannel, bool use_bias,float leaky_gamma)

int test_conv(bool use_pechannel, bool use_bias){
    float leaky_gamma = 0.1;
    std::cout<<"========== Conv_0 [32, 640, 640, 4, 6, 6, 48, 320, 320, 2, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 640, 640, 4, 6, 6, 48, 320, 320, 2, 2, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_3 [32, 320, 320, 48, 3, 3, 96, 160, 160, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 320, 320, 48, 3, 3, 96, 160, 160, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_6 [32, 160, 160, 96, 1, 1, 48, 160, 160, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 160, 160, 96, 1, 1, 48, 160, 160, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_9 [32, 160, 160, 48, 1, 1, 48, 160, 160, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 160, 160, 48, 1, 1, 48, 160, 160, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_12 [32, 160, 160, 48, 3, 3, 48, 160, 160, 1, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 160, 160, 48, 3, 3, 48, 160, 160, 1, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_27 [32, 160, 160, 96, 1, 1, 96, 160, 160, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 160, 160, 96, 1, 1, 96, 160, 160, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_30 [32, 160, 160, 96, 3, 3, 192, 80, 80, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 160, 160, 96, 3, 3, 192, 80, 80, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_33 [32, 80, 80, 192, 1, 1, 96, 80, 80, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 80, 80, 192, 1, 1, 96, 80, 80, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_36 [32, 80, 80, 96, 1, 1, 96, 80, 80, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 80, 80, 96, 1, 1, 96, 80, 80, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_39 [32, 80, 80, 96, 3, 3, 96, 80, 80, 1, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 80, 80, 96, 3, 3, 96, 80, 80, 1, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_68 [32, 80, 80, 192, 1, 1, 192, 80, 80, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 80, 80, 192, 1, 1, 192, 80, 80, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_71 [32, 80, 80, 192, 3, 3, 384, 40, 40, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 80, 80, 192, 3, 3, 384, 40, 40, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_74 [32, 40, 40, 384, 1, 1, 192, 40, 40, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 40, 40, 384, 1, 1, 192, 40, 40, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_77 [32, 40, 40, 192, 1, 1, 192, 40, 40, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 40, 40, 192, 1, 1, 192, 40, 40, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_80 [32, 40, 40, 192, 3, 3, 192, 40, 40, 1, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 40, 40, 192, 3, 3, 192, 40, 40, 1, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_123 [32, 40, 40, 384, 1, 1, 384, 40, 40, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 40, 40, 384, 1, 1, 384, 40, 40, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_126 [32, 40, 40, 384, 3, 3, 768, 20, 20, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 40, 40, 384, 3, 3, 768, 20, 20, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_129 [32, 20, 20, 768, 1, 1, 384, 20, 20, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 20, 20, 768, 1, 1, 384, 20, 20, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_132 [32, 20, 20, 384, 1, 1, 384, 20, 20, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 20, 20, 384, 1, 1, 384, 20, 20, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_135 [32, 20, 20, 384, 3, 3, 384, 20, 20, 1, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 20, 20, 384, 3, 3, 384, 20, 20, 1, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_150 [32, 20, 20, 768, 1, 1, 768, 20, 20, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 20, 20, 768, 1, 1, 768, 20, 20, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_160 [32, 20, 20, 1536, 1, 1, 768, 20, 20, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 20, 20, 1536, 1, 1, 768, 20, 20, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_169 [32, 40, 40, 768, 1, 1, 192, 40, 40, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 40, 40, 768, 1, 1, 192, 40, 40, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_197 [32, 80, 80, 384, 1, 1, 96, 80, 80, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 80, 80, 384, 1, 1, 96, 80, 80, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_219 [32, 80, 80, 192, 3, 3, 192, 40, 40, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 80, 80, 192, 3, 3, 192, 40, 40, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_271 [32, 80, 80, 192, 1, 1, 255, 80, 80, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 80, 80, 192, 1, 1, 255, 80, 80, 0, 1, use_pechannel, use_bias, leaky_gamma);
    // std::cout<<"========== Conv_245 [32, 40, 40, 384, 3, 3, 384, 20, 20, 1, 2] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    // compare_qdconv(32, 40, 40, 384, 3, 3, 384, 20, 20, 1, 2, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_289 [32, 40, 40, 384, 1, 1, 255, 40, 40, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 40, 40, 384, 1, 1, 255, 40, 40, 0, 1, use_pechannel, use_bias, leaky_gamma);
    std::cout<<"========== Conv_307 [32, 20, 20, 768, 1, 1, 255, 20, 20, 0, 1] "<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(32, 20, 20, 768, 1, 1, 255, 20, 20, 0, 1, use_pechannel, use_bias, leaky_gamma);
  return 0;
}

int main(){
  test_conv(true, true);
  return 0;
}