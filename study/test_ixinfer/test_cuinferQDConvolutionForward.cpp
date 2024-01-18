#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <cmath>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "ixinfer.h"

/// helper macros
#define ASSERT_CUDA(ret) ASSERT_EQ(cudaSuccess, ret)
#define EXPECT_CUDA(ret) EXPECT_EQ(cudaSuccess, ret)

#define ASSERT_CUINFER(ret) ASSERT_EQ(CUINFER_STATUS_SUCCESS, ret)
#define EXPECT_CUINFER(ret) EXPECT_EQ(CUINFER_STATUS_SUCCESS, ret)


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
  } else if (!active_mode.compare("CUINFER_ACTIVATION_LEAKY_RELU")) {
    return 6;
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

void cpuQDConvolutionForward(layer l,
                            const float* alpha,
                           const float* gamma,
                           const int8_t* input,
                           const int8_t* weight,
                          const float* bias,
                          bool use_leaky,
                          bool per_channel,
                          int8_t* y) {
    LayoutNHWC layoutInput(l.batch,l.h,l.w,l.padded_c);
    LayoutNHWC layoutWeight(l.padded_outc,l.size,l.size,l.padded_c);
    int P = ((l.h + l.pad * 2 - l.size) / l.stride) + 1;
    int Q = ((l.w + l.pad * 2 - l.size) / l.stride) + 1;
    int32_t* output_q = new int32_t[l.batch*l.padded_outputs];
    for (int n = 0; n < l.batch; ++n){
        for (int p = 0; p < P; ++p){
            for (int q = 0; q < Q; ++q){
                for (int k = 0; k < l.padded_outc; ++k) {
                    int32_t acc = 0;
                    int idx = n*P*Q*l.padded_outc + p*Q*l.padded_outc + q*l.padded_outc + k;// nhwc
                    // int idx = n*P*Q*l.padded_outc + k*P*Q + p*Q + q;// nchw
                    for (int r = 0; r < l.size; ++r){
                        int h = p * l.stride - l.pad + r;
                        for (int s = 0; s < l.size; ++s){
                            int w = q * l.stride - l.pad + s;
                            for (int c = 0; c < l.padded_c; ++c) {
                                int filter_r = r;
                                int filter_s = s;
                                if (h >= 0 && h < l.h && w >= 0 && w < l.w) {
                                    int a = input[layoutInput(n, h, w, c)];
                                    int b = weight[layoutWeight(k, r, s, c)];
                                    acc += a * b;
                                }
                            }
                        }
                    }
                    output_q[idx] = acc;
                    float tmp_output;
                    if(per_channel)
                        tmp_output = output_q[idx] * alpha[k];
                    else
                        tmp_output = output_q[idx] * alpha[0];
                    if(bias != nullptr)
                        tmp_output += bias[k];
                    if(tmp_output < 0){
                      if(use_leaky){
                        tmp_output = tmp_output*gamma[0];
                      }
                      else{ // use relu
                        // tmp_output = 0;
                        tmp_output = tmp_output;//no activate
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
}

void run_cuinferQDConvolutionForward(layer l,
    const int8_t *cu_h_x, const int8_t *cu_h_w, const float *cu_h_bias, const float* cu_h_alpha, const float* gamma, int8_t *cu_h_y,
     int group_count_, float *exec_time,bool use_leaky,bool use_pechannel,
    std::string conv_fwd_algo_, std::string conv_mode_,
    std::string x_tensor_format_, std::string y_tensor_format_,
    std::string filter_tensor_format_) {
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
  cuinferTensorDescriptor_t cu_bias_desc;
  cuinferActivationDescriptor_t cu_act_desc;

  /** device mem ptr */
  int8_t *cu_d_x;
  int8_t *cu_d_y;
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
    // ASSERT_CUINFER(cuinferSetTensorNdDescriptor(
    //     cu_x_desc, cu_data_type0, nb_dims_,
    //     std::vector<int>({batchsize_, chls_in_, height_in_, width_in_}).data(),
    //     std::vector<int>({x_n_stride, x_c_stride, x_h_stride, x_w_stride}).data()));

    // ASSERT_CUINFER(cuinferSetTensorNdDescriptor(
    //     cu_y_desc, cu_data_type0, nb_dims_,
    //     std::vector<int>({batchsize_, chls_out_, height_out, width_out}).data(),
    //     std::vector<int>({y_n_stride, y_c_stride, y_h_stride, y_w_stride}).data()));

    // ASSERT_CUINFER(cuinferSetConvolutionNdDescriptor(
    //     cu_conv_desc, nb_dims_ - 2, std::vector<int>({pad_h_, pad_w_}).data(),
    //     std::vector<int>({stride_h_, stride_w_}).data(),
    //     std::vector<int>({dilation_h_, dilation_w_}).data(), cu_conv_mode,
    //     cu_data_type1));

    // ASSERT_CUINFER(cuinferSetFilterNdDescriptor(
    //     cu_w_desc, cu_data_type0, cu_filter_tensor_format, nb_dims_,
    //     std::vector<int>({chls_out_, chls_in_ / group_count_, kernel_h_, kernel_w_})
    //         .data()));

    // ASSERT_CUINFER(cuinferSetTensorNdDescriptor(
    //     cu_bias_desc, cu_data_type2, nb_dims_,
    //     std::vector<int>({batchsize_, chls_out_, height_in_, width_in_}).data(),
    //     std::vector<int>({x_n_stride, 0, 0, 0}).data()));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_x_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_, chls_in_, height_in_, width_in_));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_y_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_INT8, batchsize_,chls_out_, height_out, width_out));

    ASSERT_CUINFER(cuinferSetConvolution2dDescriptor(
        cu_conv_desc,pad_h_, pad_w_,stride_h_, stride_w_, 1, 1, CUINFER_CROSS_CORRELATION, CUINFER_DATA_INT32));

    ASSERT_CUINFER(cuinferSetFilter4dDescriptor(
        cu_w_desc, CUINFER_DATA_INT8, CUINFER_TENSOR_NHWC, chls_out_, chls_in_ / group_count_, kernel_h_, kernel_w_));

    ASSERT_CUINFER(cuinferSetTensor4dDescriptor(
        cu_bias_desc, CUINFER_TENSOR_NHWC, CUINFER_DATA_FLOAT, 1, chls_out_, 1, 1));

    if(use_leaky){
      ASSERT_CUINFER(cuinferSetActivationDescriptor(
          cu_act_desc, CUINFER_ACTIVATION_LEAKY_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
    }else{
      ASSERT_CUINFER(cuinferSetActivationDescriptor(
          // cu_act_desc, CUINFER_ACTIVATION_RELU, CUINFER_NOT_PROPAGATE_NAN, 0));
          cu_act_desc, CUINFER_ACTIVATION_IDENTITY, CUINFER_NOT_PROPAGATE_NAN, 0));//no activate
    }

    
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
  EXPECT_CUDA(
      cudaMemcpy(cu_d_x, cu_h_x, mem_size_input, cudaMemcpyHostToDevice));
  EXPECT_CUDA(
      cudaMemcpy(cu_d_w, cu_h_w, mem_size_weight, cudaMemcpyHostToDevice));
  EXPECT_CUDA(cudaMemcpy(cu_d_y, cu_h_y, mem_size_out, cudaMemcpyHostToDevice));
  EXPECT_CUDA(cudaMemcpy(cu_d_alpha, cu_h_alpha, mem_size_bias*sizeof(float), cudaMemcpyHostToDevice));

  EXPECT_CUDA(
      cudaMalloc(reinterpret_cast<void **>(&cu_d_workspace), cuworkspace_size));

  if(cu_h_bias != nullptr){
    EXPECT_CUDA(cudaMalloc(reinterpret_cast<void **>(&cu_d_bias), mem_size_bias));
    EXPECT_CUDA(cudaMemcpy(cu_d_bias, cu_h_bias, mem_size_bias, cudaMemcpyHostToDevice));
  }

  /** exec API */
  EXPECT_CUINFER(cuinferQDConvolutionForward(
      cu_handle,  &alpha_, cu_d_alpha,&beta_,gamma,
      cu_x_desc,cu_d_x,cu_w_desc, cu_d_w, cu_conv_desc,cu_algo, 
      cu_d_workspace, cuworkspace_size,&beta_,cu_y_desc, cu_d_y, 
      cu_bias_desc, cu_d_bias,cu_bias_desc,use_pechannel,cu_act_desc,cu_y_desc, cu_d_y));


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
        bool use_leaky,bool use_pechannel, bool use_bias,float leaky_gamma){
    int group_num = 1;
    int input_size = batch_size*input_h*input_w*input_c;
    int weight_size = output_c*input_c*kernel_h*kernel_w;
    int output_size = batch_size*output_h*output_w*output_c;
    int8_t *input_x = new int8_t[input_size];
    int8_t *input_weight = new int8_t[weight_size];
    int8_t *cpu_output_y = new int8_t[output_size];
    int8_t *cuinfer_output_y = new int8_t[output_size];
    float *input_bias = new float[output_c];
    float *quant_alpha = new float[output_c];
    float exec_time;

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
    cpuQDConvolutionForward(cur_l, quant_alpha, &leaky_gamma, input_x,input_weight,input_bias,use_leaky,use_pechannel,cpu_output_y);
    run_cuinferQDConvolutionForward(cur_l, input_x,input_weight,input_bias,quant_alpha,&leaky_gamma,cuinfer_output_y,group_num,&exec_time,
        use_leaky,use_pechannel,"DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM","DNN_CROSS_CORRELATION","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC","DNN_TENSOR_NHWC");
    int max_abs_diff=0;
    int diff_num = 0;
    int cpu_sum=0, cuinfer_sum=0, nonzero_num=0;
    printf("cpu_output_y: %d %d %d\n",cpu_output_y[0],cpu_output_y[1],cpu_output_y[2]);
    printf("cuinfer_output_y: %d %d %d\n",cuinfer_output_y[0],cuinfer_output_y[1],cuinfer_output_y[2]);
    for(int i=0; i<output_size; ++i){
      int cur_abs_diff = std::abs(cpu_output_y[i] - cuinfer_output_y[i]);
      cpu_sum += cpu_output_y[i];
      cuinfer_sum += cuinfer_output_y[i];
      if(cur_abs_diff > max_abs_diff)
        max_abs_diff = cur_abs_diff;
      if(cur_abs_diff > 0){
        diff_num++;
        // std::cout<<"infer:"<<(int)(cpu_output_y[i])<<" "<< (int)(cuinfer_output_y[i])<<std::endl;
      } 
      if(cuinfer_output_y[i] != 0) nonzero_num++;
    }
    std::cout<<"max diff: "<< max_abs_diff <<"  diff_num: "<<diff_num<<" total_num: "<<output_size<<std::endl;
    std::cout<<"sum: "<<cpu_sum<<" "<<cuinfer_sum<<" nonzero:"<<nonzero_num<<std::endl;

    delete [] input_x;
    delete [] input_weight;
    delete [] cpu_output_y;
    delete [] cuinfer_output_y;
    if(use_bias){
      delete [] input_bias;
    }
    delete [] quant_alpha;
    return 0;
}

//int compare_qdconv(int batch_size, int input_h, int input_w, int input_c, int kernel_h, int kernel_w, int  output_c, int output_h, int output_w, int input_pad, int input_stride, bool use_leaky,bool use_pechannel, bool use_bias,float leaky_gamma)

int test_conv(bool use_leaky,bool use_pechannel, bool use_bias){
    float leaky_gamma = 0.1;
    std::cout<<"========== result of resnet conv-15:  "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 56, 56, 64, 1, 1, 128, 28, 28, 0, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//resnet18
    std::cout<<"========== result of conv2d-1:  4, 416, 416, 32, 3, 3, 32, 416, 416, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 416, 416, 4, 3, 3, 32, 416, 416, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-1
    std::cout<<"========== result of conv2d-4:  16, 416, 416, 32, 3, 3, 64, 208, 208, 1, 2, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 416, 416, 32, 3, 3, 64, 208, 208, 1, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-4
    std::cout<<"========== result of conv2d-7:  16, 208, 208, 64, 1, 1, 32, 208, 208, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 208, 208, 64, 1, 1, 32, 208, 208, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-7
    std::cout<<"========== result of conv2d-10:  16, 208, 208, 32, 3, 3, 64, 208, 208, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 208, 208, 32, 3, 3, 64, 208, 208, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-10
    std::cout<<"========== result of conv2d-14:  16, 208, 208, 64, 3, 3, 128, 104, 104, 1, 2, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 208, 208, 64, 3, 3, 128, 104, 104, 1, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-14
    std::cout<<"========== result of conv2d-17:  16, 104, 104, 128, 1, 1, 64, 104, 104, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 104, 104, 128, 1, 1, 64, 104, 104, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-17
    std::cout<<"========== result of conv2d-20:  16, 104, 104, 64, 3, 3, 128, 104, 104, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 104, 104, 64, 3, 3, 128, 104, 104, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-20
    std::cout<<"========== result of conv2d-31:  16, 104, 104, 128, 3, 3, 256, 52, 52, 1, 2, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 104, 104, 128, 3, 3, 256, 52, 52, 1, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-31
    std::cout<<"========== result of conv2d-34:  16, 52, 52, 256, 1, 1, 128, 52, 52, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 52, 52, 256, 1, 1, 128, 52, 52, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-34
    std::cout<<"========== result of conv2d-37:  16, 52, 52, 128, 3, 3, 256, 52, 52, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 52, 52, 128, 3, 3, 256, 52, 52, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-37
    std::cout<<"========== result of conv2d-90:  16, 52, 52, 256, 3, 3, 512, 26, 26, 1, 2, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 52, 52, 256, 3, 3, 512, 26, 26, 1, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-90
    std::cout<<"========== result of conv2d-93:  16, 26, 26, 512, 1, 1, 512, 26, 26, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 512, 1, 1, 512, 26, 26, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-93
    std::cout<<"========== result of conv2d-96:  16, 26, 26, 256, 3, 3, 512, 26, 26, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 256, 3, 3, 512, 26, 26, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-96
    std::cout<<"========== result of conv2d-149:  16, 26, 26, 512, 3, 3, 1024, 13, 13, 1, 2, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 512, 3, 3, 1024, 13, 13, 1, 2, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-149
    std::cout<<"========== result of conv2d-152:  16, 13, 13, 1024, 1, 1, 512, 13, 13, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 13, 13, 1024, 1, 1, 512, 13, 13, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-152
    std::cout<<"========== result of conv2d-155:  16, 13, 13, 512, 3, 3, 1024, 13, 13, 1, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 13, 13, 512, 3, 3, 1024, 13, 13, 1, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-155
    std::cout<<"========== result of conv2d-199:  16, 13, 13, 512, 1, 1, 256, 13, 13, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 13, 13, 512, 1, 1, 256, 13, 13, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-199
    std::cout<<"========== result of conv2d-204:  16, 26, 26, 768, 1, 1, 256, 26, 26, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 768, 1, 1, 256, 26, 26, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-204
    std::cout<<"========== result of conv2d-224:  16, 26, 26, 256, 1, 1, 128, 26, 26, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 256, 1, 1, 128, 26, 26, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-224
    std::cout<<"========== result of conv2d-229:  16, 52, 52, 384, 1, 1, 128, 52, 52, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 52, 52, 384, 1, 1, 128, 52, 52, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-229
    std::cout<<"========== result of conv2d-250:  16, 52, 52, 256, 1, 1, 256, 52, 52, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 52, 52, 256, 1, 1, 256, 52, 52, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-250
    std::cout<<"========== result of conv2d-251:  16, 26, 26, 512, 1, 1, 256, 26, 26, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 26, 26, 512, 1, 1, 256, 26, 26, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-251
    std::cout<<"========== result of conv2d-252:  16, 13, 13, 1024, 1, 1, 256, 13, 13, 0, 1, "<<use_leaky<<", "<<use_pechannel<<", "<<use_bias<<", 0.01"<<std::endl;
    compare_qdconv(16, 13, 13, 1024, 1, 1, 256, 13, 13, 0, 1, use_leaky, use_pechannel, use_bias, leaky_gamma);//conv2d-252
  return 0;
}

int main(){
  // bool on_off_flags[2]={true, false};
  // for(int i=0; i<2; ++i){
  //   for(int j=0; j<2; ++j){
  //     for(int k=0; k<2; ++k){
  //       test_conv(on_off_flags[i],on_off_flags[j], on_off_flags[k]);
  //     }
  //   }
  // }  

  // test_conv(true,true, true);
  test_conv(false,true, true);
  return 0;
}