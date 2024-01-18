#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudnn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "transformerKernels_int8_ix.h"
#include <vector>
using namespace std;
// #include <iostream>
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
namespace lightseq {
namespace cuda {

//step3 glu
// template <typename T>
void ker_bias_glu_ix_launcher(int8_t* outputs, const int8_t* inputs,
                              const float amax_in, const float amax_out,
                              const int n,
                              const int c, const int h, const int w,
                              cudaStream_t stream)
{
      lightseq_opt::cuda::ker_bias_glu_ix_launcher(outputs, inputs,amax_in, amax_out, n, c, h, w, stream);
}


at::Tensor convGlu( 
       at::Tensor outputs, 
       at::Tensor inputs,
       const float amax_in, 
       const float amax_out)
{

      //  const int8_t *bias = nullptr;
       const int n = outputs.size(0);
       const int c = outputs.size(1) ;
       const int h = outputs.size(2);    
       const int w = outputs.size(3);
       int8_t *out_ptr = (int8_t *)outputs.data_ptr();
       int8_t *inp_ptr = (int8_t *)inputs.data_ptr();

       cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
       ker_bias_glu_ix_launcher(out_ptr, inp_ptr,amax_in,amax_out, n, c, h, w, stream);
      //  ker_bias_glu_ix_launcher(outputs.data_ptr(), inputs.data_ptr(),alpha_, n, c, h, w, stream);

       return outputs;

}



}  // namespace cuda
}  // namespace lightseq