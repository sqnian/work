#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace relu {
namespace cuda {
at::Tensor basic_conv(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1,
                      at::Tensor weight2, at::Tensor bias2);
        
at::Tensor convrelu(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1);

}
}  // namespace relu

at::Tensor conv2d(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1,
                  at::Tensor weight2, at::Tensor bias2) {
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(weight1.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(bias1.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(bias2.scalar_type() == at::ScalarType::Half);

  return relu::cuda::basic_conv(inputs, weight1, bias1, weight2, bias2);
}

at::Tensor conv2d_relu(at::Tensor inputs, at::Tensor weight1, at::Tensor bias1) {

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(weight1.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(bias1.scalar_type() == at::ScalarType::Half);

  return relu::cuda::convrelu(inputs, weight1, bias1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &conv2d, "fast depthwise conv1d forward");
  m.def("conv2d_relu", &conv2d_relu, "fast depthwise conv2d forward");
}