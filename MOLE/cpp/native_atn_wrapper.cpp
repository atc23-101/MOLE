#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <immintrin.h>

#include <iostream>
#include <vector>

at::Tensor _native_dropout_backward_wrapper(float p, const at::Tensor &grad_output, const at::Tensor &mask)
{
    // 1 / (1 - p) = scale
    at::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());
    auto out = at::native_dropout_backward(grad_output, mask, 1.0f / (1.0f - p));
    return out;
}

std::tuple<at::Tensor, at::Tensor> _native_dropout_wrapper(float p, const at::Tensor &input)
{
    at::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());
    auto out = at::native_dropout(input, p, 1);
    return out;
}
at::Tensor _native_gelu_backward_wrapper(const at::Tensor &input, const at::Tensor &grad_output)
{
    at::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());
    auto out = at::gelu_backward(grad_output, input);
    return out;
}
at::Tensor _native_gelu_wrapper(const at::Tensor &input)
{
    at::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());
    auto out = at::gelu(input);
    return out;
}

at::Tensor _native_relu_wrapper(const at::Tensor &input)
{
    at::cuda::CUDAStreamGuard guard(c10::cuda::getCurrentCUDAStream());
    auto out = at::relu(input);
    return out;
}

PYBIND11_MODULE(native_atn_wrapper, m)
{
    m.def("native_dropout_backward_wrapper", &_native_dropout_backward_wrapper, " native_dropout_backward ");
    m.def("native_dropout_wrapper", &_native_dropout_wrapper, " native_dropout ");
    m.def("native_relu_wrapper", &_native_relu_wrapper, " native_relu ");
    m.def("native_gelu_wrapper", &_native_gelu_wrapper, " native_gelu ");
    m.def("native_gelu_backward_wrapper", &_native_gelu_backward_wrapper, " native_gelu ");
}
