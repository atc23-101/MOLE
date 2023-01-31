
#include "data_transfer.h"

void _gpufp32_to_cpufp32(at::Tensor src, at::Tensor dst, int numel)
{
  cuda_gpufp32_to_cpufp32((const float *)src.data_ptr(), (float *)dst.data_ptr(), src.numel(), c10::cuda::getCurrentCUDAStream().stream());
}

void _gpufp32_addto_cpufp32(at::Tensor src, at::Tensor dst, int numel)
{
  dst += src.to("cpu");
}

void _gpufp32_to_gpufp32(at::Tensor src, at::Tensor dst, int numel)
{
  cuda_gpufp32_to_gpufp32((const float *)src.data_ptr(), (float *)dst.data_ptr(), src.numel(), c10::cuda::getCurrentCUDAStream().stream());
}
void _cpufp32_to_gpufp32(at::Tensor src, at::Tensor dst, int numel)
{
  cuda_cpufp32_to_gpufp32((const float *)src.data_ptr(), (float *)dst.data_ptr(), src.numel(), c10::cuda::getCurrentCUDAStream().stream());
}
void _gpufp16_to_cpufp16(at::Tensor src, at::Tensor dst, int numel)
{
  printf("GPU2CPU");
  cuda_gpufp16_to_cpufp16((const __half *)src.data_ptr(), (__half *)dst.data_ptr(), src.numel(), c10::cuda::getCurrentCUDAStream().stream());
}

/*
PYBIND11_MODULE(data_transfer, m) {
  m.def("cpufp32_to_gpufp32", &_cpufp32_to_gpufp32, " cpufp32_to_gpufp32 ");
  m.def("gpufp32_to_cpufp32", &_gpufp32_to_cpufp32, " cpufp32_to_gpufp32 ");
  m.def("gpufp32_addto_cpufp32", &_gpufp32_addto_cpufp32, " cpufp32_to_gpufp32 ");

  m.def("gpufp16_to_cpufp16", &_gpufp32_to_cpufp32, " gpufp16_to_cpufp16 ");
  m.def("gpufp32_to_gpufp32", &_gpufp32_to_gpufp32, " cpufp32_to_gpufp32 ");
  m.def("fast_assign_prefetch", &_fast_assign_prefetch, " fast_assign_prefetch ");
}
*/