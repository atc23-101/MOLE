#ifndef _DATA_TRANSFER_H_
#define _DATA_TRANSFER_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <immintrin.h>

#include <torch/csrc/api/include/torch/utils.h>

#include <iostream>
#include <vector>

void cuda_gpufp32_to_cpufp32(const float *input, float *output, int size, cudaStream_t stream);
void cuda_cpufp32_to_gpufp32(const float *input, float *output, int size, cudaStream_t stream);
void cuda_gpufp32_to_gpufp32(const float *input, float *output, int size, cudaStream_t stream);
void cuda_gpufp16_to_cpufp16(const __half *input, __half *output, int size, cudaStream_t stream);

void _gpufp32_to_cpufp32(at::Tensor src, at::Tensor dst, int numel);

void _gpufp32_addto_cpufp32(at::Tensor src, at::Tensor dst, int numel);

void _gpufp32_to_gpufp32(at::Tensor src, at::Tensor dst, int numel);

void _cpufp32_to_gpufp32(at::Tensor src, at::Tensor dst, int numel);

void _gpufp16_to_cpufp16(at::Tensor src, at::Tensor dst, int numel);

#endif