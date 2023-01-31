#ifndef _GELU_HPP_
#define _GELU_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

void launch_d_gelu(float *d_output,
                   const float *input,
                   int intermediate_size,
                   int batch_size,
                   cudaStream_t stream);

#endif