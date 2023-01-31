#include "data_transfer.h"
#include "cuda_fp16.h"

void cuda_gpufp32_to_cpufp32(const float *input, float *output, int size, cudaStream_t stream)
{
    cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void cuda_cpufp32_to_gpufp32(const float *input, float *output, int size, cudaStream_t stream)
{
    cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyHostToDevice, stream);
}
void cuda_gpufp32_to_gpufp32(const float *input, float *output, int size, cudaStream_t stream)
{
    cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void cuda_gpufp16_to_cpufp16(const __half *input, __half *output, int size, cudaStream_t stream)
{
    cudaMemcpyAsync(output, input, size * sizeof(__half), cudaMemcpyDeviceToHost, stream);
}