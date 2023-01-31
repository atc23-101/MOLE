#include "gelu.hpp"
#include "define.hpp"

void gelu_inplace_backward(int bsz, int intermediate_size, float *d_output, const float *input_buf, cudaStream_t stream)
{
    launch_d_gelu(d_output, input_buf, intermediate_size, bsz, stream);
}