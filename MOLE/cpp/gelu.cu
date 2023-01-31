/*Adapt from deepspeed*/

inline __device__ float d_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanhf(sqrt_param * (x + x * x2mul));
    float dg1 = 0.5f * (1.0f + tan_h);
    float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3 = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

__global__ void d_gelu_func(float *d_output,
                            const float *gelu_input,
                            int row_stride,
                            int iterations)
{
    int row = blockIdx.x;
    int id = threadIdx.x;
    int loop_stride = blockDim.x;

    float4 *d_output_cast = reinterpret_cast<float4 *>(d_output);
    const float4 *gelu_input_cast = reinterpret_cast<const float4 *>(gelu_input);

    for (int i = 0; i < iterations; i++)
    {
        if (i * loop_stride + id < row_stride)
        {
            float4 output_data = d_output_cast[row * row_stride + i * loop_stride + id];
            float4 gelu_input_data = gelu_input_cast[row * row_stride + i * loop_stride + id];

            output_data.x *= d_gelu(gelu_input_data.x);
            output_data.y *= d_gelu(gelu_input_data.y);
            output_data.z *= d_gelu(gelu_input_data.z);
            output_data.w *= d_gelu(gelu_input_data.w);

            d_output_cast[row * row_stride + i * loop_stride + id] = output_data;
        }
    }
}

void launch_d_gelu(float *d_output,
                   const float *input,
                   int intermediate_size,
                   int batch_size,
                   cudaStream_t stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    dim3 block_dims(threads);
    dim3 grid_dims(batch_size);

    d_gelu_func<<<grid_dims, block_dims, 0, stream>>>(
        d_output, input, intermediate_size / 4, iterations);
}