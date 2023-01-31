#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "define.hpp"
#include "data_transfer.h"

at::Tensor _GPUParamCore::fetch_single(int id)
{
    return this->all_tensors.index({id});
}

at::Tensor _GPUParamCore::fetch_single_grad(int id)
{
    return this->grad_tensors.index({id});
}

_ParamProxy _GPUParamCore::fetch_batch(int start, int end)
{
    _ParamProxy ret = _ParamProxy(false, this->all_tensors.index({torch::indexing::Slice(start, end)}), -1, -1, -1, this->bias);

    ret.grad = this->grad_tensors.index({torch::indexing::Slice(start, end)});

    return ret;
}

at::Tensor _CPUParamCore::fetch_single(int id)
{
    return this->all_tensors.index({id});
}
at::Tensor _CPUParamCore::fetch_single_grad(int id)
{
    return this->grad_.index({id});
}

_ParamProxy _CPUParamCore::fetch_batch(int view_base, int start, int end)
{
    return _ParamProxy(true, this->all_tensors.index({torch::indexing::Slice(start, end)}), this->p_id, view_base, view_base + end - start, this->bias);
}
_ParamProxy _CPUParamCore::fetch_batch_vari(int view_base, int start, int end)
{
    return _ParamProxy(true, this->variance.index({torch::indexing::Slice(start, end)}), this->p_id, view_base, view_base + end - start, this->bias);
}
_ParamProxy _CPUParamCore::fetch_batch_mome(int view_base, int start, int end)
{
    return _ParamProxy(true, this->momentum.index({torch::indexing::Slice(start, end)}), this->p_id, view_base, view_base + end - start, this->bias);
}
_ParamProxy _CPUParamCore::fetch_batch_grad(int view_base, int start, int end)
{
    return _ParamProxy(true, this->grad_.index({torch::indexing::Slice(start, end)}), this->p_id, view_base, view_base + end - start, this->bias);
}

at::Tensor _CPUParamCore::prefetch_to(int start, int batch_size, at::Tensor &gpu_param)
{
    at::Tensor tmp = this->all_tensors.index({torch::indexing::Slice(start, start + batch_size)});
    _cpufp32_to_gpufp32(tmp, gpu_param, tmp.numel());
    return tmp;
}
at::Tensor _CPUParamCore::prefetch_tensor(int start, int batch_size)
{
    return this->all_tensors.index({torch::indexing::Slice(start, start + batch_size)});
}
void _CPUParamCore::store_batch_grad(int start, int end, at::Tensor &grad)
{
    _gpufp32_to_cpufp32(grad, this->grad_.index({torch::indexing::Slice(start, end)}), grad.numel());
}

void _CPUParamCore::acc_store_batch_grad(int start, int end, at::Tensor &grad)
{
    if (this->lazy_zero)
    {
        _gpufp32_to_cpufp32(grad, this->grad_.index({torch::indexing::Slice(start, end)}), grad.numel());
    }
    else
    {
        _gpufp32_addto_cpufp32(grad, this->grad_.index({torch::indexing::Slice(start, end)}), grad.numel());
    }
}
void _CPUParamCore::store_batch_para(int start, int end, at::Tensor &para)
{
    _gpufp32_to_cpufp32(para, this->all_tensors.index({torch::indexing::Slice(start, end)}), para.numel());
}
void _CPUParamCore::store_batch_mome(int start, int end, at::Tensor &mome)
{
    _gpufp32_to_cpufp32(mome, this->momentum.index({torch::indexing::Slice(start, end)}), mome.numel());
}
void _CPUParamCore::store_batch_vari(int start, int end, at::Tensor &vari)
{
    _gpufp32_to_cpufp32(vari, this->variance.index({torch::indexing::Slice(start, end)}), vari.numel());
}

void _CPUParamCore::store_variance(int start, at::Tensor &variance)
{
    _gpufp32_to_cpufp32(variance, this->variance.index({torch::indexing::Slice(start, start + 1)}), variance.numel());
}

void _CPUParamCore::store_momentum(int start, at::Tensor &momentum)
{
    _gpufp32_to_cpufp32(momentum, this->momentum.index({torch::indexing::Slice(start, start + 1)}), momentum.numel());
}

void _CPUParamCore::store_weight(int start, at::Tensor &param)
{
    _gpufp32_to_cpufp32(param, this->all_tensors.index({torch::indexing::Slice(start, start + 1)}), param.numel());
}

void _CPUParamCore::fetch_variance(int start, at::Tensor &variance)
{
    _cpufp32_to_gpufp32(this->variance.index({torch::indexing::Slice(start, start + 1)}), variance, variance.numel());
}
void _CPUParamCore::fetch_momentum(int start, at::Tensor &momentum)
{
    _cpufp32_to_gpufp32(this->momentum.index({torch::indexing::Slice(start, start + 1)}), momentum, momentum.numel());
}
