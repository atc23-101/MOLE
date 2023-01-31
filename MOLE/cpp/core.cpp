#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <c10/cuda/CUDACachingAllocator.h>

#include "define.hpp"
#include "stream.hpp"
#include <cassert>

#define EXP_SETTING
#undef EXP_SETTING

/* this function calls c10 api and may not be safe */
void RemoveTensorInstanceStorage(at::Tensor &t)
{
    t.storage().unsafeGetStorageImpl()->reset();
}

void _MOLECore::dummy_fetch(int batch_size)
{
    if (batch_size == 0)
    {
        return;
    }
    std::vector<at::Tensor> params = _dummy_fetch(batch_size);
    {
        at::cuda::CUDAStreamGuard guard(tran_stream(this->device));
        get_CPUParamCore(0)->prefetch_to(0, batch_size, params[0]);
        get_CPUParamCore(1)->prefetch_to(0, batch_size, params[1]);
        get_CPUParamCore(2)->prefetch_to(0, batch_size, params[2]);
        get_CPUParamCore(3)->prefetch_to(0, batch_size, params[3]);
    }
}

void _MOLECore::prefetch_fwd(int batch_size)
{
    if (batch_size == 0)
    {
        return;
    }
    std::vector<at::Tensor> params = _prefetch(this->moe_id, batch_size);
    {
        at::cuda::CUDAStreamGuard guard(tran_stream(this->device));
        get_CPUParamCore(0)->prefetch_to(this->fwd_base, batch_size, params[0]);
        get_CPUParamCore(1)->prefetch_to(this->fwd_base, batch_size, params[1]);
        get_CPUParamCore(2)->prefetch_to(this->fwd_base, batch_size, params[2]);
        get_CPUParamCore(3)->prefetch_to(this->fwd_base, batch_size, params[3]);

        this->fwd_base += batch_size;
    }
}

void _MOLECore::prefetch_bwd(int batch_size)
{
    if (batch_size == 0)
    {
        return;
    }
    std::vector<at::Tensor> params = _prefetch(this->moe_id, batch_size);
    {
        at::cuda::CUDAStreamGuard guard(tran_stream(this->device));
        get_CPUParamCore(0)->prefetch_to(this->bwd_base, batch_size, params[0]);
        get_CPUParamCore(1)->prefetch_to(this->bwd_base, batch_size, params[1]);
        get_CPUParamCore(2)->prefetch_to(this->bwd_base, batch_size, params[2]);
        get_CPUParamCore(3)->prefetch_to(this->bwd_base, batch_size, params[3]);

        this->bwd_base += batch_size;
    }
}

void _BatchRecord::store_grad(at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2)
{

    // context
#ifndef TRAN_BACK_STREAM
    AT_CUDA_CHECK(cudaStreamWaitEvent(tran_stream(this->device), this->comp_event, 0));

    {
        at::cuda::CUDAStreamGuard guard(tran_stream(this->device));
        this->moe->store_batch_cpu_grad(this->batch_base, this->batch_size, g_w1, g_b1, g_w2, g_b2);
    }
#else
    AT_CUDA_CHECK(cudaStreamWaitEvent(tranback_stream(this->device), this->comp_event, 0));

    {
        at::cuda::CUDAStreamGuard guard(tranback_stream(this->device));
        this->moe->store_batch_cpu_grad(this->batch_base, this->batch_size, g_w1, g_b1, g_w2, g_b2);

        std::vector<at::Tensor> groups = {g_w1, g_w2, g_b1, g_b2};
        this->moe->add_tran_tmp(groups);
        cudaStreamAddCallback(tranback_stream(this->device).stream(), release_tranback_tensor, (void*)this->moe, 0);

    }


#endif
}

void _BatchRecord::release_para()
{
    this->params.clear();
}

void _BatchRecord::fetch_para()
{
    if (this->use_cpu_param)
    {
        if (this->use_pool)
        {
            std::vector<int> co = _get_UniMem()->get_batch_record(this->view_base, this->view_base + this->batch_size);
            for (auto c : co)
            {
                AT_CUDA_CHECK(cudaStreamWaitEvent(tran_stream(this->device), this->moe->fwdcore->records[c].comp_event, 0));
            }
            
        }
        this->params = this->moe->fetch_batch_cpu_param(this->view_base, this->batch_base, this->batch_size);
        {
            at::cuda::CUDAStreamGuard guard(tran_stream(this->device));
            this->params[0].fetch(this->use_pool, this->prefetched);
            this->params[1].fetch(this->use_pool, this->prefetched);
            this->params[2].fetch(this->use_pool, this->prefetched);
            this->params[3].fetch(this->use_pool, this->prefetched);

            this->para_event.record(tran_stream(this->device));
        }

        if (this->use_pool)
        {
            _get_UniMem()->set_batch_record(this->view_base, this->view_base + this->batch_size, this->recid);
        }
    }
    else
    {
        this->params = this->moe->fetch_batch_gpu_param(this->batch_base, this->batch_size);
    }
}
at::Tensor _BatchRecord::first_a2a(at::Tensor &dispatched_input, bool is_last)
{
    int global_expert_num = this->moe->total_experts;
    int world_size = this->world_size;
    int embed_dim = this->embed_dim;
    int local_expert_num = global_expert_num / world_size;

    if (world_size == 1)
    {
        return dispatched_input.index({torch::indexing::Slice(this->base_in_all, this->base_in_all + this->batch_size)});
    }
    std::vector<int64_t> size{world_size * this->batch_size * this->token_dim, this->embed_dim};
    at::Tensor rec = torch::empty(size, torch::dtype(dispatched_input.dtype()).device(dispatched_input.device()));
    std::vector<at::Tensor> ipt_lst;
    for (int c = this->base_in_all; c < global_expert_num; c += local_expert_num)
    {
        ipt_lst.push_back(dispatched_input.index({torch::indexing::Slice(c, c + this->batch_size)}).view({-1, embed_dim}));
    }
    std::vector<at::Tensor> rec_lst = rec.chunk(world_size, 0);

    {
        _comm_wait_default();
        at::cuda::CUDAStreamGuard guard(comm_stream(this->device));
        // comm stream id 3
        _MOLEAll2Allfp32(ipt_lst, rec_lst, this->device, world_size, 3, 0, "umoe", -1);

#ifdef EXP_SETTING
        if (is_last)
        {
            RemoveTensorInstanceStorage(dispatched_input);
        }
#endif
        this->comm_event.record(comm_stream(this->device));
    }
    return rec;
}
at::Tensor _BatchRecord::second_a2a(at::Tensor &expert_output)
{
    int global_expert_num = this->moe->total_experts;
    int world_size = this->world_size;
    int embed_dim = this->embed_dim;
    int local_expert_num = global_expert_num / world_size;

    if (world_size == 1)
    {
        return expert_output;
    }
    at::Tensor ret = at::empty_like(expert_output);

    // return ret;
    std::vector<at::Tensor> ipt_lst = expert_output.chunk(world_size, 0);
    std::vector<at::Tensor> rec_lst = ret.chunk(world_size, 0);

    AT_CUDA_CHECK(cudaStreamWaitEvent(comm_stream(this->device), this->comp_event, 0));

    {
        _comm_wait_default();
        at::cuda::CUDAStreamGuard guard(comm_stream(this->device));
        // comm stream id 3
        _MOLEAll2Allfp32(ipt_lst, rec_lst, this->device, world_size, 3, 0, "umoe", -1);
#ifdef EXP_SETTING
        expert_output.resize_(0);
#endif
    }

    return ret;
}

at::Tensor _BatchRecord::expert_fwd(at::Tensor &x, bool input_requires_grad)
{
    if (this->world_size > 1)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->comm_event, 0));
    }
    if (this->use_cpu_param)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->para_event, 0));
    }
    at::Tensor out;

    {
        if (this->use_cpu_param && this->world_size == 1)
        {
            _comp_wait_default();
        }
        at::cuda::CUDAStreamGuard guard(((this->world_size > 1 || this->use_cpu_param) ? comp_stream(this->device) : default_stream(this->device)));

        if (this->world_size > 1)
        {
            x = x.view({this->world_size, this->batch_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->batch_size, this->world_size * this->token_dim, this->embed_dim});
        }

        this->bffn_fwd(x, this->params[0].material(), this->params[1].material(), this->params[2].material(), this->params[3].material(), this->act, out, input_requires_grad);

        if (this->world_size > 1)
        {
            out = out.view({this->batch_size, this->world_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->world_size * this->batch_size * this->token_dim, this->embed_dim});
        }

        if (this->world_size > 1 || this->use_cpu_param)
        {
            this->comp_event.record(comp_stream(this->device));
        }
    }
    return out;
}

std::vector<at::Tensor> _BatchRecord::expert_bwd(at::Tensor &o_grad, bool input_requires_grad, bool is_last, at::Tensor &all_grad)
{
    std::vector<at::Tensor> ret;
    if (this->world_size > 1)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->comm_event, 0));
    }
    if (this->use_cpu_param)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->para_event, 0));
    }
    at::Tensor g_w1, g_b1, g_w2, g_b2;
    at::Tensor p0, p1, p2, p3;
    at::Tensor x_grad;

    {
        at::cuda::CUDAStreamGuard guard(((this->world_size > 1 || this->use_cpu_param) ? comp_stream(this->device) : default_stream(this->device)));

        p0 = this->params[0].bmaterial();
        p1 = this->params[1].bmaterial();
        p2 = this->params[2].bmaterial();
        p3 = this->params[3].bmaterial();

        if (!this->use_cpu_param)
        {
            g_w1 = this->params[0].grad_bmaterial();
            g_b1 = this->params[1].grad_bmaterial();
            g_w2 = this->params[2].grad_bmaterial();
            g_b2 = this->params[3].grad_bmaterial();
        }
        else
        {
            g_w1 = torch::empty_like(p0);
            g_b1 = torch::empty_like(p1);
            g_w2 = torch::empty_like(p2);
            g_b2 = torch::empty_like(p3);
        }

        if (this->world_size > 1)
        {
            o_grad = o_grad.view({this->world_size, this->batch_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->batch_size, this->world_size * this->token_dim, this->embed_dim});
        }


        if (this->acc && !this->use_cpu_param)
        {
            if (this->world_size == 1)
            {
                this->bffn_bwd_acc(o_grad, p0, p2, g_w1, g_b1, g_w2, g_b2, x_grad, this->act, this->x0, this->x2, input_requires_grad, is_last, all_grad);
            }
            else
            {
                this->bffn_bwd_acc(o_grad, p0, p2, g_w1, g_b1, g_w2, g_b2, x_grad, this->act, this->x0, this->x2, input_requires_grad, is_last, o_grad);
            }
        }
        else
        {
            if (this->world_size == 1)
            {
                this->bffn_bwd(o_grad, p0, p2, g_w1, g_b1, g_w2, g_b2, x_grad, this->act, this->x0, this->x2, input_requires_grad, is_last, all_grad);
            }
            else
            {
                this->bffn_bwd(o_grad, p0, p2, g_w1, g_b1, g_w2, g_b2, x_grad, this->act, this->x0, this->x2, input_requires_grad, is_last, o_grad);
            }
        }

        if (this->world_size > 1 && input_requires_grad)
        {
            x_grad = x_grad.view({this->batch_size, this->world_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->world_size * this->batch_size * this->token_dim, this->embed_dim});
        }

        if (this->world_size > 1 || this->use_cpu_param)
        {
            this->comp_event.record(comp_stream(this->device));
        }
    }

    ret.push_back(g_w1);
    ret.push_back(g_b1);
    ret.push_back(g_w2);
    ret.push_back(g_b2);

    ret.push_back(p0);
    ret.push_back(p1);
    ret.push_back(p2);
    ret.push_back(p3);

    ret.push_back(x_grad);

    return ret;
}

void _BatchRecord::bffn_fwd(at::Tensor batched_x, at::Tensor batched_w1, at::Tensor batched_b1, at::Tensor batched_w2, at::Tensor batched_b2, act_func &act, at::Tensor &out, bool input_requires_grad)
{
    if (this->mem_optim)
    {
        if (act.act == "gelu")
        {
            at::Tensor x1 = torch::baddbmm(batched_b1, batched_x, batched_w1.permute({0, 2, 1}));
            at::Tensor x2 = at::gelu(x1);

            out = torch::baddbmm(batched_b2, x2, batched_w2);
            this->extras.push_back(x1);
            this->x0 = batched_x;
        }
        else
        {
            at::Tensor x1 = torch::baddbmm(batched_b1, batched_x, batched_w1.permute({0, 2, 1}));
            at::Tensor x2 = act.forward(x1);

            out = torch::baddbmm(batched_b2, x2, batched_w2);

            this->x2 = x2;
            this->x0 = batched_x;
        }
    }
    else
    {
        at::Tensor x1 = torch::baddbmm(batched_b1, batched_x, batched_w1.permute({0, 2, 1}));
        at::Tensor x2 = act.forward(x1);

        out = torch::baddbmm(batched_b2, x2, batched_w2);

        this->x2 = x2;
        this->x0 = batched_x;
        }
    }
}

void _BatchRecord::bffn_bwd(at::Tensor &batched_out_grad, at::Tensor &batched_w1, at::Tensor &batched_w2, at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2, at::Tensor &x_grad, act_func &act, at::Tensor &x, at::Tensor &x2, bool input_requires_grad, bool is_last, at::Tensor &remover)
{

    if (this->mem_optim)
    {
        if (act.act == "gelu")
        {
            at::Tensor x3 = at::gelu(this->extras[0]);
            torch::matmul_out(g_w2, x3.permute({0, 2, 1}), batched_out_grad);
#ifdef USE_MOLE_GELU
            at::Tensor tmp = torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1}));
            inplace_gelu_bwd(tmp, this->extras[0]);
            at::Tensor x0_grad = tmp;
            this->extras.clear();
#else
            at::Tensor x0_grad = at::gelu_backward(torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1})), this->extras[0]);
            this->extras.clear();
#endif
            torch::sum_out(g_b2, batched_out_grad, {1});
#ifdef EXP_SETTING
            if (is_last)
            {
                RemoveTensorInstanceStorage(remover);
            }
#endif
            torch::matmul_out(g_w1, x0_grad.permute({0, 2, 1}), x);

            torch::sum_out(g_b1, x0_grad, {1});

            if (input_requires_grad)
            {
                x_grad = torch::matmul(x0_grad, batched_w1);
            }
        }
        else
        {
            torch::matmul_out(g_w2, x2.permute({0, 2, 1}), batched_out_grad);
            at::Tensor x0_grad = act.backward(torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1})));
            torch::sum_out(g_b2, batched_out_grad, {1});
#ifdef EXP_SETTING
            if (is_last)
            {
                RemoveTensorInstanceStorage(remover);
            }
#endif
            torch::matmul_out(g_w1, x0_grad.permute({0, 2, 1}), x);

            torch::sum_out(g_b1, x0_grad, {1});

            if (input_requires_grad)
            {
                x_grad = torch::matmul(x0_grad, batched_w1);
            }
        }
    }
    else
    {
        torch::matmul_out(g_w2, x2.permute({0, 2, 1}), batched_out_grad);
        at::Tensor x0_grad = act.backward(torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1})));
        torch::sum_out(g_b2, batched_out_grad, {1});
#ifdef EXP_SETTING
        if (is_last)
        {
            RemoveTensorInstanceStorage(remover);
        }
#endif
        torch::matmul_out(g_w1, x0_grad.permute({0, 2, 1}), x);

        torch::sum_out(g_b1, x0_grad, {1});

        if (input_requires_grad)
        {
            x_grad = torch::matmul(x0_grad, batched_w1);
        }
    }
}
void _BatchRecord::bffn_bwd_acc(at::Tensor &batched_out_grad, at::Tensor &batched_w1, at::Tensor &batched_w2, at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2, at::Tensor &x_grad, act_func &act, at::Tensor &x, at::Tensor &x2, bool input_requires_grad, bool is_last, at::Tensor &remover)
{
    if (this->mem_optim)
    {
        if (act.act == "gelu")
        {

            at::Tensor x3 = at::gelu(this->extras[0]);
            g_w2 += torch::matmul(x3.permute({0, 2, 1}), batched_out_grad);
#ifdef USE_MOLE_GELU
            at::Tensor tmp = torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1}));
            inplace_gelu_bwd(tmp, this->extras[0]);
            at::Tensor x0_grad = tmp;
            this->extras.clear();
#else
            at::Tensor x0_grad = at::gelu_backward(torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1})), this->extras[0]);
            this->extras.clear();
#endif
            g_b2 += torch::sum(batched_out_grad, {1});
#ifdef EXP_SETTING
            if (is_last)
            {
                RemoveTensorInstanceStorage(remover);
            }
#endif

            g_w1 += torch::matmul(x0_grad.permute({0, 2, 1}), x);

            g_b1 += torch::sum(x0_grad, {1});

            if (input_requires_grad)
            {
                x_grad = torch::matmul(x0_grad, batched_w1);
            }
        }
    }
    else
    {
        g_w2 += torch::matmul(x2.permute({0, 2, 1}), batched_out_grad);
        at::Tensor x0_grad = act.backward(torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1})));
        g_b2 += torch::sum(batched_out_grad, {1});
#ifdef EXP_SETTING
        if (is_last)
        {
            RemoveTensorInstanceStorage(remover);
        }
#endif

        g_w1 += torch::matmul(x0_grad.permute({0, 2, 1}), x);

        g_b1 += torch::sum(x0_grad, {1});

        if (input_requires_grad)
        {
            x_grad = torch::matmul(x0_grad, batched_w1);
        }
    }
}

at::Tensor _BatchRecord::a2a_before(at::Tensor &dispatched_input)
{
    int global_expert_num = this->moe->total_experts;
    int world_size = this->world_size;
    int embed_dim = this->embed_dim;
    int local_expert_num = global_expert_num / world_size;

    if (world_size == 1)
    {
        return dispatched_input.index({torch::indexing::Slice(this->base_in_all, this->base_in_all + this->batch_size)});
    }
    std::vector<int64_t> size{world_size * this->batch_size * this->token_dim, this->embed_dim};
    at::Tensor rec = torch::empty(size, torch::dtype(dispatched_input.dtype()).device(dispatched_input.device()));
    std::vector<at::Tensor> ipt_lst;
    for (int c = this->base_in_all; c < global_expert_num; c += local_expert_num)
    {
        ipt_lst.push_back(dispatched_input.index({torch::indexing::Slice(c, c + this->batch_size)}).view({-1, embed_dim}));
    }
    std::vector<at::Tensor> rec_lst = rec.chunk(world_size, 0);

    {
        _comm_wait_default();
        at::cuda::CUDAStreamGuard guard(comm_stream(this->device));
        // comm stream id 3
        _MOLEAll2Allfp32(ipt_lst, rec_lst, this->device, world_size, 3, 0, "umoe", -1);
        this->comm_event.record(comm_stream(this->device));
    }
    return rec;
}

at::Tensor _BatchRecord::a2a_after(at::Tensor &expert_output, at::Tensor &all_output)
{
    int global_expert_num = this->moe->total_experts;
    int world_size = this->world_size;
    int embed_dim = this->embed_dim;
    int local_expert_num = global_expert_num / world_size;

    if (world_size == 1)
    {
        return expert_output;
    }
    at::Tensor ret;
    if (all_output.numel() == 0)
    {
        ret = at::empty_like(expert_output);
    }
    else
    {
        ret = all_output;
    }
    std::vector<at::Tensor> ipt_lst = expert_output.chunk(world_size, 0);
    std::vector<at::Tensor> rec_lst;

    if (all_output.numel() != 0)
    {
        for (int c = this->base_in_all; c < global_expert_num; c += local_expert_num)
        {
            rec_lst.push_back(ret.index({torch::indexing::Slice(c, c + this->batch_size)}).view({-1, embed_dim}));
        }
    }
    else
    {
        rec_lst = ret.chunk(world_size, 0);
    }

    AT_CUDA_CHECK(cudaStreamWaitEvent(comm_stream(this->device), this->comp_event, 0));

    {
        at::cuda::CUDAStreamGuard guard(comm_stream(this->device));
        // comm stream id 3
        _MOLEAll2Allfp32(ipt_lst, rec_lst, this->device, world_size, 3, 0, "umoe", -1);
    }

    return ret;
}

at::Tensor make_second_a2a_send_buffer(at::Tensor input, at::Tensor all_output, int batch_base, int batch_size, int world_size, bool input_requires_grad)
{
    if (!input_requires_grad)
    {
        return all_output;
    }
    if (world_size > 1)
    {
        return torch::empty_like(input);
    }
    else
    {
        return all_output.index({torch::indexing::Slice(batch_base, batch_base + batch_size)});
    }
}

std::vector<at::Tensor> bffn_fwd(at::Tensor batched_x, at::Tensor batched_w1, at::Tensor batched_b1, at::Tensor batched_w2, at::Tensor batched_b2, act_func &act, at::Tensor &out)
{
    at::Tensor x1 = torch::baddbmm(batched_b1, batched_x, batched_w1.permute({0, 2, 1}));
    at::Tensor x2 = act.forward(x1);

    std::cerr << batched_b1.size(1) << " " << batched_x.size(1) << " " << batched_w1.size(1) << x1.size(1) << " " << x2.size(1) << std::endl;

    out = torch::baddbmm(batched_b2, x2, batched_w2);

    std::cerr << out.size(1) << " " << batched_b2.size(1) << " " << batched_w2.size(1) << std::endl;

    std::vector<at::Tensor> ret;
    ret.push_back(x2);
    ret.push_back(batched_x);

    return ret;
}

void bffn_bwd(at::Tensor &batched_out_grad, at::Tensor &batched_w1, at::Tensor &batched_w2, at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2, at::Tensor &x_grad, act_func &act, at::Tensor &x, at::Tensor &x2, bool input_requires_grad)
{
    at::Tensor x1_grad = torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1}));

    torch::matmul_out(g_w2, x2.permute({0, 2, 1}), batched_out_grad);

    torch::sum_out(g_b2, batched_out_grad, {1});

    at::Tensor x0_grad = act.backward(x1_grad);

    torch::matmul_out(g_w1, x0_grad.permute({0, 2, 1}), x);

    torch::sum_out(g_b1, x0_grad, {1});

    if (input_requires_grad)
    {
        if (x_grad.numel() == 0)
        {
            x_grad = torch::matmul(x0_grad, batched_w1);
        }
        else
        {
            torch::matmul_out(x_grad, x0_grad, batched_w1);
        }
    }
}

void bffn_bwd_acc(at::Tensor batched_out_grad, at::Tensor batched_w1, at::Tensor batched_w2, at::Tensor g_w1, at::Tensor g_b1, at::Tensor g_w2, at::Tensor g_b2, at::Tensor x_grad, act_func &act, at::Tensor x, at::Tensor x2, bool input_requires_grad)
{
    at::Tensor x1_grad = torch::matmul(batched_out_grad, batched_w2.permute({0, 2, 1}));
    g_w2 += torch::matmul(x2.permute({0, 2, 1}), batched_out_grad);

    g_b2 += torch::sum(batched_out_grad, {1});
    at::Tensor x0_grad = act.backward(x1_grad);
    if (input_requires_grad)
    {
        torch::matmul_out(x_grad, x0_grad, batched_w1);
    }
    g_w1 += torch::matmul(x0_grad.permute({0, 2, 1}), x);

    g_b1 += torch::sum(x0_grad, {1});
}

at::Tensor _BatchRecord::forward(at::Tensor x, at::Tensor all_output, bool input_requires_grad)
{
    if (this->world_size > 1)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->comm_event, 0));
    }
    if (this->use_cpu_param)
    {
        AT_CUDA_CHECK(cudaStreamWaitEvent(comp_stream(this->device), this->para_event, 0));
    }
    std::cerr << "FWD " << std::endl;
    at::Tensor out;
    std::vector<at::Tensor> intermedia;

    {
        if (this->use_cpu_param && this->world_size == 1)
        {
            _comp_wait_default();
        }
        at::cuda::CUDAStreamGuard guard(((this->world_size > 1 || this->use_cpu_param) ? comp_stream(this->device) : default_stream(this->device)));

        if (this->world_size > 1)
        {
            x = x.view({this->world_size, this->batch_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->batch_size, this->world_size * this->token_dim, this->embed_dim});
        }

        std::cerr << "ffn " << this->params.size() << std::endl;

        std::cerr << "ffn DONE " << intermedia[0].size(0) << " " << intermedia[1].size(0) << std::endl;

        if (this->world_size > 1)
        {
            out = out.view({this->batch_size, this->world_size, this->token_dim, this->embed_dim}).permute({1, 0, 2, 3}).contiguous().view({this->world_size * this->batch_size * this->token_dim, this->embed_dim});
        }

        if (this->world_size > 1 || this->use_cpu_param)
        {
            this->comp_event.record(comp_stream(this->device));
        }
    }

    this->x2 = intermedia[0];
    this->x0 = intermedia[1];

    return out;
}



void _MOLECore::set_CPUParamCore(int id, at::Tensor all_tensors, at::Tensor grad_, at::Tensor momentum, at::Tensor variance, bool bias, int p_id)
{
    if (id == 0)
    {
        this->FC1_W_CPU = new _CPUParamCore(all_tensors, grad_, momentum, variance, bias, p_id);
    }
    else if (id == 1)
    {
        this->FC1_B_CPU = new _CPUParamCore(all_tensors, grad_, momentum, variance, bias, p_id);
    }
    else if (id == 2)
    {
        this->FC2_W_CPU = new _CPUParamCore(all_tensors, grad_, momentum, variance, bias, p_id);
    }
    else if (id == 3)
    {
        this->FC2_B_CPU = new _CPUParamCore(all_tensors, grad_, momentum, variance, bias, p_id);
    }
    else
    {
        assert(false);
    }
}
void _MOLECore::set_GPUParamCore(int id, at::Tensor all_tensors, at::Tensor grad_tensors, bool bias)
{
    if (id == 0)
    {
        this->FC1_W_GPU = new _GPUParamCore(all_tensors, grad_tensors, bias);
    }
    else if (id == 1)
    {
        this->FC1_B_GPU = new _GPUParamCore(all_tensors, grad_tensors, bias);
    }
    else if (id == 2)
    {
        this->FC2_W_GPU = new _GPUParamCore(all_tensors, grad_tensors, bias);
    }
    else if (id == 3)
    {
        this->FC2_B_GPU = new _GPUParamCore(all_tensors, grad_tensors, bias);
    }
    else
    {
        assert(false);
    }
}
_CPUParamCore *_MOLECore::get_CPUParamCore(int i)
{
    if (i == 0)
    {
        return FC1_W_CPU;
    }
    else if (i == 1)
    {
        return FC1_B_CPU;
    }
    else if (i == 2)
    {
        return FC2_W_CPU;
    }
    else if (i == 3)
    {
        return FC2_B_CPU;
    }
    assert(false);
    return nullptr;
}
_GPUParamCore *_MOLECore::get_GPUParamCore(int i)
{
    if (i == 0)
    {
        return FC1_W_GPU;
    }
    else if (i == 1)
    {
        return FC1_B_GPU;
    }
    else if (i == 2)
    {
        return FC2_W_GPU;
    }
    else if (i == 3)
    {
        return FC2_B_GPU;
    }
    assert(false);
    return nullptr;
}

void _MOLECore::store_batch_cpu_grad(int base, int batch_size, at::Tensor &fc1_w_grad, at::Tensor &fc1_b_grad, at::Tensor &fc2_w_grad, at::Tensor &fc2_b_grad)
{

    if (this->acc_step > 1)
    {
        this->FC1_W_CPU->acc_store_batch_grad(base, base + batch_size, fc1_w_grad);
        this->FC1_B_CPU->acc_store_batch_grad(base, base + batch_size, fc1_b_grad);
        this->FC2_W_CPU->acc_store_batch_grad(base, base + batch_size, fc2_w_grad);
        this->FC2_B_CPU->acc_store_batch_grad(base, base + batch_size, fc2_b_grad);
    }
    else
    {
        this->FC1_W_CPU->store_batch_grad(base, base + batch_size, fc1_w_grad);
        this->FC1_B_CPU->store_batch_grad(base, base + batch_size, fc1_b_grad);
        this->FC2_W_CPU->store_batch_grad(base, base + batch_size, fc2_w_grad);
        this->FC2_B_CPU->store_batch_grad(base, base + batch_size, fc2_b_grad);
    }
}
void _MOLECore::store_batch_cpu_para(int base, int batch_size, at::Tensor &fc1_w_para, at::Tensor &fc1_b_para, at::Tensor &fc2_w_para, at::Tensor &fc2_b_para)
{
    this->FC1_W_CPU->store_batch_para(base, base + batch_size, fc1_w_para);
    this->FC1_B_CPU->store_batch_para(base, base + batch_size, fc1_b_para);
    this->FC2_W_CPU->store_batch_para(base, base + batch_size, fc2_w_para);
    this->FC2_B_CPU->store_batch_para(base, base + batch_size, fc2_b_para);
}

void _MOLECore::store_batch_cpu_mome(int base, int batch_size, at::Tensor &fc1_w_mome, at::Tensor &fc1_b_mome, at::Tensor &fc2_w_mome, at::Tensor &fc2_b_mome)
{
    this->FC1_W_CPU->store_batch_mome(base, base + batch_size, fc1_w_mome);
    this->FC1_B_CPU->store_batch_mome(base, base + batch_size, fc1_b_mome);
    this->FC2_W_CPU->store_batch_mome(base, base + batch_size, fc2_w_mome);
    this->FC2_B_CPU->store_batch_mome(base, base + batch_size, fc2_b_mome);
}
void _MOLECore::store_batch_cpu_vari(int base, int batch_size, at::Tensor &fc1_w_vari, at::Tensor &fc1_b_vari, at::Tensor &fc2_w_vari, at::Tensor &fc2_b_vari)
{
    this->FC1_W_CPU->store_batch_vari(base, base + batch_size, fc1_w_vari);
    this->FC1_B_CPU->store_batch_vari(base, base + batch_size, fc1_b_vari);
    this->FC2_W_CPU->store_batch_vari(base, base + batch_size, fc2_w_vari);
    this->FC2_B_CPU->store_batch_vari(base, base + batch_size, fc2_b_vari);
}

void _MOLECore::instant_adam(_BatchRecord *record, int base, int batch_size, int view_base, at::Tensor weight_fc1_w, at::Tensor weight_fc1_b, at::Tensor weight_fc2_w, at::Tensor weight_fc2_b, at::Tensor fc1_w_grad, at::Tensor fc1_b_grad, at::Tensor fc2_w_grad, at::Tensor fc2_b_grad, int hybrid_adam_count)
{
    if (base >= hybrid_adam_count)
    {
        return;
    }
    int new_batch_size = std::min(batch_size, hybrid_adam_count - base);
    if (new_batch_size != batch_size)
    {
        weight_fc1_w = weight_fc1_w.index({torch::indexing::Slice(0, new_batch_size)});
        weight_fc1_b = weight_fc1_b.index({torch::indexing::Slice(0, new_batch_size)});
        weight_fc2_w = weight_fc1_w.index({torch::indexing::Slice(0, new_batch_size)});
        weight_fc2_b = weight_fc2_b.index({torch::indexing::Slice(0, new_batch_size)});

        fc1_w_grad = fc1_w_grad.index({torch::indexing::Slice(0, new_batch_size)});
        fc1_b_grad = fc1_b_grad.index({torch::indexing::Slice(0, new_batch_size)});
        fc2_w_grad = fc2_w_grad.index({torch::indexing::Slice(0, new_batch_size)});
        fc2_b_grad = fc2_b_grad.index({torch::indexing::Slice(0, new_batch_size)});
    }
    std::vector<_ParamProxy> m_ = this->fetch_batch_cpu_mome(view_base + batch_size, base, new_batch_size);

    // mome_use_pool notion that this function is not finished
    std::vector<_ParamProxy> v_ = this->fetch_batch_cpu_vari(view_base + batch_size + batch_size * this->mome_use_pool, base, new_batch_size);

    AT_CUDA_CHECK(cudaStreamWaitEvent(tran_stream(this->device), record->comp_event, 0));

    at::Tensor m_fc1_w, m_fc1_b, m_fc2_w, m_fc2_b;
    at::Tensor v_fc1_w, v_fc1_b, v_fc2_w, v_fc2_b;

    std::vector<at::Tensor> fc1_w, fc1_b, fc2_w, fc2_b;
    {
        at::cuda::CUDAStreamGuard guard(tran_stream(this->device));

        m_[0].fetch(this->mome_use_pool, 0);
        m_[1].fetch(this->mome_use_pool, 0);
        m_[2].fetch(this->mome_use_pool, 0);
        m_[3].fetch(this->mome_use_pool, 0);

        v_[0].fetch(this->vari_use_pool, 0);
        v_[1].fetch(this->vari_use_pool, 0);
        v_[2].fetch(this->vari_use_pool, 0);
        v_[3].fetch(this->vari_use_pool, 0);

        m_fc1_w = m_[0].bmaterial();
        m_fc1_b = m_[1].bmaterial();
        m_fc2_w = m_[2].bmaterial();
        m_fc2_b = m_[3].bmaterial();

        v_fc1_w = v_[0].bmaterial();
        v_fc1_b = v_[1].bmaterial();
        v_fc2_w = v_[2].bmaterial();
        v_fc2_b = v_[3].bmaterial();

        if (this->acc_step > 1)
        {
            std::vector<_ParamProxy> cpu_grads = this->fetch_batch_cpu_grad(view_base + batch_size + batch_size * (this->vari_use_pool + this->mome_use_pool), base, new_batch_size);

            cpu_grads[0].fetch(this->grad_use_pool, 0);
            cpu_grads[1].fetch(this->grad_use_pool, 0);
            cpu_grads[2].fetch(this->grad_use_pool, 0);
            cpu_grads[3].fetch(this->grad_use_pool, 0);

            fc1_w_grad += cpu_grads[0].bmaterial();
            fc1_b_grad += cpu_grads[1].bmaterial();
            fc2_w_grad += cpu_grads[2].bmaterial();
            fc2_b_grad += cpu_grads[3].bmaterial();
        }
        fc1_w = this->adam(weight_fc1_w, m_fc1_w, v_fc1_w, fc1_w_grad);
        fc1_b = this->adam(weight_fc1_b, m_fc1_b, v_fc1_b, fc1_b_grad);
        fc2_w = this->adam(weight_fc2_w, m_fc2_w, v_fc2_w, fc2_w_grad);
        fc2_b = this->adam(weight_fc2_b, m_fc2_b, v_fc2_b, fc2_b_grad);

#ifndef TRAN_BACK_STREAM
        this->store_batch_cpu_para(base, new_batch_size, fc1_w[0], fc1_b[0], fc2_w[0], fc2_b[0]);
        this->store_batch_cpu_mome(base, new_batch_size, fc1_w[1], fc1_b[1], fc2_w[1], fc2_b[1]);
        this->store_batch_cpu_vari(base, new_batch_size, fc1_w[2], fc1_b[2], fc2_w[2], fc2_b[2]);
#endif
        //        _synchronize_tran();
    }

#ifdef TRAN_BACK_STREAM
    _tran_back_wait_tran();
    {
        at::cuda::CUDAStreamGuard guard(tranback_stream(this->device));
        this->store_batch_cpu_para(base, new_batch_size, fc1_w[0], fc1_b[0], fc2_w[0], fc2_b[0]);
        this->store_batch_cpu_mome(base, new_batch_size, fc1_w[1], fc1_b[1], fc2_w[1], fc2_b[1]);
        this->store_batch_cpu_vari(base, new_batch_size, fc1_w[2], fc1_b[2], fc2_w[2], fc2_b[2]);

        std::vector<at::Tensor> groups = {fc1_w[0], fc1_b[0], fc2_w[0], fc2_b[0], fc1_w[1], fc1_b[1], fc2_w[1], fc2_b[1], fc1_w[2], fc1_b[2], fc2_w[2], fc2_b[2]};
        this->add_tran_tmp(groups);
        cudaStreamAddCallback(tranback_stream(this->device).stream(), release_tranback_tensor, (void*)this, 0);
    }
#endif
    //_default_wait_tran();
    //_synchronize_tran();
}

std::vector<_ParamProxy> _MOLECore::fetch_batch_gpu_param(int batch_base, int batch_size)
{
    std::vector<_ParamProxy> ret;
    ret.push_back(this->FC1_W_GPU->fetch_batch(batch_base, batch_base + batch_size));
    ret.push_back(this->FC1_B_GPU->fetch_batch(batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_W_GPU->fetch_batch(batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_B_GPU->fetch_batch(batch_base, batch_base + batch_size));
    return ret;
}
std::vector<_ParamProxy> _MOLECore::fetch_batch_cpu_param(int view_base, int batch_base, int batch_size)
{
    std::vector<_ParamProxy> ret;
    ret.push_back(this->FC1_W_CPU->fetch_batch(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC1_B_CPU->fetch_batch(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_W_CPU->fetch_batch(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_B_CPU->fetch_batch(view_base, batch_base, batch_base + batch_size));
    return ret;
}
std::vector<_ParamProxy> _MOLECore::fetch_batch_cpu_vari(int view_base, int batch_base, int batch_size)
{
    std::vector<_ParamProxy> ret;
    ret.push_back(this->FC1_W_CPU->fetch_batch_vari(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC1_B_CPU->fetch_batch_vari(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_W_CPU->fetch_batch_vari(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_B_CPU->fetch_batch_vari(view_base, batch_base, batch_base + batch_size));
    return ret;
}
std::vector<_ParamProxy> _MOLECore::fetch_batch_cpu_mome(int view_base, int batch_base, int batch_size)
{
    std::vector<_ParamProxy> ret;
    ret.push_back(this->FC1_W_CPU->fetch_batch_mome(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC1_B_CPU->fetch_batch_mome(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_W_CPU->fetch_batch_mome(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_B_CPU->fetch_batch_mome(view_base, batch_base, batch_base + batch_size));
    return ret;
}
std::vector<_ParamProxy> _MOLECore::fetch_batch_cpu_grad(int view_base, int batch_base, int batch_size)
{
    std::vector<_ParamProxy> ret;
    ret.push_back(this->FC1_W_CPU->fetch_batch_grad(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC1_B_CPU->fetch_batch_grad(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_W_CPU->fetch_batch_grad(view_base, batch_base, batch_base + batch_size));
    ret.push_back(this->FC2_B_CPU->fetch_batch_grad(view_base, batch_base, batch_base + batch_size));
    return ret;
}

at::Tensor _BatchFusedFWD::forward(at::Tensor &dispatched_input, _MOLECore *moe, bool requires_grad)
{
    int token_dim = dispatched_input.size(1);
    int embed_dim = dispatched_input.size(2);
    auto sizes = dispatched_input.sizes();

    this->records.clear();
    this->moe = moe;
    this->input_requires_grad = requires_grad;

    _get_UniMem()->clear_batch_record();

    std::vector<std::vector<int>> group_launch;
    std::vector<int> cur_group;
    std::vector<int64_t> split_sizes;

    at::Tensor all_output; // = torch::empty_like(dispatched_input);

    for (int i = 0; i < moe->s_experts; i += moe->ebatch_size)
    {
        int size = std::min(moe->ebatch_size, moe->s_experts - i);
        cur_group.push_back(this->records.size());
        this->records.emplace_back(
            moe->device, i, i, size, false, moe, token_dim, embed_dim, this->records.size(), 0, 0, moe->world_size, moe->act_name);
    }

    int prefetched = _get_UniMem()->fetch_top_prefetched_batch(moe->moe_id);
    assert(prefetched == this->moe->fwd_base);
    this->moe->fwd_base = 0;
    int view_pool_size = _get_UniMem()->view_pool_size();
    int factor = (moe->adv_adam ? (moe->grad_use_pool + moe->mome_use_pool + moe->vari_use_pool + 1) : 1);
    int d_batch_size = std::min(moe->ebatch_size, ((moe->use_pool == false) ? 10000 : view_pool_size));
    int view_base = 0;

    for (int i = 0; i < moe->d_experts; i += d_batch_size)
    {
        int size = std::min(d_batch_size, moe->d_experts - i);
        if (moe->use_pool && view_pool_size - view_base < size)
        {
            view_base = 0;
            group_launch.push_back(cur_group);
            cur_group.clear();
        }
        cur_group.push_back(this->records.size());
        int cur_prefetched;
        if (prefetched > i)
        {
            cur_prefetched = std::min(size, prefetched - i);
        }
        else
        {
            cur_prefetched = 0;
        }
        this->records.emplace_back(
            moe->device, i, i + moe->s_experts, size, true, moe, token_dim, embed_dim, this->records.size(), cur_prefetched, view_base, moe->world_size, moe->act_name);
        view_base += size;
    }
    if (cur_group.size() > 0)
    {
        group_launch.push_back(cur_group);
    }
    std::vector<at::Tensor> a2a_o, cmp_o, a2a2_o;

    std::vector<at::Tensor> pre_a2a;

    for (int i0 = 0; i0 < group_launch.size(); i0++)
    {
        auto &g = group_launch[i0];
        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];
            _BatchRecord *record = &this->records[ri];
            a2a_o.push_back(record->first_a2a(dispatched_input, (i0 == group_launch.size() - 1) && (i1 == g.size() - 1)));
            record->fetch_para();
        }

        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];

            _BatchRecord *record = &this->records[ri];

            cmp_o.push_back(record->expert_fwd(a2a_o[ri], requires_grad));
        }

        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];

            _BatchRecord *record = &this->records[ri];

            a2a2_o.push_back(record->second_a2a(cmp_o[ri]));
        }
    }

    if (moe->world_size > 1)
    {
        _default_wait_comm();
    }
    else if (moe->d_experts > 0)
    {
        _default_wait_comp();
    }

    at::Tensor ret;

    if (a2a2_o.size() > 1)
    {
        ret = torch::empty(sizes, torch::dtype(dispatched_input.dtype()).device(dispatched_input.device()));

        for (int i = 0; i < a2a2_o.size(); i++)
        {
            _BatchRecord *record = &this->records[i];
            int global_expert_num = record->moe->total_experts;
            int world_size = record->world_size;
            int embed_dim = record->embed_dim;
            int local_expert_num = global_expert_num / world_size;
            std::vector<at::Tensor> ao = a2a2_o[i].chunk(world_size, 0);

            int cc = 0;
            for (int c = record->base_in_all; c < global_expert_num; c += local_expert_num)
            {
                auto sz0 = ret.index({torch::indexing::Slice(c, c + record->batch_size)});
                auto sz1 = ao[cc].sizes();

                sz0.copy_(ao[cc].view(sz0.sizes()));
                cc += 1;
            }
        }
    }
    else
    {
        ret = a2a2_o[0].view(sizes);
    }

    for (auto &r : this->records)
    {
        r.release_para();
    }
    return ret;
}

at::Tensor _BatchFusedFWD::backward(at::Tensor &all_output_grad)
{
    _get_UniMem()->clear_batch_record();
    _MOLECore *moe = this->moe;

    auto sizes = all_output_grad.sizes();
    std::vector<at::Tensor> a2a_o, cmp_o, a2a2_o;
    at::Tensor dummy_tensor;

    std::vector<std::vector<int>> group_launch;
    std::vector<int> cur_group;
    int s_counter = 0;

    for (int i = 0; i < moe->s_experts; i += moe->ebatch_size)
    {
        cur_group.push_back(s_counter);
        s_counter += 1;
    }

    int prefetched = _get_UniMem()->fetch_top_prefetched_batch(moe->moe_id);
    assert(prefetched == this->moe->bwd_base);
    this->moe->bwd_base = 0;

    int view_pool_size = _get_UniMem()->view_pool_size();

    int d_batch_size = std::min(moe->ebatch_size, ((moe->use_pool == false) ? 10000 : view_pool_size));
    int view_base = 0;
    int d_counter = 0;
    int factor = (moe->adv_adam ? (moe->grad_use_pool + moe->mome_use_pool + moe->vari_use_pool + 1) : 1);

    std::vector<std::vector<at::Tensor>> bwds;
    for (int i = 0; i < moe->d_experts; i += d_batch_size)
    {
        int size = std::min(d_batch_size, moe->d_experts - i) * factor;
        int cur_prefetched = 0;
        if (prefetched > i)
        {
            cur_prefetched = std::min(size, prefetched - i);
        }
        else
        {
            cur_prefetched = 0;
        }

        if (moe->use_pool && view_pool_size - view_base < size)
        {
            view_base = 0;
            group_launch.push_back(cur_group);
            cur_group.clear();
        }

        cur_group.push_back(d_counter + s_counter);
        this->records[d_counter + s_counter].view_base = view_base;
        this->records[d_counter + s_counter].prefetched = cur_prefetched;
        view_base += size;
        d_counter += 1;
    }

    if (cur_group.size() > 0)
    {
        group_launch.push_back(cur_group);
    }

    for (int i0 = 0; i0 < group_launch.size(); i0++)
    {
        auto &g = group_launch[i0];
        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];
            _BatchRecord *record = &this->records[ri];
            a2a_o.push_back(record->first_a2a(all_output_grad, (i0 == group_launch.size() - 1) && (i1 == g.size() - 1)));
            record->fetch_para();
        }

        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];
            _BatchRecord *record = &this->records[ri];

            bwds.push_back(record->expert_bwd(a2a_o[ri], this->input_requires_grad, (i0 == group_launch.size() - 1) && (i1 == g.size() - 1), all_output_grad));

            cmp_o.push_back(bwds.back()[8]);

            if (record->use_cpu_param)
            {
                if (moe->adv_adam && ((moe->step + 1) % moe->acc_step == 0))
                {
                    moe->instant_adam(record, record->batch_base, record->batch_size, record->view_base, bwds.back()[4], bwds.back()[5], bwds.back()[6], bwds.back()[7],
                                      bwds.back()[0], bwds.back()[1], bwds.back()[2], bwds.back()[3], moe->hybrid_adam_count);
                }
                else if (!this->moe->adv_adam || (this->moe->acc_step > 1 && (this->moe->step + 1 % this->moe->acc_step != 0)))
                {
                    record->store_grad(bwds.back()[0], bwds.back()[1], bwds.back()[2], bwds.back()[3]);
                }
            }
        }

        for (int i1 = 0; i1 < g.size(); i1++)
        {
            auto &ri = g[i1];
            _BatchRecord *record = &this->records[ri];
            if (this->input_requires_grad)
            {
                a2a2_o.push_back(record->second_a2a(cmp_o[ri]));
            }
        }
    }

    if (moe->world_size > 1)
    {
        _default_wait_comm();
    }
    else if (moe->d_experts > 0)
    {
        _default_wait_comp();
    }
    at::Tensor ret;

    /*
    *******************
    ****************** remember to unset lazy zero outsidethis
    */

    if (this->input_requires_grad)
    {

        if (a2a2_o.size() > 1)
        {
            ret = torch::empty(sizes, torch::dtype(all_output_grad.dtype()).device(all_output_grad.device()));

            for (int i = 0; i < a2a2_o.size(); i++)
            {

                _BatchRecord *record = &this->records[i];
                int global_expert_num = record->moe->total_experts;
                int world_size = record->world_size;
                int embed_dim = record->embed_dim;
                int local_expert_num = global_expert_num / world_size;
                std::vector<at::Tensor> ao = a2a2_o[i].chunk(world_size, 0);
                int cc = 0;
                for (int c = record->base_in_all; c < global_expert_num; c += local_expert_num)
                {
                    auto rix = ret.index({torch::indexing::Slice(c, c + record->batch_size)});
                    rix.copy_(ao[cc].view(rix.sizes()));
                    cc += 1;
                }
            }
        }
        else
        {
            ret = a2a2_o[0].view(sizes);
        }
    }

    _synchronize_tran();

    for (auto &r : this->records)
    {
        r.release_para();
    }

    moe->step += 1;
    moe->unset_lazy_zero();

    // end
    this->records.clear();
    return ret;
}

std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<int>> _fast_assign_prefetch(int P, std::vector<int> F, std::vector<int> E)
{
    int len = E.size();
    std::vector<std::vector<std::vector<int>>> GPscheme(len);
    std::vector<int> LPscheme(len);
    std::vector<std::vector<int>> GPstk;
    int P_cur = P;
    for (int i = 0; i < len; i++)
    {
        int fetchable = std::min(P_cur, F[i]);
        if (fetchable <= E[i])
        {
            int need = E[i] - fetchable;
            LPscheme[i] = fetchable;
            while (GPstk.size() > 0 && need > 0)
            {
                int top = GPstk.back()[1];
                int top_id = GPstk.back()[0];
                if (top > need)
                {
                    GPstk.back()[1] = top - need;
                    GPscheme[top_id].emplace_back(std::vector<int>({i, need}));
                    P_cur += need;
                    need = 0;
                }
                else
                {
                    GPscheme[top_id].emplace_back(std::vector<int>({i, top}));
                    GPstk.pop_back();
                    need -= top;
                    P_cur += top;
                }
            }
        }
        else
        {
            int newGP = fetchable - E[i];
            P_cur -= newGP;
            GPstk.emplace_back(std::vector<int>({i, newGP}));
            LPscheme[i] = E[i];
        }
    }
    return std::tuple<std::vector<std::vector<std::vector<int>>>, std::vector<int>>({GPscheme, LPscheme});
}

PYBIND11_MODULE(core, m)
{
    /* _BatchFusedFWD */
    pybind11::class_<_BatchFusedFWD>(m, "BatchFusedFWD")
        .def("forward", &_BatchFusedFWD::forward)
        .def("backward", &_BatchFusedFWD::backward);

    /* _MOLECore */
    pybind11::class_<_MOLECore>(m, "MOLECore")
        .def(pybind11::init<bool, bool, bool, bool, int, int, int, int, std::string, bool, int, int, int, int,
                            int>())
        .def("set_CPUParamCore", &_MOLECore::set_CPUParamCore)
        .def("set_GPUParamCore", &_MOLECore::set_GPUParamCore)
        .def("GPUParamCore_get_all_tensors", &_MOLECore::_GPUParamCore_get_all_tensors)
        .def("GPUParamCore_get_grad_tensors", &_MOLECore::_GPUParamCore_get_grad_tensors)
        .def("GPUParamCore_fetch_single", &_MOLECore::_GPUParamCore_fetch_single)
        .def("GPUParamCore_fetch_single_grad", &_MOLECore::_GPUParamCore_fetch_single_grad)
        .def("GPUParamCore_fetch_batch", &_MOLECore::_GPUParamCore_fetch_batch)
        .def("CPUParamCore_set_lazy_zero", &_MOLECore::_CPUParamCore_set_lazy_zero)
        .def("CPUParamCore_unset_lazy_zero", &_MOLECore::_CPUParamCore_unset_lazy_zero)
        .def("CPUParamCore_fetch_single", &_MOLECore::_CPUParamCore_fetch_single)
        .def("CPUParamCore_fetch_single_grad", &_MOLECore::_CPUParamCore_fetch_single_grad)
        .def("CPUParamCore_prefetch_to", &_MOLECore::_CPUParamCore_prefetch_to)
        .def("CPUParamCore_prefetch_tensor", &_MOLECore::_CPUParamCore_prefetch_tensor)
        .def("CPUParamCore_store_batch_grad", &_MOLECore::_CPUParamCore_store_batch_grad)
        .def("CPUParamCore_store_batch_para", &_MOLECore::_CPUParamCore_store_batch_para)
        .def("CPUParamCore_store_batch_mome", &_MOLECore::_CPUParamCore_store_batch_mome)
        .def("CPUParamCore_store_batch_vari", &_MOLECore::_CPUParamCore_store_batch_vari)
        .def("CPUParamCore_acc_store_batch_grad", &_MOLECore::_CPUParamCore_acc_store_batch_grad)
        .def("CPUParamCore_fetch_batch", &_MOLECore::_CPUParamCore_fetch_batch)
        .def("CPUParamCore_fetch_batch_vari", &_MOLECore::_CPUParamCore_fetch_batch_vari)
        .def("CPUParamCore_fetch_batch_mome", &_MOLECore::_CPUParamCore_fetch_batch_mome)
        .def("CPUParamCore_fetch_batch_grad", &_MOLECore::_CPUParamCore_fetch_batch_grad)

        .def("forward", &_MOLECore::forward)
        .def("backward", &_MOLECore::backward)
        .def("set_adam", &_MOLECore::set_adam)
        .def("set_lazy_zero", &_MOLECore::set_lazy_zero)

        .def("prefetch_fwd", &_MOLECore::prefetch_fwd)
        .def("prefetch_bwd", &_MOLECore::prefetch_bwd)
        .def("dummy_fetch", &_MOLECore::dummy_fetch)
        .def("clear_tran_tmp", &_MOLECore::clear_tran_tmp);

    /* BatchRecord */
    pybind11::class_<_BatchRecord>(m, "BatchRecord")
        .def(pybind11::init<int, int, int, int, bool, _MOLECore *, int, int, int, int, int, int, std::string>())
        .def("store_grad", &_BatchRecord::store_grad);

    /* communication */
    m.def("ensure_nccl", &_ensure_nccl);
    m.def("MOLEAll2Allfp32", &_MOLEAll2Allfp32);

    /* stream */
    m.def("disable_multi_stream", &_disable_multi_stream);
    m.def("get_stream_cdata", &_get_stream_cdata);
    m.def("default_wait_comp", &_default_wait_comp);
    m.def("default_wait_comm", &_default_wait_comm);
    m.def("default_wait_tran", &_default_wait_tran);
    m.def("synchronize_tran", &_synchronize_tran);

    /* ParamProxy */
    pybind11::class_<_ParamProxy>(m, "ParamProxy")
        .def(pybind11::init<bool, at::Tensor, int, int, int, bool>())
        .def("material", &_ParamProxy::material)
        .def("bmaterial", &_ParamProxy::bmaterial)
        .def("fetch", &_ParamProxy::fetch)
        .def("get_param", &_ParamProxy::_get_param);

    /* UniMem */
    pybind11::class_<_UniMem>(m, "UniMem")
        .def(pybind11::init<int, int, const std::string &, std::vector<std::vector<int64_t>>>())
        .def("getn", &_UniMem::get);

    m.def("fetch_batch", &_fetch_batch);
    m.def("clear_batch_record", &_clear_batch_record);
    m.def("create_UniMem", &_create_UniMem);
    m.def("clear_virtual_mem", &_clear_virtual_mem);
    m.def("pool_prefetch", &_prefetch);
    m.def("fetch_top_prefetched_batch", &_fetch_top_prefetched_batch);
    m.def("view_pool_size", &_view_pool_size);
    m.def("set_batch_record", &_set_batch_record);
    m.def("get_batch_record", &_get_batch_record);

    /* fast assign prefetch scheme */
    m.def("fast_assign_prefetch", &_fast_assign_prefetch);

    /* GPUParamCore */
    pybind11::class_<_GPUParamCore>(m, "GPUParamCore")
        .def(pybind11::init<at::Tensor, at::Tensor, bool>())
        .def("fetch_single", &_GPUParamCore::fetch_single)
        .def("fetch_single_grad", &_GPUParamCore::fetch_single_grad)
        .def("fetch_batch", &_GPUParamCore::fetch_batch)
        .def("get_all_tensors", &_GPUParamCore::_get_all_tensors)
        .def("get_grad_tensors", &_GPUParamCore::_get_grad_tensors);

    /* CPUParamCore */
    pybind11::class_<_CPUParamCore>(m, "CPUParamCore")
        .def(pybind11::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool, int>())
        .def("get_all_tensors", &_CPUParamCore::get_all_tensors)
        .def("fetch_single", &_CPUParamCore::fetch_single)
        .def("fetch_single_grad", &_CPUParamCore::fetch_single_grad)
        .def("fetch_batch", &_CPUParamCore::fetch_batch)
        .def("fetch_batch_vari", &_CPUParamCore::fetch_batch_vari)
        .def("fetch_batch_mome", &_CPUParamCore::fetch_batch_mome)
        .def("fetch_batch_grad", &_CPUParamCore::fetch_batch_grad)
        .def("prefetch_to", &_CPUParamCore::prefetch_to)

        .def("acc_store_batch_grad", &_CPUParamCore::acc_store_batch_grad)
        .def("store_batch_grad", &_CPUParamCore::store_batch_grad)
        .def("store_batch_para", &_CPUParamCore::store_batch_para)
        .def("store_batch_mome", &_CPUParamCore::store_batch_mome)
        .def("store_batch_vari", &_CPUParamCore::store_batch_vari)

        .def("store_variance", &_CPUParamCore::store_variance)
        .def("store_momentum", &_CPUParamCore::store_momentum)
        .def("store_weight", &_CPUParamCore::store_weight)
        .def("fetch_variance", &_CPUParamCore::fetch_variance)
        .def("fetch_momentum", &_CPUParamCore::fetch_momentum)

        .def("set_lazy_zero", &_CPUParamCore::set_lazy_zero)
        .def("unset_lazy_zero", &_CPUParamCore::unset_lazy_zero);
}
