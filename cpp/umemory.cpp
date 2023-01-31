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

inline size_t key(int i, int j) { return (size_t)i << 32 | (unsigned int)j; }

void _UniMem::clear_batch_record()
{
    this->batch_records.clear();
}

std::vector<int> _UniMem::get_batch_record(int start, int end)
{
    std::vector<int> ret;
    for (auto &v : this->batch_records)
    {
        std::pair<int, int> key = v.first;
        int value = v.second;
        if ((start >= key.first && start < key.second) || (end > key.first && end <= key.second))
        {
            ret.push_back(value);
        }
    }
    return ret;
}

void _UniMem::set_batch_record(int start, int end, int batch_record)
{
    /* if totally cover some other record, delete it*/
    std::vector<std::pair<int, int>> remove;
    for (auto &v : this->batch_records)
    {
        std::pair<int, int> key = v.first;
        if (start <= key.first && end >= key.second)
        {
            remove.push_back(key);
        }
    }
    for (auto &rm : remove)
    {
        this->batch_records.erase(rm);
    }

    this->batch_records[std::make_pair(start, end)] = batch_record;
}

std::vector<_UniMem *> UniMemInst;
_UniMem *_get_UniMem()
{
    return UniMemInst[0];
}
void _clear_virtual_mem()
{
    UniMemInst[0]->clear_virtual_mem();
}
void _clear_batch_record()
{
    UniMemInst[0]->clear_batch_record();
}

std::vector<int> _get_batch_record(int start, int end)
{
    return UniMemInst[0]->get_batch_record(start, end);
}
void _set_batch_record(int start, int end, int batch_record)
{
    UniMemInst[0]->set_batch_record(start, end, batch_record);
}
_UniMem _create_UniMem(int device, int pool_size, const std::string &dtype, std::vector<std::vector<int64_t>> sizes)
{
    if (UniMemInst.size() > 0)
    {
        return *(UniMemInst[0]);
    }
    _UniMem *ret = new _UniMem(device, pool_size, dtype, sizes);
    UniMemInst.push_back(ret);
    return *ret;
}
at::Tensor _fetch_batch(int p_id, int start, int end)
{
    return UniMemInst[0]->fetch_batch(p_id, start, end);
}

at::Tensor _UniMem::fetch_batch(int p_id, int start, int end)
{
    if (this->pool_size == -1)
    {
        std::vector<int64_t> s;
        s.push_back(end - start);
        s.insert(s.end(), this->sizes[p_id].begin(), this->sizes[p_id].end());

        return torch::empty(s, torch::dtype(this->dtype).device(torch::kCUDA, this->device));
    }

    at::Tensor ret = this->batched_gpu_memory[p_id].index({torch::indexing::Slice(start + this->virtual_offset, end + this->virtual_offset)

    });
    return ret;
}

void _UniMem::clear_virtual_mem()
{
    this->stk.clear(); // = []
    this->virtual_offset = 0;
    this->view_pool_size_attr = this->pool_size;
}
int _UniMem::view_pool_size()
{
    return this->view_pool_size_attr;
}
int _view_pool_size()
{
    return UniMemInst[0]->view_pool_size();
}
std::vector<at::Tensor> _prefetch(int idx, int batch_size)
{
    return UniMemInst[0]->prefetch(idx, batch_size);
}
std::vector<at::Tensor> _dummy_fetch(int batch_size)
{
    return UniMemInst[0]->dummy_fetch(batch_size);
}
std::vector<at::Tensor> _UniMem::prefetch(int idx, int batch_size)
{
    if (batch_size == 0)
    {
        return std::vector<at::Tensor>();
    }

    if (this->stk.size() > 0 && this->stk.back()[0] == idx)
    {
        this->stk.back()[1] += batch_size;
    }
    else
    {
        this->stk.push_back(std::vector<int>{idx, batch_size});
    }
    std::vector<at::Tensor> ret;
    for (int p_id = 0; p_id < this->sizes.size(); p_id++)
    {
        ret.push_back(this->batched_gpu_memory[p_id].index({torch::indexing::Slice(this->virtual_offset, this->virtual_offset + batch_size)}));
    }
    this->virtual_offset += batch_size;
    this->view_pool_size_attr -= batch_size;
    return ret;
}
std::vector<at::Tensor> _UniMem::dummy_fetch(int batch_size)
{
    if (batch_size == 0)
    {
        return std::vector<at::Tensor>();
    }

    std::vector<at::Tensor> ret;
    for (int p_id = 0; p_id < this->sizes.size(); p_id++)
    {
        ret.push_back(this->batched_gpu_memory[p_id].index({torch::indexing::Slice(0, 0 + batch_size)}));
    }
    return ret;
}
int _UniMem::fetch_top_prefetched_batch(int moe_id)
{
    if (this->stk.size() > 0 && this->stk.back()[0] == moe_id)
    {
        int batch_size = this->stk.back()[1];
        this->stk.pop_back();
        this->virtual_offset -= batch_size;
        this->view_pool_size_attr += batch_size;
        return batch_size;
    }
    else
    {
        return 0;
    }
}
int _fetch_top_prefetched_batch(int moe_id)
{
    return UniMemInst[0]->fetch_top_prefetched_batch(moe_id);
}

void _ParamProxy::fetch(bool from_pool, int prefetched)
{
    if (this->cpu_param)
    {
        if (from_pool)
        {
            this->ret = UniMemInst[0]->fetch_batch(this->p_id, this->start, this->end);
        }
        else
        {
            this->ret = torch::empty_like(this->param, torch::Device(torch::kCUDA, UniMemInst[0]->device));
        }

        _cpufp32_to_gpufp32(this->param.index({torch::indexing::Slice(prefetched, torch::indexing::None)}),
                            this->ret.index({torch::indexing::Slice(prefetched, torch::indexing::None)}), this->param.index({torch::indexing::Slice(prefetched, torch::indexing::None)}).numel());
    }
}
