#ifndef __MOLE_DEFINE__
#define __MOLE_DEFINE__

#define USE_C10D_NCCL

#define USE_MOLE_GELU
// #undef USE_MOLE_GELU
#define TRAN_BACK_STREAM
//#undef TRAN_BACK_STREAM

#include <nccl.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif
#include <torch/csrc/cuda/nccl.h>

#include <vector>
#include <string>
#include <unordered_map>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

/*gelu*/
#ifdef USE_MOLE_GELU
void gelu_inplace_backward(int bsz, int intermediate_size, float *d_output, const float *input_buf, cudaStream_t stream);
inline void inplace_gelu_bwd(at::Tensor &grad, at::Tensor &input)
{
    gelu_inplace_backward(grad.numel() / grad.size(-1), grad.size(grad.dim() - 1), (float *)grad.data_ptr(), (const float *)input.data_ptr(), c10::cuda::getCurrentCUDAStream().stream());
}
#endif

class _ParamProxy
{
public:
    _ParamProxy(bool cpu_param, at::Tensor param, int p_id, int start, int end, bool bias)
    {
        this->cpu_param = cpu_param;
        this->param = param;
        this->p_id = p_id;
        this->start = start;
        this->end = end;
        this->bias = bias;
    }

    at::Tensor material()
    {
        if (this->cpu_param)
        {
            if (this->bias)
            {
                return this->ret.unsqueeze(1);
            }
            else
            {
                return this->ret;
            }
        }
        else
        {
            if (this->bias)
            {
                return this->param.unsqueeze(1);
            }
            else
            {
                return this->param;
            }
        }
    }
    void fetch(bool from_pool, int prefetched);

    at::Tensor grad_bmaterial()
    {
        if (this->cpu_param)
        {
            assert(False);
            return this->ret;
        }
        else
        {
            return this->grad;
        }
    }

    at::Tensor bmaterial()
    {
        if (this->cpu_param)
        {
            return this->ret;
        }
        else
        {
            return this->param;
        }
    }
    at::Tensor _get_param() const
    {
        return this->param;
    }
    bool cpu_param;
    at::Tensor param, ret, grad;
    int p_id;
    int start;
    int end;
    bool bias;
    ~_ParamProxy()
    {
    }
};

//from stackflow
struct pair_int_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

class _UniMem
{
public:
    _UniMem(int device, int pool_size, const std::string &dtype, std::vector<std::vector<int64_t>> sizes)
    {

        this->device = device;
        this->virtual_offset = 0;

        if (dtype == "torch.float32")
        {
            this->dtype = torch::kFloat32;
        }

        this->view_pool_size_attr = pool_size;
        this->pool_size = pool_size;
        this->sizes = sizes;
        if (pool_size != -1)
        {
            for (int i = 0; i < sizes.size(); i++)
            {
                std::vector<int64_t> s;
                s.push_back(pool_size);
                s.insert(s.end(), sizes[i].begin(), sizes[i].end());
                this->batched_gpu_memory.push_back(torch::zeros(s, torch::dtype(this->dtype).device(torch::kCUDA, device)));
            }
        }
    }
    std::string get() { return "all but me"; }

    int device;
    int virtual_offset;
    int pool_size;
    std::vector<at::Tensor> batched_gpu_memory;
    int view_pool_size_attr;
    torch::Dtype dtype;
    std::unordered_map<std::pair<int, int>, int, pair_int_hash> batch_records;
    std::vector<std::vector<int>> stk;
    std::vector<std::vector<int64_t>> sizes;

    void clear_batch_record();
    std::vector<int> get_batch_record(int start, int end);
    void set_batch_record(int start, int end, int batch_record);
    at::Tensor fetch_batch(int p_id, int start, int end);
    void clear_virtual_mem();
    int view_pool_size();
    std::vector<at::Tensor> prefetch(int idx, int batch_size);
    std::vector<at::Tensor> dummy_fetch(int batch_size);

    int fetch_top_prefetched_batch(int moe_id);
};

class _CPUParamCore
{
public:
    bool bias;
    at::Tensor all_tensors;
    at::Tensor grad_;
    at::Tensor momentum;
    at::Tensor variance;
    int p_id;
    bool lazy_zero;
    _CPUParamCore(at::Tensor all_tensors, at::Tensor grad_, at::Tensor momentum, at::Tensor variance, bool bias, int p_id)
    {
        this->bias = bias;
        this->all_tensors = all_tensors;
        this->grad_ = grad_;
        this->momentum = momentum;
        this->variance = variance;
        this->p_id = p_id;
        this->lazy_zero = true;
    }
    at::Tensor get_all_tensors() const
    {
        return this->all_tensors;
    }

    void set_lazy_zero() { lazy_zero = true; }
    void unset_lazy_zero() { lazy_zero = false; }

    at::Tensor fetch_single(int id);
    at::Tensor fetch_single_grad(int id);
    at::Tensor prefetch_to(int start, int batch_size, at::Tensor &gpu_param);
    at::Tensor prefetch_tensor(int start, int batch_size);
    void store_batch_grad(int start, int end, at::Tensor &grad);
    void store_batch_para(int start, int end, at::Tensor &para);
    void store_batch_mome(int start, int end, at::Tensor &mome);
    void store_batch_vari(int start, int end, at::Tensor &vari);
    void acc_store_batch_grad(int start, int end, at::Tensor &grad);

    void store_variance(int start, at::Tensor &variance);
    void store_momentum(int start, at::Tensor &momentum);
    void store_weight(int start, at::Tensor &param);
    void fetch_variance(int start, at::Tensor &variance);
    void fetch_momentum(int start, at::Tensor &momentum);

    _ParamProxy fetch_batch(int view_base, int start, int end);
    _ParamProxy fetch_batch_vari(int view_base, int start, int end);
    _ParamProxy fetch_batch_mome(int view_base, int start, int end);
    _ParamProxy fetch_batch_grad(int view_base, int start, int end);
};

class _GPUParamCore
{
public:
    at::Tensor all_tensors;
    at::Tensor grad_tensors;
    bool bias;
    _GPUParamCore(at::Tensor all_tensors, at::Tensor grad_tensors, bool bias)
    {
        this->all_tensors = all_tensors;
        this->grad_tensors = grad_tensors;
        this->bias = bias;
    }
    at::Tensor _get_all_tensors() const
    {
        return all_tensors;
    }
    at::Tensor _get_grad_tensors() const
    {
        return grad_tensors;
    }
    at::Tensor fetch_single(int id);
    at::Tensor fetch_single_grad(int id);
    _ParamProxy fetch_batch(int start, int end);
};

class act_func
{
public:
    std::string act;
    std::vector<at::Tensor> saved;
    act_func(std::string act_f)
    {
        this->act = act_f;
    }
    at::Tensor forward(at::Tensor &input)
    {
        at::Tensor ret;
        if (this->act == "relu")
        {
            input.relu_();
            this->saved.push_back(input);
            return input;
        }
        else if (this->act == "gelu")
        {
            ret = at::gelu(input);
            this->saved.push_back(input);
        }
        return ret;
    }

    at::Tensor backward(at::Tensor grad)
    {
        at::Tensor ret;
        if (this->act == "relu")
        {
            saved[0].greater_(0);

            saved[0] *= grad;
            return saved[0];
        }
        else if (this->act == "gelu")
        {
#ifdef USE_MOLE_GELU
            inplace_gelu_bwd(grad, this->saved.back());
            saved.clear();
            return grad;
#else
            ret = at::gelu_backward(grad, saved[0]);
            saved.clear();
            return ret;
#endif
        }

        return ret;
    }
};
class _MOLECore;
class _BatchRecord;
class _BatchFusedFWD
{

    _MOLECore *moe;

public:
    ~_BatchFusedFWD()
    {
    }
    bool input_requires_grad;
    std::vector<_BatchRecord> records;
    at::Tensor forward(at::Tensor &dispatched_input, _MOLECore *moe, bool requires_grad);
    at::Tensor backward(at::Tensor &all_output_grad);
};
class _MOLECore
{
public:
    bool use_pool;
    int acc_step;
    bool adv_adam;
    int step;
    int device;
    int moe_id;
    bool mome_use_pool, vari_use_pool, grad_use_pool;
    int total_experts;
    int s_experts, d_experts, ebatch_size;
    int world_size;
    int fwd_base, bwd_base;
    int hybrid_adam_count;
    std::string act_name;

    _BatchFusedFWD *fwdcore;
    _CPUParamCore *FC1_W_CPU;
    _CPUParamCore *FC1_B_CPU;
    _CPUParamCore *FC2_W_CPU;
    _CPUParamCore *FC2_B_CPU;

    _GPUParamCore *FC1_W_GPU;
    _GPUParamCore *FC1_B_GPU;
    _GPUParamCore *FC2_W_GPU;
    _GPUParamCore *FC2_B_GPU;

    void add_tran_tmp(std::vector<at::Tensor> &tmp){
        tran_tmp_list.push_back(tmp);
    }
    std::vector<std::vector<at::Tensor> >tran_tmp_list;
    int tran_tmp_list_counter;
    void remove_tran_tmp(){
        tran_tmp_list[tran_tmp_list_counter].clear();
        tran_tmp_list_counter += 1;
    }
    void clear_tran_tmp(){
        tran_tmp_list.clear();
        tran_tmp_list_counter = 0;
    }

    _MOLECore(bool use_pool, bool mome_use_pool, bool vari_use_pool, bool grad_use_pool, int s_experts, int d_experts, int ebatch_size, int world_size, std::string act_name, bool adv_adam, int acc_step, int step, int device, int moe_id,
              int total_experts) : FC1_W_CPU(nullptr), FC1_B_CPU(nullptr), FC2_W_CPU(nullptr), FC2_B_CPU(nullptr)
    {
        this->use_pool = use_pool;
        this->acc_step = acc_step;
        this->adv_adam = adv_adam;
        this->step = step;
        this->device = device;
        this->moe_id = moe_id;
        this->mome_use_pool = mome_use_pool;
        this->vari_use_pool = vari_use_pool;
        this->grad_use_pool = grad_use_pool;
        this->total_experts = total_experts;
        this->s_experts = s_experts;
        this->d_experts = d_experts;
        this->ebatch_size = ebatch_size;
        this->world_size = world_size;
        this->act_name = act_name;

        this->fwdcore = nullptr;
        this->fwd_base = 0;
        this->bwd_base = 0;

        this->hybrid_adam_count = d_experts;
        this->tran_tmp_list_counter = 0;
    }
    void set_CPUParamCore(int id, at::Tensor all_tensors, at::Tensor grad_, at::Tensor momentum, at::Tensor variance, bool bias, int p_id);
    void set_GPUParamCore(int id, at::Tensor all_tensors, at::Tensor grad_tensors, bool bias);
    _CPUParamCore *get_CPUParamCore(int i);
    _GPUParamCore *get_GPUParamCore(int i);

    at::Tensor _GPUParamCore_get_all_tensors(int p_id)
    {
        return get_GPUParamCore(p_id)->_get_all_tensors();
    }
    at::Tensor _GPUParamCore_get_grad_tensors(int p_id)
    {
        return get_GPUParamCore(p_id)->_get_grad_tensors();
    }
    at::Tensor _GPUParamCore_fetch_single(int p_id, int id)
    {
        return get_GPUParamCore(p_id)->fetch_single(id);
    }
    at::Tensor _GPUParamCore_fetch_single_grad(int p_id, int id)
    {
        return get_GPUParamCore(p_id)->fetch_single_grad(id);
    }
    _ParamProxy _GPUParamCore_fetch_batch(int p_id, int start, int end)
    {
        return get_GPUParamCore(p_id)->fetch_batch(start, end);
    }
    void prefetch_fwd(int batch_size);
    void prefetch_bwd(int batch_size);
    void dummy_fetch(int batch_size);
    void set_lazy_zero()
    {
        if (get_CPUParamCore(0))
            get_CPUParamCore(0)->set_lazy_zero();
        if (get_CPUParamCore(1))
            get_CPUParamCore(1)->set_lazy_zero();
        if (get_CPUParamCore(2))
            get_CPUParamCore(2)->set_lazy_zero();
        if (get_CPUParamCore(3))
            get_CPUParamCore(3)->set_lazy_zero();
    }

    void unset_lazy_zero()
    {
        if (get_CPUParamCore(0))
            get_CPUParamCore(0)->unset_lazy_zero();
        if (get_CPUParamCore(1))
            get_CPUParamCore(1)->unset_lazy_zero();
        if (get_CPUParamCore(2))
            get_CPUParamCore(2)->unset_lazy_zero();
        if (get_CPUParamCore(3))
            get_CPUParamCore(3)->unset_lazy_zero();
    }

    void _CPUParamCore_set_lazy_zero(int p_id)
    {
        get_CPUParamCore(p_id)->set_lazy_zero();
    }
    void _CPUParamCore_unset_lazy_zero(int p_id)
    {
        get_CPUParamCore(p_id)->unset_lazy_zero();
    }
    at::Tensor _CPUParamCore_fetch_single(int p_id, int id)
    {
        return get_CPUParamCore(p_id)->fetch_single(id);
    }
    at::Tensor _CPUParamCore_fetch_single_grad(int p_id, int id)
    {
        return get_CPUParamCore(p_id)->fetch_single_grad(id);
    }
    at::Tensor _CPUParamCore_prefetch_to(int p_id, int start, int batch_size, at::Tensor &gpu_param)
    {
        return get_CPUParamCore(p_id)->prefetch_to(start, batch_size, gpu_param);
    }
    at::Tensor _CPUParamCore_prefetch_tensor(int p_id, int start, int batch_size)
    {
        return get_CPUParamCore(p_id)->prefetch_tensor(start, batch_size);
    }

    void _CPUParamCore_store_batch_grad(int p_id, int start, int end, at::Tensor &grad)
    {
        get_CPUParamCore(p_id)->store_batch_grad(start, end, grad);
    }
    void _CPUParamCore_store_batch_para(int p_id, int start, int end, at::Tensor &para)
    {
        get_CPUParamCore(p_id)->store_batch_para(start, end, para);
    }
    void _CPUParamCore_store_batch_mome(int p_id, int start, int end, at::Tensor &mome)
    {
        get_CPUParamCore(p_id)->store_batch_mome(start, end, mome);
    }
    void _CPUParamCore_store_batch_vari(int p_id, int start, int end, at::Tensor &vari)
    {
        get_CPUParamCore(p_id)->store_batch_vari(start, end, vari);
    }
    void _CPUParamCore_acc_store_batch_grad(int p_id, int start, int end, at::Tensor &grad)
    {
        get_CPUParamCore(p_id)->acc_store_batch_grad(start, end, grad);
    }

    _ParamProxy _CPUParamCore_fetch_batch(int p_id, int view_base, int start, int end)
    {
        return get_CPUParamCore(p_id)->fetch_batch(view_base, start, end);
    }
    _ParamProxy _CPUParamCore_fetch_batch_vari(int p_id, int view_base, int start, int end)
    {
        return get_CPUParamCore(p_id)->fetch_batch_vari(view_base, start, end);
    }
    _ParamProxy _CPUParamCore_fetch_batch_mome(int p_id, int view_base, int start, int end)
    {
        return get_CPUParamCore(p_id)->fetch_batch_mome(view_base, start, end);
    }
    _ParamProxy _CPUParamCore_fetch_batch_grad(int p_id, int view_base, int start, int end)
    {
        return get_CPUParamCore(p_id)->fetch_batch_grad(view_base, start, end);
    }

    float beta1, beta2, lr, lam, eps;
    int t;
    bool adamW;
    void set_adam(float beta1, float beta2, int t, float lr, float lam, float eps, bool adamW, int hybrid_adam_count)
    {
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->t = t;
        this->lr = lr;
        this->lam = lam;
        this->eps = eps;
        this->adamW = adamW;
        if (hybrid_adam_count == -1)
        {
            this->hybrid_adam_count = this->d_experts;
        }
        else
        {
            this->hybrid_adam_count = hybrid_adam_count;
        }
    }

    std::vector<at::Tensor> adam(at::Tensor p, at::Tensor m, at::Tensor v, at::Tensor g)
    {
        float beta1 = this->beta1;
        float beta2 = this->beta2;
        int t = this->t + 1;
        float lr = this->lr;
        float lam = this->lam;
        float eps = this->eps;

        if (this->adamW)
        {
            p = p * (1 - lam * lr);
        }
        else
        {
            g = g + lam * p;
        }
        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g * g;

        at::Tensor mt = m / (1 - std::pow(beta1, t));
        at::Tensor vt = v / (1 - std::pow(beta2, t));
#ifdef TRAN_BACK_STREAM
        at::Tensor p_back = p - lr * mt / (vt.sqrt() + eps);
#else 
        p = p - lr * mt / (vt.sqrt() + eps);
#endif
        std::vector<at::Tensor> ret;
#ifdef TRAN_BACK_STREAM
        ret.push_back(p_back);
#else 
        ret.push_back(p);
#endif
        ret.push_back(m);
        ret.push_back(v);
        return ret;
    }

    void store_batch_cpu_grad(int base, int batch_size, at::Tensor &fc1_w_grad, at::Tensor &fc1_b_grad, at::Tensor &fc2_w_grad, at::Tensor &fc2_b_grad);
    void store_batch_cpu_para(int base, int batch_size, at::Tensor &fc1_w_para, at::Tensor &fc1_b_para, at::Tensor &fc2_w_para, at::Tensor &fc2_b_para);
    void store_batch_cpu_mome(int base, int batch_size, at::Tensor &fc1_w_mome, at::Tensor &fc1_b_mome, at::Tensor &fc2_w_mome, at::Tensor &fc2_b_mome);
    void store_batch_cpu_vari(int base, int batch_size, at::Tensor &fc1_w_vari, at::Tensor &fc1_b_vari, at::Tensor &fc2_w_vari, at::Tensor &fc2_b_vari);

    std::vector<_ParamProxy> fetch_batch_gpu_param(int batch_base, int batch_size);
    std::vector<_ParamProxy> fetch_batch_cpu_param(int view_base, int batch_base, int batch_size);
    std::vector<_ParamProxy> fetch_batch_cpu_vari(int view_base, int batch_base, int batch_size);
    std::vector<_ParamProxy> fetch_batch_cpu_mome(int view_base, int batch_base, int batch_size);
    std::vector<_ParamProxy> fetch_batch_cpu_grad(int view_base, int batch_base, int batch_size);

    void instant_adam(_BatchRecord *record, int base, int batch_size, int view_base, at::Tensor weight_fc1_w, at::Tensor weight_fc1_b, at::Tensor weight_fc2_w, at::Tensor weight_fc2_b, at::Tensor fc1_w_grad, at::Tensor fc1_b_grad, at::Tensor fc2_w_grad, at::Tensor fc2_b_grad, int hybrid_adam_count);

    at::Tensor forward(at::Tensor &dispatched_input, bool requires_grad)
    {
        this->fwdcore = new _BatchFusedFWD();
        return this->fwdcore->forward(dispatched_input, this, requires_grad);
    }
    at::Tensor backward(at::Tensor &all_output_grad)
    {
        at::Tensor ret = this->fwdcore->backward(all_output_grad);

        delete this->fwdcore;
        this->fwdcore = nullptr;
        return ret;
    }
};

inline void CUDART_CB release_tranback_tensor(cudaStream_t stream, cudaError_t status, void *data){
     ((_MOLECore*)data)->remove_tran_tmp();
}

class _BatchRecord
{
public:
    int device;
    int token_dim;
    int embed_dim;
    bool use_cpu_param;
    _MOLECore *moe;
    bool use_pool;
    int recid;
    act_func act;
    std::string act_name;
    bool acc;
    int batch_base;
    int batch_size;
    int base_in_all;
    int view_base;
    int world_size;
    bool mem_optim;

    int prefetched;
    at::cuda::CUDAEvent comp_event;
    at::cuda::CUDAEvent comm_event;
    at::cuda::CUDAEvent para_event;

    at::Tensor x2, x0;
    std::vector<at::Tensor> extras;

    std::vector<_ParamProxy> params;
    _BatchRecord(int device, int base_in_group, int base_in_all, int batch_size, bool use_cpu_param, _MOLECore *moe, int token_dim, int embed_dim, int recid, int prefetched, int view_base, int world_size, std::string act_name) : act(act_name)
    {
        this->device = device;
        this->batch_base = base_in_group;
        this->base_in_all = base_in_all;
        this->batch_size = batch_size;
        this->use_cpu_param = use_cpu_param;
        this->moe = moe;
        this->token_dim = token_dim;
        this->embed_dim = embed_dim;
        this->recid = recid;
        this->prefetched = prefetched;
        this->view_base = view_base;
        this->world_size = world_size;
        this->use_pool = moe->use_pool;
        this->acc = moe->acc_step > 1;
        this->act_name = act_name;

        this->mem_optim = true;
    }

    void store_grad(at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2);
    void fetch_para();
    void release_para();
    at::Tensor a2a_before(at::Tensor &dispatched_input);
    at::Tensor a2a_after(at::Tensor &expert_output, at::Tensor &all_output);
    at::Tensor forward(at::Tensor x, at::Tensor all_output, bool input_requires_grad);

    void bffn_bwd_acc(at::Tensor &batched_out_grad, at::Tensor &batched_w1, at::Tensor &batched_w2, at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2, at::Tensor &x_grad, act_func &act, at::Tensor &x, at::Tensor &x2, bool input_requires_grad, bool is_last, at::Tensor &remover);
    void bffn_bwd(at::Tensor &batched_out_grad, at::Tensor &batched_w1, at::Tensor &batched_w2, at::Tensor &g_w1, at::Tensor &g_b1, at::Tensor &g_w2, at::Tensor &g_b2, at::Tensor &x_grad, act_func &act, at::Tensor &x, at::Tensor &x2, bool input_requires_grad, bool is_last, at::Tensor &remover);
    void bffn_fwd(at::Tensor batched_x, at::Tensor batched_w1, at::Tensor batched_b1, at::Tensor batched_w2, at::Tensor batched_b2, act_func &act, at::Tensor &out, bool input_requires_grad);
    at::Tensor first_a2a(at::Tensor &dispatched_input, bool is_last);
    at::Tensor second_a2a(at::Tensor &expert_output);
    at::Tensor expert_fwd(at::Tensor &x, bool input_requires_grad);
    std::vector<at::Tensor> expert_bwd(at::Tensor &o_grad, bool input_requires_grad, bool is_last, at::Tensor &all_grad);
};

class moca
{
public:
    at::Tensor fds;
    at::Tensor get()
    {
        return fds;
    }
    void set(at::Tensor x)
    {
        fds = x;
    }
};
class test_Tensor
{
public:
    moca *mo;
    void setmoca(moca &x)
    {
        mo = &x;
    }
    moca getmoca()
    {
        return *mo;
    }
    moca getmocaref()
    {
        return *mo;
    }

    at::Tensor ga;
    at::Tensor gcr;
    at::Tensor gr;
    test_Tensor() {}
    void set(at::Tensor a, const at::Tensor &cr, at::Tensor &r)
    {
        ga = a;
        gcr = cr;
        gr = r;

        a[0][0] = 1;
        r[0][0] = 1;
    }
    at::Tensor get0()
    {
        return ga;
    }
    at::Tensor get1()
    {
        return gcr;
    }
    at::Tensor get2()
    {
        return gr;
    }
};

/* communication */
void _ensure_nccl(c10d::ProcessGroupNCCL &p, int device);
void _MOLEAll2Allfp32(std::vector<at::Tensor> src, std::vector<at::Tensor> tgt, int device, int world_size, int stream_id, uint64_t s, std::string group, int world_rank);

/* stream */
uint64_t _get_stream_cdata(int device, int idx);

/* UniMem */
_UniMem *_get_UniMem();

at::Tensor _fetch_batch(int p_id, int start, int end);
void _clear_batch_record();
_UniMem _create_UniMem(int device, int pool_size, const std::string &dtype, std::vector<std::vector<int64_t>> sizes);
void _clear_virtual_mem();
std::vector<at::Tensor> _prefetch(int idx, int batch_size);
std::vector<at::Tensor> _dummy_fetch(int batch_size);
int _fetch_top_prefetched_batch(int moe_id);
int _view_pool_size();
void _set_batch_record(int start, int end, int batch_record);
std::vector<int> _get_batch_record(int start, int end);

#endif