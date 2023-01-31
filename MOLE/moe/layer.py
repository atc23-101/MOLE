import time
import torch
from UniMoE.utils import Ugate
import tutel.impls.fast_dispatch as tutel_fast_dispatcher
import UniMoE.utils.Udist as Udist
from UniMoE.utils.Uparam import CPUHostBatchedParameter, GPUHostBatchedParameter, UnifiedCPUTensor
from UniMoE.utils.Uact import build_activation_fwd, build_activation_bwd
from UniMoE.utils.Udist import MOLEDist
from UniMoE.utils.Ustream import ExpertStreamCtx
from UniMoE.utils.Utimer import UATimer

import UniMoE.core as core
from UniMoE.core import MOLECore
from pickle import dump



class MOLEModule(torch.nn.Module):
    _instance = []
    _fwdgaps = []
    _bwdgaps = []
    _forwarding_count = None
    _backwarding_count = None
    _cpu_optimizer = None
    use_prefetch = True
    _moe_scale = 1.0
    step = 0

    def __init__(self, total_d_experts, total_s_experts, embed_dim, ffn_dim, pool_size, topk, ebatch_size, dist, act, acc_step, adv_adam, use_pool, pool_for_extra=[False, False, False], overlap=True, gate_control_by_umoe=True, hybrid_adam=False, device_s_experts=-1):
        super().__init__()

        self.acc_step = acc_step

        self.topk = topk

        # distributed info
        self.local_rank = dist["local_rank"]
        self.world_size = dist["world_size"]
        self.rank = dist["rank"]
        self.device = torch.device(f"cuda:{self.local_rank}")

        print("CREATE")
        core.create_UniMem(self.local_rank, pool_size, "torch.float32",
                           [[ffn_dim, embed_dim],
                            [ffn_dim],
                               [ffn_dim, embed_dim],
                               [embed_dim]])

        # experts
        assert total_d_experts % self.world_size == 0
        assert total_s_experts % self.world_size == 0

        use_pool = use_pool if pool_size != -1 else False

        self.MOLECore = MOLECore(use_pool, pool_for_extra[0], pool_for_extra[1], pool_for_extra[2], total_s_experts // self.world_size, total_d_experts // self.world_size,
                                 ebatch_size, dist["world_size"], act, adv_adam, acc_step, 0, dist["local_rank"], len(MOLEModule._instance), total_d_experts + total_s_experts)
        print("DONE CREATE")

        # only support EP = DP
        self.d_experts = total_d_experts // self.world_size
        self.s_experts = total_s_experts // self.world_size
        self.total_experts = total_d_experts + total_s_experts

        print("EXPERT: ", self.s_experts, self.d_experts)
        # pool size = -1 means no pool
        self.pool_size = pool_size
        self.use_pool = use_pool
        self.ebatch_size = ebatch_size

        # linear dims
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        # not handled by torch
        self.gate = [Ugate.MOLETopk(embed_dim, self.total_experts)]
        if not gate_control_by_umoe:
            self.gate = torch.nn.ModuleList(self.gate)
        self.gate[0].linear = self.gate[0].linear.to(device=self.device)

        # dummy parameter for autograd
        self.dummy_param = torch.nn.Parameter().to(self.device)
        self.capacity_on = True

        # activation function
        self.act_str = act
        self.act = build_activation_fwd(act)
        assert isinstance(act, str)

        self.FC1_W_GPU = GPUHostBatchedParameter(
            size=[self.ffn_dim, self.embed_dim], count=self.s_experts, dtype=torch.float32, device=self.device, moe=self.MOLECore, p_id=0)
        self.FC1_B_GPU = GPUHostBatchedParameter(size=[self.ffn_dim], count=self.s_experts, dtype=torch.float32,
                                                 device=self.device, bias=True, moe=self.MOLECore, p_id=1, fan_in=self.embed_dim)
        self.FC2_W_GPU = GPUHostBatchedParameter(size=[self.ffn_dim, self.embed_dim], count=self.s_experts,
                                                 dtype=torch.float32, device=self.device, moe=self.MOLECore, p_id=2, transposed=True)
        self.FC2_B_GPU = GPUHostBatchedParameter(size=[self.embed_dim], count=self.s_experts, dtype=torch.float32,
                                                 device=self.device, bias=True, moe=self.MOLECore, p_id=3, fan_in=self.ffn_dim)

        UnifiedCPUTensor()
        UnifiedCPUTensor.init([[self.ffn_dim, self.embed_dim], [self.ffn_dim], [
                              self.ffn_dim, self.embed_dim], [self.embed_dim]], device_s_experts, torch.float32)
        self.FC1_W_CPU = CPUHostBatchedParameter(
            size=[self.ffn_dim, self.embed_dim], count=self.d_experts, dtype=torch.float32, p_id=0, moe=self.MOLECore, device_s_experts=device_s_experts)
        self.FC1_B_CPU = CPUHostBatchedParameter(
            size=[self.ffn_dim], count=self.d_experts, dtype=torch.float32, p_id=1, bias=True, moe=self.MOLECore, fan_in=self.embed_dim, device_s_experts=device_s_experts)
        self.FC2_W_CPU = CPUHostBatchedParameter(
            size=[self.ffn_dim, self.embed_dim], count=self.d_experts, dtype=torch.float32, p_id=2, moe=self.MOLECore, transposed=True, device_s_experts=device_s_experts)
        self.FC2_B_CPU = CPUHostBatchedParameter(
            size=[self.embed_dim], count=self.d_experts, dtype=torch.float32, p_id=3, bias=True, moe=self.MOLECore, fan_in=self.ffn_dim, device_s_experts=device_s_experts)

        self.overlap = overlap
        self.adv_adam = adv_adam
        self.hybrid_adam = hybrid_adam
        self.hybrid_adam_count = 0
        self.use_prefetch = False

        # prefetch
        self.local_prefetch_fwd_count = 0
        self.local_prefetch_bwd_count = 0
        self.global_prefetch_fwd_count = 0
        self.global_prefetch_bwd_count = 0

        self.global_prefetch = False

        self.grad_use_pool, self.mome_use_pool, self.vari_use_pool = pool_for_extra
        print("POOL FOR EXTRA: ", self.grad_use_pool,
              self.mome_use_pool, self.vari_use_pool)
        # instance append
        self.moe_id = len(MOLEModule._instance)
        MOLEModule._instance.append(self)
        if hybrid_adam:
            d = self.d_experts
            for ins in MOLEModule._instance:
                assert d == ins.d_experts, "Mixed Adam only support all moes have same experts"

    def set_hybrid_adam_count(self, hybrid_adam_count):
        self.MOLECore.set_hybrid_adam_count(hybrid_adam_count)

    def dummy_fetch(self, batch_size):
        torch.cuda.synchronize()
        time0 = time.time()
        if self.pool_size >= 1 and self.d_experts > 0:
            self.MOLECore.dummy_fetch(min(self.d_experts, batch_size))
        if self.d_experts < batch_size:
            print("DuMMY FETCH ", self.d_experts)
        torch.cuda.synchronize()
        time1 = time.time()
        return time1 - time0

    def global_prefetch_fwd(self):
        batch_sizes = self.global_prefetch_fwd_count
        for batch in batch_sizes:
            MOLEModule._instance[batch[0]].MOLECore.prefetch_fwd(batch[1])

    def global_prefetch_bwd(self):
        batch_sizes = self.global_prefetch_bwd_count
        for batch in batch_sizes:
            MOLEModule._instance[batch[0]].MOLECore.prefetch_bwd(batch[1])

    def local_prefetch_fwd(self):
        batch_size = self.local_prefetch_fwd_count
        self.MOLECore.prefetch_fwd(batch_size)

    def local_prefetch_bwd(self):
        batch_size = self.local_prefetch_bwd_count
        self.MOLECore.prefetch_bwd(batch_size)

    def prefetch_fwd(self, batch_size):
        self.MOLECore.prefetch_fwd(batch_size)

    def prefetch_bwd(self, batch_size):
        self.MOLECore.prefetch_bwd(batch_size)

    def set_prefetch(self, fwd_prefetch_num, bwd_prefetch_num):
        self.fwd_prefetch_num = fwd_prefetch_num
        self.bwd_prefetch_num = bwd_prefetch_num
        self.actual_prefetch_num = max(
            self.fwd_prefetch_num, self.bwd_prefetch_num)
        self.use_prefetch = True
        self.prefetched_fwd = 0
        self.prefetched_bwd = 0

    def instant_adam(self, record, base, batch_size, view_base, weight_fc1_w, weight_fc1_b, weight_fc2_w, weight_fc2_b, fc1_w_grad, fc1_b_grad, fc2_w_grad, fc2_b_grad):
        # torch.cuda.synchronize()
        m_fc1_w_, m_fc1_b_, m_fc2_w_, m_fc2_b_ = self.fetch_batch_cpu_mome(
            view_base + batch_size, base, batch_size)
        v_fc1_w_, v_fc1_b_, v_fc2_w_, v_fc2_b_ = self.fetch_batch_cpu_vari(
            view_base + batch_size + batch_size * self.mome_use_pool, base, batch_size)
        p_fc1_w = weight_fc1_w
        p_fc1_b = weight_fc1_b
        p_fc2_w = weight_fc2_w
        p_fc2_b = weight_fc2_b

        g_fc1_w = fc1_w_grad
        g_fc1_b = fc1_b_grad
        g_fc2_w = fc2_w_grad
        g_fc2_b = fc2_b_grad

        ExpertStreamCtx.tran_stream().wait_event(record.comp_event)

        with torch.cuda.stream(ExpertStreamCtx.tran_stream()):
            m_fc1_w_.fetch(self.mome_use_pool)
            m_fc1_b_.fetch(self.mome_use_pool)
            m_fc2_w_.fetch(self.mome_use_pool)
            m_fc2_b_.fetch(self.mome_use_pool)

            v_fc1_w_.fetch(self.vari_use_pool)
            v_fc1_b_.fetch(self.vari_use_pool)
            v_fc2_w_.fetch(self.vari_use_pool)
            v_fc2_b_.fetch(self.vari_use_pool)

            m_fc1_w = m_fc1_w_.bmaterial()
            m_fc1_b = m_fc1_b_.bmaterial()
            m_fc2_w = m_fc2_w_.bmaterial()
            m_fc2_b = m_fc2_b_.bmaterial()

            v_fc1_w = v_fc1_w_.bmaterial()
            v_fc1_b = v_fc1_b_.bmaterial()
            v_fc2_w = v_fc2_w_.bmaterial()
            v_fc2_b = v_fc2_b_.bmaterial()

        # torch.cuda.synchronize()
            if False and self.acc_step > 1:
                gw1, gb1, gw2, gb2 = self.fetch_batch_cpu_grad(
                    base, batch_size)

                gw1.fetch(False)
                gb1.fetch(False)
                gw2.fetch(False)
                gb2.fetch(False)

                g_fc1_w += gw1.bmaterial()
                g_fc1_b += gb1.bmaterial()
                g_fc2_w += gw2.bmaterial()
                g_fc2_b += gb2.bmaterial()

            p_fc1_w, m_fc1_w, v_fc1_w = self.adam(
                p_fc1_w, m_fc1_w, v_fc1_w, g_fc1_w)
            p_fc1_b, m_fc1_b, v_fc1_b = self.adam(
                p_fc1_b, m_fc1_b, v_fc1_b, g_fc1_b)
            p_fc2_w, m_fc2_w, v_fc2_w = self.adam(
                p_fc2_w, m_fc2_w, v_fc2_w, g_fc2_w)
            p_fc2_b, m_fc2_b, v_fc2_b = self.adam(
                p_fc2_b, m_fc2_b, v_fc2_b, g_fc2_b)
        # torch.cuda.synchronize()

            self.store_batch_cpu_para(
                base, batch_size, p_fc1_w, p_fc1_b, p_fc2_w, p_fc2_b)
            self.store_batch_cpu_mome(
                base, batch_size, m_fc1_w, m_fc1_b, m_fc2_w, m_fc2_b)
            self.store_batch_cpu_vari(
                base, batch_size, v_fc1_w, v_fc1_b, v_fc2_w, v_fc2_b)
        # torch.cuda.synchronize()

    def fetch_batch_gpu_param(self, base, batch_size):
        fc1_w = self.FC1_W_GPU.fetch_batch(base, base + batch_size)
        fc1_b = self.FC1_B_GPU.fetch_batch(base, base + batch_size)
        fc2_w = self.FC2_W_GPU.fetch_batch(base, base + batch_size)
        fc2_b = self.FC2_B_GPU.fetch_batch(base, base + batch_size)
        return fc1_w, fc1_b, fc2_w, fc2_b

    def fetch_batch_cpu_param(self, view_base, base, batch_size):
        fc1_w = self.FC1_W_CPU.fetch_batch(view_base, base, base + batch_size)
        fc1_b = self.FC1_B_CPU.fetch_batch(view_base, base, base + batch_size)
        fc2_w = self.FC2_W_CPU.fetch_batch(view_base, base, base + batch_size)
        fc2_b = self.FC2_B_CPU.fetch_batch(view_base, base, base + batch_size)
        return fc1_w, fc1_b, fc2_w, fc2_b

    def fetch_batch_cpu_vari(self, view_base, base, batch_size):
        fc1_w = self.FC1_W_CPU.fetch_batch_vari(
            view_base, base, base + batch_size)
        fc1_b = self.FC1_B_CPU.fetch_batch_vari(
            view_base, base, base + batch_size)
        fc2_w = self.FC2_W_CPU.fetch_batch_vari(
            view_base, base, base + batch_size)
        fc2_b = self.FC2_B_CPU.fetch_batch_vari(
            view_base, base, base + batch_size)
        return fc1_w, fc1_b, fc2_w, fc2_b

    def fetch_batch_cpu_mome(self, view_base, base, batch_size):
        fc1_w = self.FC1_W_CPU.fetch_batch_mome(
            view_base, base, base + batch_size)
        fc1_b = self.FC1_B_CPU.fetch_batch_mome(
            view_base, base, base + batch_size)
        fc2_w = self.FC2_W_CPU.fetch_batch_mome(
            view_base, base, base + batch_size)
        fc2_b = self.FC2_B_CPU.fetch_batch_mome(
            view_base, base, base + batch_size)
        return fc1_w, fc1_b, fc2_w, fc2_b

    def fetch_batch_cpu_grad(self, view_base, base, batch_size):
        fc1_w = self.FC1_W_CPU.fetch_batch_grad(
            view_base, base, base + batch_size)
        fc1_b = self.FC1_B_CPU.fetch_batch_grad(
            view_base, base, base + batch_size)
        fc2_w = self.FC2_W_CPU.fetch_batch_grad(
            view_base, base, base + batch_size)
        fc2_b = self.FC2_B_CPU.fetch_batch_grad(
            view_base, base, base + batch_size)
        return fc1_w, fc1_b, fc2_w, fc2_b

    def store_batch_cpu_grad(self, base, batch_size, fc1_w_grad, fc1_b_grad, fc2_w_grad, fc2_b_grad):
        if self.acc_step > 1:
            self.FC1_W_CPU.acc_store_batch_grad(
                base, base + batch_size, fc1_w_grad)
            self.FC1_B_CPU.acc_store_batch_grad(
                base, base + batch_size, fc1_b_grad)
            self.FC2_W_CPU.acc_store_batch_grad(
                base, base + batch_size, fc2_w_grad)
            self.FC2_B_CPU.acc_store_batch_grad(
                base, base + batch_size, fc2_b_grad)
        else:
            self.FC1_W_CPU.store_batch_grad(
                base, base + batch_size, fc1_w_grad)
            self.FC1_B_CPU.store_batch_grad(
                base, base + batch_size, fc1_b_grad)
            self.FC2_W_CPU.store_batch_grad(
                base, base + batch_size, fc2_w_grad)
            self.FC2_B_CPU.store_batch_grad(
                base, base + batch_size, fc2_b_grad)
        # torch.cuda.synchronize()

    def store_batch_cpu_para(self, base, batch_size, fc1_w_para, fc1_b_para, fc2_w_para, fc2_b_para):
        self.FC1_W_CPU.store_batch_para(base, base + batch_size, fc1_w_para)
        self.FC1_B_CPU.store_batch_para(base, base + batch_size, fc1_b_para)
        self.FC2_W_CPU.store_batch_para(base, base + batch_size, fc2_w_para)
        self.FC2_B_CPU.store_batch_para(base, base + batch_size, fc2_b_para)
        torch.cuda.synchronize()

    def store_batch_cpu_mome(self, base, batch_size, fc1_w_mome, fc1_b_mome, fc2_w_mome, fc2_b_mome):
        self.FC1_W_CPU.store_batch_mome(base, base + batch_size, fc1_w_mome)
        self.FC1_B_CPU.store_batch_mome(base, base + batch_size, fc1_b_mome)
        self.FC2_W_CPU.store_batch_mome(base, base + batch_size, fc2_w_mome)
        self.FC2_B_CPU.store_batch_mome(base, base + batch_size, fc2_b_mome)
        torch.cuda.synchronize()

    def store_batch_cpu_vari(self, base, batch_size, fc1_w_vari, fc1_b_vari, fc2_w_vari, fc2_b_vari):
        self.FC1_W_CPU.store_batch_vari(base, base + batch_size, fc1_w_vari)
        self.FC1_B_CPU.store_batch_vari(base, base + batch_size, fc1_b_vari)
        self.FC2_W_CPU.store_batch_vari(base, base + batch_size, fc2_w_vari)
        self.FC2_B_CPU.store_batch_vari(base, base + batch_size, fc2_b_vari)
        torch.cuda.synchronize()

    def unset_lazy_zero(self):
        self.FC1_W_CPU.unset_lazy_zero()
        self.FC1_B_CPU.unset_lazy_zero()
        self.FC2_W_CPU.unset_lazy_zero()
        self.FC2_B_CPU.unset_lazy_zero()

    @torch.no_grad()
    def set_layer_state(self, states, grad):
        gate_state = states["gate"]

        self.gate[0].linear.weight.copy_(gate_state[0])
        if self.gate[0].linear.bias is not None:
            self.gate[0].linear.bias.copy_(gate_state[1])
        expert_state = states["experts"]
        j = 0
        for i in range(self.s_experts):
            expert = expert_state[j]
            self.FC1_W_GPU.fetch_single(i).copy_(expert[0])
            self.FC1_B_GPU.fetch_single(i).copy_(expert[1])
            self.FC2_W_GPU.fetch_single(i).copy_(expert[2])
            self.FC2_B_GPU.fetch_single(i).copy_(expert[3])
            j += 1
        for i in range(self.d_experts):
            expert = expert_state[j]
            self.FC1_W_CPU.fetch_single(i).copy_(expert[0])
            self.FC1_B_CPU.fetch_single(i).copy_(expert[1])
            self.FC2_W_CPU.fetch_single(i).copy_(expert[2])
            self.FC2_B_CPU.fetch_single(i).copy_(expert[3])
            j += 1
        if grad:
            grad_state = states["grad"]
            j = 0
            for i in range(self.s_experts):
                expertg = grad_state[j]
                self.FC1_W_GPU.fetch_single_grad(i).copy_(expertg[0])
                self.FC1_B_GPU.fetch_single_grad(i).copy_(expertg[1])
                self.FC2_W_GPU.fetch_single_grad(i).copy_(expertg[2])
                self.FC2_B_GPU.fetch_single_grad(i).copy_(expertg[3])
                j += 1
            for i in range(self.d_experts):
                expertg = grad_state[j]
                self.FC1_W_CPU.fetch_single_grad(i).copy_(expertg[0])
                self.FC1_B_CPU.fetch_single_grad(i).copy_(expertg[1])
                self.FC2_W_CPU.fetch_single_grad(i).copy_(expertg[2])
                self.FC2_B_CPU.fetch_single_grad(i).copy_(expertg[3])
                j += 1
            self.gate[0].linear.weight.grad = grad_state[j].to(
                self.gate[0].linear.weight.device)
            j += 1
            if self.gate[0].linear.bias is not None:
                self.gate[0].linear.bias.grad = grad_state[j].to(
                    self.gate[0].linear.bias.device)

    @torch.no_grad()
    def get_layer_state(self, grad):
        moe_state = {}
        moe_state["gate"] = [self.gate[0].linear.weight.cpu(
        ), self.gate[0].linear.bias.cpu() if self.gate[0].linear.bias is not None else None]
        expert_states = []
        grad_states = []
        for i in range(self.s_experts):
            expert = [self.FC1_W_GPU.fetch_single(i).cpu(), self.FC1_B_GPU.fetch_single(i).cpu(),
                      self.FC2_W_GPU.fetch_single(i).cpu(), self.FC2_B_GPU.fetch_single(i).cpu()]
            expert_states.append(expert)
        for i in range(self.d_experts):
            expert = [self.FC1_W_CPU.fetch_single(i).cpu(), self.FC1_B_CPU.fetch_single(i).cpu(),
                      self.FC2_W_CPU.fetch_single(i).cpu(), self.FC2_B_CPU.fetch_single(i).cpu()]
            expert_states.append(expert)

        if grad:
            for i in range(self.s_experts):
                expertg = [self.FC1_W_GPU.fetch_single_grad(i).cpu(), self.FC1_B_GPU.fetch_single_grad(i).cpu(),
                           self.FC2_W_GPU.fetch_single_grad(i).cpu(), self.FC2_B_GPU.fetch_single_grad(i).cpu()]
                grad_states.append(expertg)
            for i in range(self.d_experts):
                expertg = [self.FC1_W_CPU.fetch_single_grad(i).cpu(), self.FC1_B_CPU.fetch_single_grad(i).cpu(),
                           self.FC2_W_CPU.fetch_single_grad(i).cpu(), self.FC2_B_CPU.fetch_single_grad(i).cpu()]
                grad_states.append(expertg)
            grad_states.append(self.gate[0].linear.weight.grad.cpu())
            grad_states.append(self.gate[0].linear.bias.grad.cpu(
            ) if self.gate[0].linear.bias is not None else None)

        moe_state["experts"] = expert_states
        moe_state["grad"] = grad_states
        return moe_state

    def forward(self, input_, adam_param={"beta1": 0.9, "beta2": 0.999, "lr": 0.001, "t": 100, "lam": 1, "eps": 1e-8}, capacity_factor=1.0, time_it=False, prefetch=False):


        original_shape = input_.shape
        original_dtype = input_.dtype

        if prefetch:
            input_ = pfetchbwd.apply(input_, self)



        resized_input = input_.reshape(-1, self.embed_dim)

        scores = self.gate[0](resized_input)

        if self.capacity_on:
            crit, l_aux = tutel_fast_dispatcher.extract_critical(
                scores, top_k=self.topk, capacity_factor=capacity_factor)
            self.l_aux = l_aux
            dispatched_input = tutel_fast_dispatcher.fast_encode(
                data=input_, critial_data=crit)
        else:
            pass

        experts_output = BatchFusedFWD.apply(
            dispatched_input, self, self.dummy_param, time_it, prefetch)

        if self.capacity_on:
            expert_output = tutel_fast_dispatcher.fast_decode(
                data=experts_output, critial_data=crit)

        expert_output = expert_output.reshape(original_shape)

        out = expert_output.to(original_dtype)

        if prefetch:
            if self.moe_id < len(MOLEModule._instance) - 1:
                if self.global_prefetch:
                    MOLEModule._instance[self.moe_id + 1].global_prefetch_fwd()
                MOLEModule._instance[self.moe_id + 1].local_prefetch_fwd()
            elif self.moe_id == len(MOLEModule._instance) - 1:
                if self.global_prefetch:
                    MOLEModule._instance[-1].global_prefetch_bwd()
                MOLEModule._instance[-1].local_prefetch_bwd()



        return out

    def d_experts_param(self):
        ret = []
        if self.d_experts > 0:
            ret.append(self.FC1_W_CPU.all_tensors)
            ret.append(self.FC1_B_CPU.all_tensors)
            ret.append(self.FC2_W_CPU.all_tensors)
            ret.append(self.FC2_B_CPU.all_tensors)
        return ret

    def s_experts_param(self):
        ret = []
        if self.s_experts > 0:
            ret.append(self.FC1_W_GPU.fetch_all_with_grad())
            ret.append(self.FC1_B_GPU.fetch_all_with_grad())
            ret.append(self.FC2_W_GPU.fetch_all_with_grad())
            ret.append(self.FC2_B_GPU.fetch_all_with_grad())
        return ret

    def gate_param(self):
        if self.gate[0].linear.bias is not None:
            return [self.gate[0].linear.weight, self.gate[0].linear.bias]
        else:
            return [self.gate[0].linear.weight]

