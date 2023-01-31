import torch
import math
from UniMoE.core import ParamProxy

from UniMoE.core import GPUParamCore, CPUParamCore


class UnifiedCPUTensor(object):
    _instance = None
    _optimizer = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            print("#!#building UniMoE UnifiedCPUTensor: ")
            cls._instance = object.__new__(cls)
            cls._instance.tensors = {}
            cls._instance.grad = {}
            cls._instance.momentum = {}
            cls._instance.variance = {}

            cls._instance.counter = {}
            cls._instance.counter_grad = {}
            cls._instance.counter_momentum = {}
            cls._instance.counter_variance = {}
            cls._instance.inited = False
        return cls._instance

    def __init__(self, *args, **kw):
        pass

    @staticmethod
    def init(sizes, device_s_experts, dtype):
        if UnifiedCPUTensor._instance.inited == False:
            if device_s_experts > 0:
                x = device_s_experts
                ss = 0
                for i in sizes:
                    t = 1
                    for xx in i:
                        t *= xx
                    ss += t
                x *= ss

                UnifiedCPUTensor._instance.numel = x
                UnifiedCPUTensor._instance.pool_t = torch.zeros(
                    x, device="cpu", dtype=dtype, pin_memory=True)
                UnifiedCPUTensor._instance.pool_g = torch.zeros(
                    x, device="cpu", dtype=dtype, pin_memory=True)
                UnifiedCPUTensor._instance.pool_m = torch.zeros(
                    x, device="cpu", dtype=dtype, pin_memory=True)
                UnifiedCPUTensor._instance.pool_v = torch.zeros(
                    x, device="cpu", dtype=dtype, pin_memory=True)

                UnifiedCPUTensor._instance.pool_t.grad_ = UnifiedCPUTensor._instance.pool_g
                UnifiedCPUTensor._instance.pool_t.momentum = UnifiedCPUTensor._instance.pool_m
                UnifiedCPUTensor._instance.pool_t.variance = UnifiedCPUTensor._instance.pool_v

                UnifiedCPUTensor._instance.pool_t_d = 0
                UnifiedCPUTensor._instance.pool_g_d = 0
                UnifiedCPUTensor._instance.pool_m_d = 0
                UnifiedCPUTensor._instance.pool_v_d = 0
                UnifiedCPUTensor._instance.inited = True
            else:
                UnifiedCPUTensor._instance.numel = 0
                UnifiedCPUTensor._instance.inited = True

    @staticmethod
    def get_tensor(size, device_s_experts, count, dtype, p_id):
        if p_id not in UnifiedCPUTensor._instance.tensors.keys():
            shape = (device_s_experts, *list(size))
            x = 1
            for s in shape:
                x *= s

            UnifiedCPUTensor._instance.tensors[p_id] = UnifiedCPUTensor._instance.pool_t[
                UnifiedCPUTensor._instance.pool_t_d:UnifiedCPUTensor._instance.pool_t_d + x].view(shape)
            UnifiedCPUTensor._instance.grad[p_id] = UnifiedCPUTensor._instance.pool_g[
                UnifiedCPUTensor._instance.pool_g_d:UnifiedCPUTensor._instance.pool_g_d + x].view(shape)
            UnifiedCPUTensor._instance.momentum[p_id] = UnifiedCPUTensor._instance.pool_m[
                UnifiedCPUTensor._instance.pool_m_d:UnifiedCPUTensor._instance.pool_m_d + x].view(shape)
            UnifiedCPUTensor._instance.variance[p_id] = UnifiedCPUTensor._instance.pool_v[
                UnifiedCPUTensor._instance.pool_v_d:UnifiedCPUTensor._instance.pool_v_d + x].view(shape)
            print(UnifiedCPUTensor._instance.pool_t_d,
                  UnifiedCPUTensor._instance.pool_t.numel())

            UnifiedCPUTensor._instance.pool_t_d += x
            UnifiedCPUTensor._instance.pool_g_d += x
            UnifiedCPUTensor._instance.pool_m_d += x
            UnifiedCPUTensor._instance.pool_v_d += x

            print(UnifiedCPUTensor._instance.pool_t_d,
                  UnifiedCPUTensor._instance.pool_t.numel())
            UnifiedCPUTensor._instance.counter[p_id] = 0
            UnifiedCPUTensor._instance.counter_grad[p_id] = 0
            UnifiedCPUTensor._instance.counter_momentum[p_id] = 0
            UnifiedCPUTensor._instance.counter_variance[p_id] = 0

            UnifiedCPUTensor._instance.tensors[p_id].grad_ = UnifiedCPUTensor._instance.grad[p_id]
            UnifiedCPUTensor._instance.tensors[p_id].momentum = UnifiedCPUTensor._instance.momentum[p_id]
            UnifiedCPUTensor._instance.tensors[p_id].variance = UnifiedCPUTensor._instance.variance[p_id]

        # print(UnifiedCPUTensor._instance.tensors[p_id].size())
        counter = UnifiedCPUTensor._instance.counter[p_id]

        ret = UnifiedCPUTensor._instance.tensors[p_id][counter: counter + count]
        UnifiedCPUTensor._instance.counter[p_id] += count
        return ret

    @staticmethod
    def get_grad(size, device_s_experts, count, dtype, p_id):

        counter = UnifiedCPUTensor._instance.counter_grad[p_id]

        ret = UnifiedCPUTensor._instance.grad[p_id][counter: counter + count]
        UnifiedCPUTensor._instance.counter_grad[p_id] += count
        return ret

    @staticmethod
    def get_momentum(size, device_s_experts, count, dtype, p_id):

        counter = UnifiedCPUTensor._instance.counter_momentum[p_id]

        ret = UnifiedCPUTensor._instance.momentum[p_id][counter: counter + count]
        UnifiedCPUTensor._instance.counter_momentum[p_id] += count
        return ret

    @staticmethod
    def get_variance(size, device_s_experts, count, dtype, p_id):

        counter = UnifiedCPUTensor._instance.counter_variance[p_id]

        ret = UnifiedCPUTensor._instance.variance[p_id][counter: counter + count]
        UnifiedCPUTensor._instance.counter_variance[p_id] += count
        return ret

    @staticmethod
    def get_cpu_experts():
        return [UnifiedCPUTensor._instance.pool_t] if UnifiedCPUTensor._instance.numel > 0 else []
        ret = []
        for key in UnifiedCPUTensor._instance.tensors.keys():
            ret.append(UnifiedCPUTensor._instance.tensors[key])
        return ret


class CPUHostBatchedParameter(object):
    def __init__(self, size, count, dtype, moe, p_id, bias=False, fan_in=0, transposed=False, device_s_experts=-1):
        self.count = count
        self.size = size
        self.dtype = dtype
        self.device = "cpu"
        self.p_id = p_id
        self.adam_state = True
        self.bias = bias

        if count > 0:
            if device_s_experts == -1:
                self.all_tensors = torch.rand(
                    (count, *list(size)), device="cpu", dtype=dtype, pin_memory=True)
            else:
                self.all_tensors = UnifiedCPUTensor.get_tensor(
                    size, device_s_experts, count, dtype, p_id)
            # print(self.all_tensors.size())

            if self.bias:
                for i in range(count):
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(
                        self.all_tensors[i], -bound, bound)
            else:
                for i in range(count):
                    torch.nn.init.kaiming_uniform_(self.all_tensors[i], a=math.sqrt(
                        5), mode="fan_out" if transposed else "fan_in")

            self.all_tensors.adam_state = self.adam_state
            if device_s_experts == -1:
                self.all_tensors.grad_ = torch.zeros(
                    (count, *list(size)), device="cpu", dtype=dtype, pin_memory=True)
                self.all_tensors.momentum = torch.zeros(
                    (count, *list(size)), device="cpu", dtype=dtype, pin_memory=True)
                self.all_tensors.variance = torch.zeros(
                    (count, *list(size)), device="cpu", dtype=dtype, pin_memory=True)
            else:
                self.all_tensors.grad_ = UnifiedCPUTensor.get_grad(
                    size, device_s_experts, count, dtype, p_id)
                self.all_tensors.momentum = UnifiedCPUTensor.get_momentum(
                    size, device_s_experts, count, dtype, p_id)
                self.all_tensors.variance = UnifiedCPUTensor.get_variance(
                    size, device_s_experts, count, dtype, p_id)
            self.all_tensors.grad_.lazy_zero = True

            moe.set_CPUParamCore(p_id, self.all_tensors, self.all_tensors.grad_,
                                 self.all_tensors.momentum, self.all_tensors.variance, self.bias, self.p_id)
            self.moe = moe
#            self.core = CPUParamCore(self.all_tensors, self.all_tensors.grad_, self.all_tensors.momentum, self.all_tensors.variance, self.bias, self.p_id)
        else:
            self.all_tensors = None

    def fetch_single(self, id):
        return self.moe.CPUParamCore_fetch_single(self.p_id, id)
        return self.core.fetch_single(id)
        return self.all_tensors[id, :]

    def fetch_single_grad(self, id):
        return self.moe.CPUParamCore_fetch_single_grad(self.p_id, id)
        return self.core.fetch_single_grad(id)
        # print(self.grads_offset)
        return self.all_tensors.grad_[id, :]

    def fetch_batch(self, view_base, start, end):
        return self.moe.CPUParamCore_fetch_batch(self.p_id, view_base, start, end)
        return self.core.fetch_batch(view_base, start, end)
      #  print("cpu fetch batch: ", start, end)
        ret = self.all_tensors[start:end, :]
        return ParamProxy(True, ret, self.p_id, view_base, view_base + end - start, self.bias)

    def fetch_batch_vari(self, view_base, start, end):
        return self.moe.CPUParamCore_fetch_batch_vari(self.p_id, view_base, start, end)
        return self.core.fetch_batch_vari(view_base, start, end)
        # if torch.distributed.get_rank() == 0:
        #    print("cpu vari fetch batch: ", start, end)
        ret = self.all_tensors.variance[start:end, :]
        return ParamProxy(True, ret, self.p_id, view_base, view_base + end - start, self.bias)

    def fetch_batch_mome(self, view_base, start, end):
        return self.moe.CPUParamCore_fetch_batch_mome(self.p_id, view_base, start, end)
        return self.core.fetch_batch_mome(view_base, start, end)
      #  print("cpu fetch batch: ", start, end)
        ret = self.all_tensors.momentum[start:end, :]
        return ParamProxy(True, ret, self.p_id, view_base, view_base + end - start, self.bias)

    def fetch_batch_grad(self, view_base, start, end):
        return self.moe.CPUParamCore_fetch_batch_grad(self.p_id, view_base, start, end)
        return self.core.fetch_batch_grad(view_base, start, end)
      #  print("cpu fetch batch: ", start, end)
        ret = self.all_tensors.grad_[start:end, :]
        return ParamProxy(True, ret, self.p_id, 0, end - start, self.bias)

    def prefetch_to(self, start, batch_size, gpu_param):
        return self.moe.CPUParamCore_prefetch_to(self.p_id, start, batch_size, gpu_param)
        return self.core.prefetch_to(start, batch_size, gpu_param)
        tmp = self.all_tensors[start:start+batch_size, :]
        cpufp32_to_gpufp32(tmp, gpu_param, tmp.numel())

     #   with torch.no_grad():
      #      gpu_param.copy_(tmp)
       #     torch.cuda.synchronize()

        #    print("prefetch to: ", tmp.numel(), start, batch_size, gpu_param.data_ptr(), gpu_param[0] - tmp[0].cuda())
        return gpu_param

    def unset_lazy_zero(self):
        if self.count > 0:
            self.moe.CPUParamCore_unset_lazy_zero(self.p_id)
#            self.CPUParamCore.unset_lazy_zero()
#            self.all_tensors.grad_.lazy_zero = False

    def store_batch_grad(self, start, end, grad):
        self.moe.CPUParamCore_store_batch_grad(self.p_id, start, end, grad)
#        self.core.store_batch_grad(start, end,grad)
        return
#
#  self.all_tensors.grad[start:end,:].copy_(grad)
        #print("store size: ", self.all_tensors.grad_.size(), start, end)
        gpufp32_to_cpufp32(grad, self.all_tensors.grad_[
                           start:end, :], grad.numel())

    def store_batch_para(self, start, end, para):
        self.moe.CPUParamCore_store_batch_para(self.p_id, start, end, para)
#        self.core.store_batch_para(start, end, para)
        return
#        self.all_tensors.grad[start:end,:].copy_(grad)
        #print("store size: ", self.all_tensors.grad_.size(), start, end)
        gpufp32_to_cpufp32(para, self.all_tensors[start:end, :], para.numel())

    def store_batch_mome(self, start, end, mome):
        self.moe.CPUParamCore_store_batch_mome(self.p_id, start, end, mome)
#        self.core.store_batch_mome(start, end, mome)
        return
#        self.all_tensors.grad[start:end,:].copy_(grad)
        #print("store size: ", self.all_tensors.grad_.size(), start, end)
        gpufp32_to_cpufp32(
            mome, self.all_tensors.momentum[start:end, :], mome.numel())

    def store_batch_vari(self, start, end, vari):
        self.moe.CPUParamCore_store_batch_vari(self.p_id, start, end, vari)
#        self.core.store_batch_vari(start, end, vari)
        return
#        self.all_tensors.grad[start:end,:].copy_(grad)
        #print("store size: ", self.all_tensors.grad_.size(), start, end)
        gpufp32_to_cpufp32(
            vari, self.all_tensors.variance[start:end, :], vari.numel())

    def acc_store_batch_grad(self, start, end, grad):
        # NOTION this need to be modified

        if self.all_tensors.grad_.lazy_zero:
            self.moe.CPUParamCore_set_lazy_zero(self.p_id)
            # self.core.set_lazy_zero()
            #gpufp32_to_cpufp32(grad, self.all_tensors.grad_[start:end,:], grad.numel())
        else:
            self.moe.CPUParamCore_unset_lazy_zero(self.p_id)
#            self.core.unset_lazy_zero()
            #gpufp32_addto_cpufp32(grad, self.all_tensors.grad_[start:end,:], grad.numel())
        self.moe.CPUParamCore_acc_store_batch_grad(self.p_id, start, end, grad)
      #  self.core.acc_store_batch_grad(start, end, grad)
    '''
    def store_variance(self, start, variance):
        #print(self.all_tensors.variance[start:start + 1,:].numel(), variance.numel())
        self.core.store_variance(start, variance)
        return 
        gpufp32_to_cpufp32(variance, self.all_tensors.variance[start:start + 1,:], variance.numel()) 

    def store_momentum(self, start, momentum):
        #print(self.all_tensors.momentum[start:start + 1,:].numel(), momentum.numel())
        self.core.store_momentum(start, momentum)
        return 
        gpufp32_to_cpufp32(momentum, self.all_tensors.momentum[start:start + 1,:], momentum.numel()) 

    def store_weight(self, start, param):
        self.core.store_weight(start, param)
        return 
        #print(self.all_tensors[start:start + 1,:].numel(), param.numel())
        gpufp32_to_cpufp32(param, self.all_tensors[start:start + 1,:], param.numel()) 

    def fetch_variance(self, start, variance):
        self.core.fetch_variance(start, variance)
        return 
        cpufp32_to_gpufp32(self.all_tensors.variance[start:start + 1,:], variance, variance.numel()) 

    def fetch_momentum(self, start, momentum):
        self.core.fetch_momentum(start, momentum)
        return
        cpufp32_to_gpufp32(self.all_tensors.momentum[start:start + 1,:], momentum, momentum.numel()) 
    '''


class GPUHostBatchedParameter(object):
    def __init__(self, size, count, dtype, device, moe, p_id, bias=False, fan_in=0, transposed=False):
        self.count = count
        self.size = size
        self.dtype = dtype
        self.device = device
        self.grads = []
        self.grads_batch_size = []
        self.grads_offset = []
        self.bias = bias

        if count > 0:
            self.all_tensors = torch.rand(
                (count, *list(size)), device=device, dtype=dtype).contiguous()
            self.grad_tensors = torch.zeros(
                (count, *list(size)), device=device, dtype=dtype).contiguous()

            if self.bias:
                for i in range(count):
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(self.all_tensors[i], -bound, bound)
            else:
                for i in range(count):
                    # print(self.all_tensors.size())
                    torch.nn.init.kaiming_uniform_(self.all_tensors[i], a=math.sqrt(
                        5), mode="fan_out" if transposed else "fan_in")
            self.all_tensors.grad = self.grad_tensors
            self.moe = moe
            self.p_id = p_id
            moe.set_GPUParamCore(p_id, self.all_tensors,
                                 self.grad_tensors, self.bias)
            #self.core = GPUParamCore(self.all_tensors, self.grad_tensors, self.bias)

        else:
            self.all_tensors = 0
        self.lazy_zero_grad = True

    def fetch_single(self, id):
        return self.moe.GPUParamCore_fetch_single(self.p_id, id)

        return self.core.fetch_single(id)
        return self.all_tensors[id, :]

    def fetch_single_grad(self, id):
        return self.moe.GPUParamCore_fetch_single_grad(self.p_id, id)
        return self.core.fetch_single_grad(id)
        # print(self.grads_offset)
        return self.grad_tensors[id, :]

    def fetch_batch(self, start, end):
        '''
        if self.bias:
            ret = self.all_tensors[start:end,:]
            ret.grad_ = self.grad_tensors[start:end,:].unsqueeze(dim=1)
        else:
        '''
        return self.moe.GPUParamCore_fetch_batch(self.p_id, start, end)
        ret = self.moe.GPUParamCore_fetch_batch(self.p_id, start, end)
        ret.get_param().grad = self.grad_tensors[start:end, :]
        return ret
        ret = self.all_tensors[start:end, :]
        ret.grad_ = self.grad_tensors[start:end, :]
        return ParamProxy(False, ret, -1,  -1, -1, self.bias)
        return ParamProxy(False, ret, bias=self.bias)

    def fetch_all_with_grad(self):
        ret = self.all_tensors
        ret.grad = self.grad_tensors
        return ret
