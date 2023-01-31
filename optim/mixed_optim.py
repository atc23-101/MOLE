# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# adapted from fairseq

from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
from torch import optim
import time
from copy import deepcopy
import math
import torch
from torch import Tensor
from typing import List, Optional
from UniMoE.core import synchronize_tran

def _get_cpu_adam():
    try:
        from deepspeed.ops.op_builder import CPUAdamBuilder

        return CPUAdamBuilder().load()
    except ImportError:
        # fbcode
        from deepspeed.ops.adam import DeepSpeedCPUAdam as ds_opt_adam

        return ds_opt_adam


class CPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adamw_mode=True,
    ):
        print("build optimizer: ", len(params))
        default = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "adamw_mode": adamw_mode
        }
        super().__init__(params, default)

        self.opt_id = CPUAdam.optimizer_id
        CPUAdam.optimizer_id = CPUAdam.optimizer_id + 1

        self.ds_opt_adam = _get_cpu_adam()
        self.ds_opt_adam.create_adam(
            self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, False)

        self.adv_adam = False
        self.hybrid_adam = False
        self.hybrid_adam_count = 0

    @torch.no_grad()
    def zero_grad(self):
        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group["params"]):
                if p.grad_ is None:
                    continue
                else:
                    p.grad_.lazy_zero = True
        return

    @torch.no_grad()
    def adam_setting(self, moes, adamW, hybrid_adam_count, hybrid_adam):
        self.hybrid_adam_count = hybrid_adam_count
        self.hybrid_adam = hybrid_adam
        self.adv_adam = True

        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param_id, p in enumerate(group["params"]):
                if p.grad_ is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    dtype = p.data.dtype
                    state["exp_avg"] = p.momentum
                    state["exp_avg_sq"] = p.variance
                step = state["step"]
        # return beta1, beta2, step, lr, wd, eps
        for moe in moes:
            moe.MOLECore.set_adam(beta1, beta2, step, lr,
                                  wd, eps, adamW, hybrid_adam_count)

    @torch.no_grad()
    def step(self, closure=None, groups=None, scale=1.0):
        if self.adv_adam and (not self.hybrid_adam or self.hybrid_adam_count == -1):
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group["params"]):
                    if p.grad_ is None:
                        continue
                    state = self.state[p]
                    state["step"] += 1
            return

      #  torch.cuda.synchronize()
        count_param = 0
        szs = []
        time0 = time.time()
        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group["params"]):
                if p.grad_ is None:
                    continue
                state = self.state[p]
                # print(self.state.keys())
                count_param += 1
                if len(state) == 0:
                    state["step"] = 0
                    dtype = p.data.dtype

                    state["exp_avg"] = p.momentum
                    state["exp_avg_sq"] = p.variance

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                beta1, beta2 = group["betas"]
                #print("STEP: ", self.hybrid_adam_count, p.data.size(), p.data[self.hybrid_adam_count:].size())
                if self.hybrid_adam_count < p.data.size()[0]:
                    # szs.append(p.data[self.hybrid_adam_count:])
                    self.ds_opt_adam.adam_update(
                        self.opt_id,
                        state["step"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                        group["bias_correction"],
                        p.data[self.hybrid_adam_count:].view(-1),
                        p.grad_[self.hybrid_adam_count:].data.view(-1),
                        exp_avg[self.hybrid_adam_count:],
                        exp_avg_sq[self.hybrid_adam_count:],
                    )
        time1 = time.time()
        print("cpu adam: ", time1 - time0, count_param, szs)


class MOLEMixedAdam:
    def __init__(self, CPUParams, GPUParams, GateParams,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 adamW=False,
                 ):

        self.GateParams = GateParams
        self.CPUParams = CPUParams
        self.GPUParams = GPUParams
        self.hasCPU = False
        self.hasGPU = False
        self.hasGate = False
        if len(CPUParams) > 0 and CPUParams[0] is not None:
            self.hasCPU = True

            self.cpu_optimizer = CPUAdam(
                CPUParams,
                lr=lr,
                bias_correction=True,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adamw_mode=adamW)

        if len(GPUParams) > 0:
            self.hasGPU = True
            if adamW:
                self.gpu_optimizer = torch.optim.AdamW(
                    params=GPUParams, lr=lr,
                    betas=betas, eps=eps, weight_decay=weight_decay
                )
            else:
                self.gpu_optimizer = torch.optim.Adam(
                    params=GPUParams, lr=lr,
                    betas=betas, eps=eps, weight_decay=weight_decay
                )
        if len(GateParams) > 0:
            print("GATE")
            self.hasGate = True
            if adamW:
                self.gate_optimizer = torch.optim.AdamW(
                    params=GateParams, lr=lr,
                    betas=betas, eps=eps, weight_decay=weight_decay
                )
            else:
                self.gate_optimizer = torch.optim.Adam(
                    params=GateParams, lr=lr,
                    betas=betas, eps=eps, weight_decay=weight_decay
                )

    @torch.no_grad()
    def set_optim_state(self, states):
        # build essential parts by calling step
        self.step()

        gate_state = states[1]
        expert_state = states[0]
        if self.hasGate:
            gate_cur = self.gate_optimizer.state_dict()["state"]
            for k in gate_cur.keys():
                gate_cur[k]['exp_avg'].copy_(gate_state[k]['exp_avg'])
                gate_cur[k]['exp_avg_sq'].copy_(gate_state[k]['exp_avg_sq'])
                gate_cur[k]["step"].copy_(gate_state[k]["step"])
                #print("gtSTEPs: ", gate_cur[k]["step"])
        if self.hasGPU:
            gpu_cur = self.gpu_optimizer.state_dict()["state"]
            for k in gpu_cur.keys():
                shape_dim = gpu_cur[k]['exp_avg'].size()[0]
                gpu_cur[k]['exp_avg'].copy_(
                    expert_state[k]['exp_avg'][:shape_dim, :])
                gpu_cur[k]['exp_avg_sq'].copy_(
                    expert_state[k]['exp_avg_sq'][:shape_dim, :])
                gpu_cur[k]["step"].copy_(expert_state[k]["step"])
                #print("gSTEPs: ", gpu_cur[k]["step"])

        if self.hasCPU:
            cpu_cur = self.cpu_optimizer.state_dict()["state"]
            for k in cpu_cur.keys():
                shape_dim = cpu_cur[k]['exp_avg'].size()[0]
                cpu_cur[k]['exp_avg'].copy_(
                    expert_state[k]['exp_avg'][-shape_dim:, :])
                cpu_cur[k]['exp_avg_sq'].copy_(
                    expert_state[k]['exp_avg_sq'][-shape_dim:, :])
                cpu_cur[k]["step"] = expert_state[k]["step"]
                #print("cSTEPs: ", cpu_cur[k]["step"])

    @torch.no_grad()
    def get_optim_state(self):
        gate_state = None
        expert_state = None
        if self.hasGate:
            gate_state = deepcopy(self.gate_optimizer.state_dict()["state"])
            for k in gate_state.keys():
                gate_state[k]['exp_avg'] = gate_state[k]['exp_avg'].cpu()
                gate_state[k]['exp_avg_sq'] = gate_state[k]['exp_avg_sq'].cpu()
        if self.hasGPU:
            gpu_state = deepcopy(self.gpu_optimizer.state_dict()["state"])
            for k in gpu_state.keys():
                gpu_state[k]['exp_avg'] = gpu_state[k]['exp_avg'].cpu()
                gpu_state[k]['exp_avg_sq'] = gpu_state[k]['exp_avg_sq'].cpu()
        if self.hasCPU:
            cpu_state = deepcopy(self.cpu_optimizer.state_dict()["state"])
            for k in cpu_state.keys():
                cpu_state[k]['exp_avg'] = cpu_state[k]['exp_avg'].cpu()
                cpu_state[k]['exp_avg_sq'] = cpu_state[k]['exp_avg_sq'].cpu()
        if self.hasGPU and self.hasCPU:
            assert sorted(cpu_state.keys()) == sorted(
                gpu_state.keys()), "should be Four"
            expert_state = gpu_state
            for k in expert_state.keys():
                expert_state[k]['exp_avg'] = torch.cat(
                    [gpu_state[k]['exp_avg'], cpu_state[k]['exp_avg']], dim=0)
                expert_state[k]['exp_avg_sq'] = torch.cat(
                    [gpu_state[k]['exp_avg_sq'], cpu_state[k]['exp_avg_sq']], dim=0)
        elif self.hasGPU:
            expert_state = gpu_state
        elif self.hasCPU:
            expert_state = cpu_state

        return [expert_state, gate_state]

    @torch.no_grad()
    def zero_grad(self):
        if self.hasGate:
            self.gate_optimizer.zero_grad()
        if self.hasGPU:
            self.gpu_optimizer.zero_grad()
        if self.hasCPU:
            self.cpu_optimizer.zero_grad()

    def get_lr(self):
        lrs = []
        if self.hasGPU:
            for group in self.gpu_optimizer.param_groups:
                lrs.append(group['lr'])

        if self.hasCPU:
            for group in self.cpu_optimizer.param_groups:
                lrs.append(group['lr'])

        if self.hasGate:
            for group in self.gate_optimizer.param_groups:
                lrs.append(group['lr'])
        print("lrates ", lrs)

    def set_lr(self, lr):
        #        print("learning rate: ", lr)
        if self.hasGPU:
            for group in self.gpu_optimizer.param_groups:
                group['lr'] = lr[0] if type(lr) == list else lr

        if self.hasCPU:
            for group in self.cpu_optimizer.param_groups:
                group['lr'] = lr[0] if type(lr) == list else lr

        if self.hasGate:
            for group in self.gate_optimizer.param_groups:
                group['lr'] = lr[0] if type(lr) == list else lr

    @torch.no_grad()
    def step(self, CPU_step=True, GPU_step=True, Gate_step=True, AllreduceGate=True):
        #torch.cuda.synchronize()
        
        #time0 = time.time()
        if self.hasGate and Gate_step:
            #print("Gate Step")
            self.gate_optimizer.step()

            if AllreduceGate:
                for p in self.GateParams:
                    torch.distributed.all_reduce(
                        p, op=torch.distributed.ReduceOp.AVG)
        if self.hasGPU and GPU_step:
            #print("GPU Step")
            self.gpu_optimizer.step()

        if self.hasCPU and CPU_step:
            #print("CPU Step")
            # torch.cuda.synchronize()
            # default_wait_tran()
            #synchronize_tran()
            self.cpu_optimizer.step()

        # torch.cuda.synchronize()
        #time1 = time.time()
        #print("STEP TIME: ", time1 - time0)
