import torch
from UniMoE.moe import MOLE
from UniMoE.optim import mixed_optim
from UniMoE.utils import Udist
from UniMoE.utils import Ustream
import UniMoE.core as MOLE_util
from UniMoE.utils import Utimer
from UniMoE.utils.Uparam import UnifiedCPUTensor
import math


class UniScheduler:
    _instance = None
    _optimizer = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            print("#!#building UniMoE Scheduler: ")
            cls._instance = object.__new__(cls)

            Ustream.ExpertStreamCtx(
                device=f"cuda:{kw['local_rank']}", disable=kw["disable_overlap"])
            Udist.MOLEDist(local_rank=kw["local_rank"])

            cls._instance.device = f"cuda:{kw['local_rank']}"
            cls._instance.adv_adam = kw["adv_adam"]
            cls._instance.hybrid_adam = kw.get("hybrid_adam", False)
            if cls._instance.hybrid_adam:
                cls._instance.adv_adam = True
                cls._instance.hybrid_adam_extra = []

            if cls._instance.adv_adam:
                cls._instance.hybrid_adam_count = -1
            else:
                cls._instance.hybrid_adam_count = 0

            cls._instance.gate_control_by_MOLE = kw["gate_control_by_MOLE"]

            cls._instance.ftime = None
            cls._instance.fwd_stat = None
            cls._instance.bwd_stat = None
            cls._instance.cur_step = -1
            cls._instance.prefetch_on = False
            cls._instance.profile_step = -1
            cls._instance.global_prefetch = False
            cls._instance.deactivate_first_prefetch = True

            cls._instance.device_s_experts = kw.get("device_s_experts", -1)

            assert cls._instance.hybrid_adam == False or cls._instance.device_s_experts == - \
                1, "NOT SUPPORT HYB ADAM and CONTIG CPU MEM"
        return cls._instance

    def __init__(self, *args, **kw):
        pass

    @staticmethod
    def schedule_policy(profile_step, prefetch_on, deactivate_first_prefetch=True,prefetch_floor=False, global_prefetch=False, schedule_hybrid=False):
        UniScheduler._instance.profile_step = profile_step
        UniScheduler._instance.prefetch_on = prefetch_on
        UniScheduler._instance.deactivate_first_prefetch = deactivate_first_prefetch
        UniScheduler._instance.global_prefetch = global_prefetch
        UniScheduler._instance.schedule_hybrid = schedule_hybrid
        UniScheduler._instance.prefetch_floor = prefetch_floor

    @staticmethod
    def save_states(state_file="MOLE_states", grad=False):
        all_state = []
        model_state = []
        for instance in MOLE.MOLEModule._instance:
            model_state.append(instance.get_layer_state(grad))
        optim_state = UniScheduler._optimizer.get_optim_state()
        all_state.append(model_state)
        all_state.append(optim_state)
        torch.save(all_state, state_file + ".pt")

    # MOLE API does not promise the continuous of the optimizer param_groups
    @staticmethod
    def load_states(state_file="MOLE_states", grad=False):
        all_state = torch.load(state_file + ".pt")
        optim_state = all_state[1]
        model_state = all_state[0]

        # first set, setting grads
        for i, instance in enumerate(MOLE.MOLEModule._instance):
            instance.set_layer_state(model_state[i], grad)

        UniScheduler._optimizer.set_optim_state(optim_state)
        # optimizer must be called before model

        for i, instance in enumerate(MOLE.MOLEModule._instance):
            instance.set_layer_state(model_state[i], grad)

    @staticmethod
    def build_optimizer(lr=1e-3,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0, gate_only=False, adamW=False):

        UniScheduler._instance.adamW = adamW

        if not gate_only:
            cpu_params = []
            gpu_params = []
            gate_params = []

            for instance in MOLE.MOLEModule._instance:
                if UniScheduler._instance.device_s_experts == -1:
                    cpu_params += instance.d_experts_param()
                gpu_params += instance.s_experts_param()
                if UniScheduler._instance.gate_control_by_MOLE:
                    gate_params += instance.gate_param()
            if UniScheduler._instance.device_s_experts != -1:
                cpu_params += UnifiedCPUTensor.get_cpu_experts()
            if len(cpu_params) > 0 or len(gpu_params) > 0 or len(gate_params) > 0:
                UniScheduler._optimizer = mixed_optim.MOLEMixedAdam(
                    cpu_params, gpu_params, gate_params, lr, betas, eps, weight_decay, adamW)
            else:
                UniScheduler._optimizer = None
        else:
            gate_params = []
            for instance in MOLE.MOLEModule._instance:
                gate_params += instance.gate_param()
            UniScheduler._optimizer = mixed_optim.MOLEMixedAdam(
                [], [], gate_params, lr, betas, eps, weight_decay, adamW)

        if UniScheduler._optimizer.hasCPU:
            MOLE.MOLEModule._cpu_optimizer = UniScheduler._optimizer.cpu_optimizer
        UniScheduler.adam_set()
        return UniScheduler._optimizer

    @staticmethod
    def clear():
        MOLE_util.clear_virtual_mem()
#        Umemory.UniMemPool.clear_virtual_mem()

    @staticmethod
    def set_hybrid(hybrid_adam_count):
        if UniScheduler._instance.hybrid_adam:
            UniScheduler._instance.hybrid_adam_count = hybrid_adam_count

    @staticmethod
    def adam_set():
        if UniScheduler._optimizer.hasCPU and UniScheduler._instance.adv_adam:
            MOLE.MOLEModule._cpu_optimizer.adam_setting(
                MOLE.MOLEModule._instance, UniScheduler._instance.adamW, UniScheduler._instance.hybrid_adam_count, UniScheduler._instance.hybrid_adam)

    @staticmethod
    def zero_grad():
        if UniScheduler._optimizer.hasCPU:
            for inst in MOLE.MOLEModule._instance:
                inst.MOLECore.set_lazy_zero()

    @staticmethod
    def clear_timer():
        Utimer.UATimer.clear()

    @staticmethod
    def track_fwd_start():
        Utimer.UATimer.track_fwd_start()

    # set step at the begining of every step
    @staticmethod
    def set_step(cur_step):
        UniScheduler._instance.cur_step = cur_step
        if UniScheduler.time_it():
            UniScheduler.clear_timer()
            UniScheduler.track_fwd_start()
        UniScheduler.clear()

    @staticmethod
    def time_it():

        if UniScheduler._instance.cur_step < UniScheduler._instance.profile_step:
            return True

        return False

    @staticmethod
    def prefetch_it():
        if UniScheduler._instance.cur_step > UniScheduler._instance.profile_step and UniScheduler._instance.prefetch_on:
            return True

        return False

    @staticmethod
    def dummy_fetch():
        fetch_time = []
        for i in range(len(MOLE.MOLEModule._instance)):
            fetch_time.append(MOLE.MOLEModule._instance[i].dummy_fetch(1))
        return fetch_time

    @staticmethod
    def attach_step_time(step_time, cur_step):
        if UniScheduler._instance.schedule_hybrid:
            if cur_step - 10 >= 0 and cur_step - 10 <= MOLE.MOLEModule._instance[0].d_experts:
                UniScheduler._instance.hybrid_adam_extra.append(step_time)

    @staticmethod
    def schedule_hybrid_adam(cur_step):
        if UniScheduler._instance.schedule_hybrid:
            if cur_step - 10 >= 0 and cur_step - 10 <= MOLE.MOLEModule._instance[0].d_experts:
                UniScheduler.set_hybrid(cur_step - 10)
            if cur_step == MOLE.MOLEModule._instance[0].d_experts + 11:
                min_ = 100000
                min_idx = -1
                for i in range(len(UniScheduler._instance.hybrid_adam_extra)):
                    tmp = UniScheduler._instance.hybrid_adam_extra[i][0]

                    if Udist.MOLEDist.get_MOLE_world_size() > 1:
                        tmptensor = torch.tensor(tmp).to(
                            UniScheduler._instance.device)

                        torch.distributed.all_reduce(
                            tmptensor, op=torch.distributed.ReduceOp.AVG)
                        tmp = tmptensor.item()

                    if tmp < min_:
                        min_ = tmp
                        min_idx = i
                torch.save(
                    [UniScheduler._instance.hybrid_adam_extra, min_idx], "hybrid.pt")
                UniScheduler.set_hybrid(min_idx)
                print(UniScheduler._instance.hybrid_adam_extra)
                print("SELECT: ", min_idx)

    @staticmethod
    def profile_prefetch(force_prefetch_scheme):
        if UniScheduler.time_it():
            if UniScheduler._instance.cur_step < 2:
                return
            fetch_time = UniScheduler.dummy_fetch()
            ftime = sum(fetch_time) / len(fetch_time)
            if UniScheduler._instance.ftime is None:
                UniScheduler._instance.ftime = ftime
            else:
                UniScheduler._instance.ftime = (
                    UniScheduler._instance.ftime + ftime) / 2
            fwd_stat, bwd_stat = Utimer.UATimer.stats()
            # print("STAT: ", fwd_stat, bwd_stat,
            #      UniScheduler._instance.fwd_stat, UniScheduler._instance.bwd_stat)
            if UniScheduler._instance.fwd_stat is None and len(fwd_stat) > 0:
                UniScheduler._instance.fwd_stat = fwd_stat
            elif len(fwd_stat) > 0:
                UniScheduler._instance.fwd_stat = [
                    (x[0] + x[1]) / 2 for x in zip(UniScheduler._instance.fwd_stat, fwd_stat)]

            if UniScheduler._instance.bwd_stat is None and len(bwd_stat) > 0:
                UniScheduler._instance.bwd_stat = bwd_stat
            elif len(bwd_stat) > 0:
                UniScheduler._instance.bwd_stat = [
                    (x[0] + x[1]) / 2 for x in zip(UniScheduler._instance.bwd_stat, bwd_stat)]

            print("FWD: ", UniScheduler._instance.fwd_stat)
            print("BWD: ", UniScheduler._instance.bwd_stat)
            print("FETCH: ", UniScheduler._instance.ftime)
            if UniScheduler._instance.prefetch_on:
                UniScheduler.set_prefetch_scheme(force_prefetch_scheme)

    @staticmethod
    def set_prefetch_scheme(force_prefetch_scheme):
        instance = UniScheduler._instance
        if instance.prefetch_floor:
            instance.fwd_pf = [
                math.floor(f / instance.ftime) for f in instance.fwd_stat]
            instance.bwd_pf = [
                math.floor(b / instance.ftime) for b in instance.bwd_stat]
        else:
            instance.fwd_pf = [
                math.ceil(f / instance.ftime) for f in instance.fwd_stat]
            instance.bwd_pf = [
                math.ceil(b / instance.ftime) for b in instance.bwd_stat]

        if force_prefetch_scheme != "":  # often list: [[1,1,1,1,1],[1,1,1,1,]]
            force_prefetch_scheme = eval(force_prefetch_scheme)
            instance.global_prefetch = False
            instance.fwd_pf = force_prefetch_scheme[0]
            instance.bwd_pf = force_prefetch_scheme[1]

        if instance.deactivate_first_prefetch:
            instance.fwd_pf[0] = 0

        instance.E = [MOLE.MOLEModule._instance[i].d_experts for i in range(
            len(MOLE.MOLEModule._instance))]
        instance.revE = instance.E[::-1]
        if Udist.MOLEDist.get_MOLE_world_size() > 1:
            F_fwd_tensor = torch.tensor(instance.fwd_pf).to(
                instance.device)
            F_bwd_tensor = torch.tensor(instance.bwd_pf).to(
                instance.device)

            torch.distributed.all_reduce(
                F_fwd_tensor, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(
                F_bwd_tensor, op=torch.distributed.ReduceOp.MIN)

            instance.fwd_pf = [int(i.item()) for i in list(F_fwd_tensor)]
            instance.bwd_pf = [int(i.item()) for i in list(F_bwd_tensor)]

        pool_size = max(0, MOLE.MOLEModule._instance[0].pool_size)
        fwd = 0
        bwd = 0
        if not instance.global_prefetch:
            for i in range(len(MOLE.MOLEModule._instance)):
                MOLE.MOLEModule._instance[i].global_prefetch = False

                MOLE.MOLEModule._instance[i].local_prefetch_fwd_count = min(
                    instance.fwd_pf[i], MOLE.MOLEModule._instance[i].d_experts, pool_size)
                MOLE.MOLEModule._instance[i].local_prefetch_bwd_count = min(
                    instance.bwd_pf[len(MOLE.MOLEModule._instance) - 1 - i], MOLE.MOLEModule._instance[i].d_experts, pool_size)
                if torch.distributed.get_rank() == 0:
                    print(f"Layer {i}", MOLE.MOLEModule._instance[i].local_prefetch_fwd_count, MOLE.MOLEModule._instance[i].local_prefetch_bwd_count,
                          MOLE.MOLEModule._instance[i].global_prefetch_fwd_count, MOLE.MOLEModule._instance[i].global_prefetch_bwd_count)
                    fwd += (MOLE.MOLEModule._instance[i].local_prefetch_fwd_count + MOLE.MOLEModule._instance[i].global_prefetch_fwd_count)
                    bwd += (MOLE.MOLEModule._instance[i].local_prefetch_bwd_count + MOLE.MOLEModule._instance[i].global_prefetch_bwd_count)

        else:
            GPscheme_fwd, LPscheme_fwd = MOLE_util.fast_assign_prefetch(
                pool_size, instance.fwd_pf, instance.E)
            GPscheme_bwd, LPscheme_bwd = MOLE_util.fast_assign_prefetch(
                pool_size, instance.bwd_pf, instance.revE)

            if torch.distributed.get_rank() == 0:
                print("Policy: ")
                print(GPscheme_fwd, LPscheme_fwd)
                print(GPscheme_bwd, LPscheme_bwd)
            for i in range(len(MOLE.MOLEModule._instance)):
                MOLE.MOLEModule._instance[i].global_prefetch = True
                MOLE.MOLEModule._instance[i].local_prefetch_fwd_count = LPscheme_fwd[i]
                MOLE.MOLEModule._instance[i].local_prefetch_bwd_count = LPscheme_bwd[len(
                    MOLE.MOLEModule._instance) - 1 - i]
                MOLE.MOLEModule._instance[i].global_prefetch_fwd_count = sorted(
                    GPscheme_fwd[i], reverse=True)
                MOLE.MOLEModule._instance[i].global_prefetch_bwd_count = sorted([[len(
                    MOLE.MOLEModule._instance) - 1 - x[0], x[1]]for x in GPscheme_bwd[len(
                        MOLE.MOLEModule._instance) - 1 - i]])
                if torch.distributed.get_rank() == 0:
                    print(f"Layer {i}", MOLE.MOLEModule._instance[i].local_prefetch_fwd_count, MOLE.MOLEModule._instance[i].local_prefetch_bwd_count,
                          MOLE.MOLEModule._instance[i].global_prefetch_fwd_count, MOLE.MOLEModule._instance[i].global_prefetch_bwd_count)
                    fwd += (MOLE.MOLEModule._instance[i].local_prefetch_fwd_count + sum(x[1] for x in MOLE.MOLEModule._instance[i].global_prefetch_fwd_count))
                    bwd += (MOLE.MOLEModule._instance[i].local_prefetch_bwd_count + sum(x[1] for x in MOLE.MOLEModule._instance[i].global_prefetch_bwd_count))

        print("PREFETCH SET: ", fwd, bwd)
    @ staticmethod
    def prefetch_fwd():
        if UniScheduler.prefetch_it():
            if UniScheduler._instance.global_prefetch:
                MOLE.MOLEModule._instance[0].global_prefetch_fwd()
            MOLE.MOLEModule._instance[0].local_prefetch_fwd()

    @staticmethod
    def sync_tran():
        torch.cuda.synchronize()
        for ins in MOLE.MOLEModule._instance:
            ins.MOLECore.clear_tran_tmp()
