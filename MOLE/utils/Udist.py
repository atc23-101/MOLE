import torch
import torch.distributed as dist
from UniMoE.core import MOLEAll2Allfp32, ensure_nccl


class MOLEDist:
    _instance = None 

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

            if dist.is_initialized():
                cls._instance.world_size = dist.get_world_size()
                cls._instance.use_distributed = True 
                cls._instance.rank = dist.get_rank()
                cls._instance.local_rank = kw["local_rank"]
                cls._instance.device = torch.device("cuda", cls._instance.local_rank)
                cls._instance.MOLE_group = dist.new_group(ranks=kw.get("MOLE_group_ranks", [i for i in range(cls._instance.world_size)]))


                ensure_nccl(cls._instance.MOLE_group, cls._instance.local_rank)
            else:
                cls._instance.world_size = 1
                cls._instance.use_distributed = False
                cls._instance.rank = 0
                cls._instance.local_rank = 0
                cls._instance.device = torch.device("cuda", cls._instance.local_rank)
                cls._instance.MOLE_group = None

        return cls._instance

    @staticmethod
    def enabled():
        return MOLEDist._instance.use_distributed

    @staticmethod
    def all2all(rec_lst, inp_lst):
        instance = MOLEDist._instance
        MOLEAll2Allfp32(inp_lst, rec_lst, instance.local_rank, instance.world_size, -1, torch.cuda.current_stream()._cdata, "MOLEG", instance.rank)

        #print("ALL2ALL :" ,torch.distributed.get_rank(), torch.cuda.current_stream()._cdata)
        #dist.all_to_all(rec_lst, inp_lst, MOLEDist._instance.MOLE_group)

    @staticmethod
    def all2allsingle(rec_lst, inp_lst):
        #print("A2A: ", rec_lst.size(), inp_lst.size())
        #print("ALL2ALL :" ,torch.distributed.get_rank(), torch.cuda.current_stream()._cdata)
        dist.all_to_all_single(rec_lst, inp_lst, group=MOLEDist._instance.MOLE_group)

    @staticmethod
    def get_MOLE_world_size():
        return MOLEDist._instance.world_size
    
    @staticmethod
    def get_MOLE_local_rank():
        return MOLEDist._instance.local_rank
    
    @staticmethod
    def get_MOLE_rank():
        return MOLEDist._instance.rank
    
    @staticmethod
    def get_MOLE_device():
        return MOLEDist._instance.device