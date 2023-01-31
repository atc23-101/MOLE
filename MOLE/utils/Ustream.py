import torch
import os
from UniMoE.core import get_stream_cdata, disable_multi_stream


class ExpertStreamCtx(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            print("#!#building Stream Contex: ", kw["device"])
            cls._instance = object.__new__(cls)
            cls._instance.disable = kw["disable"]
            cls._instance.device = kw["device"]
            cls._instance.stream_pool = []
            cls._instance.default = torch.cuda.current_stream(
                device=cls._instance.device)

            idx = int(torch.device(cls._instance.device).index)
            if cls._instance.disable:
                disable_multi_stream(idx)
                cls._instance.stream_pool = [
                    cls._instance.default for _ in range(16)]
                return
            for i in range(16):

                tmp_stream = torch.cuda.Stream(_cdata=get_stream_cdata(idx, i))
                if len(cls._instance.stream_pool) == 0 or cls._instance.stream_pool[0] != tmp_stream:
                    cls._instance.stream_pool.append(tmp_stream)
                else:
                    break
            cls._instance.max_stream_count = len(cls._instance.stream_pool)
            print("STREAM COUNT: ", cls._instance.max_stream_count)

            torch.cuda.set_stream(ExpertStreamCtx.comp_stream())
            print("Set Stream COMP")

        return cls._instance

    def __init__(self, *args, **kw):
        pass

    @staticmethod
    def max_stream_count():
        return ExpertStreamCtx._instance.max_stream_count

    @staticmethod
    def stream(idx):
        return ExpertStreamCtx._instance.stream_pool[idx]

    @staticmethod
    def tran_stream():
        return ExpertStreamCtx._instance.stream_pool[1]

    @staticmethod
    def comp_stream():
        return ExpertStreamCtx._instance.stream_pool[2]

    @staticmethod
    def comm_stream():
        return ExpertStreamCtx._instance.stream_pool[3]

    @staticmethod
    def tran_back_stream():
        return ExpertStreamCtx._instance.stream_pool[4]

    @staticmethod
    def default_stream():
        return ExpertStreamCtx._instance.default

    @staticmethod
    def all_streams_cdata():
        return [s._cdata for s in ExpertStreamCtx._instance.stream_pool]
