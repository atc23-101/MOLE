import time
import torch


class UTimer:
    def __init__(self) -> None:
        self.gap = []
        self.type = None
        self.lst = None

    def syncstep_start(self):
        if self.type is None:
            self.type = "sync"
        elif self.type != "sync":
            assert False, "SYNC TIMER ERR"
        torch.cuda.synchronize()
        self.lst = time.time()

    def syncstep_end(self):
        if self.type is None:
            self.type = "sync"
        elif self.type != "sync":
            assert False, "SYNC TIMER ERR"
        torch.cuda.synchronize()
        self.gap.append(time.time() - self.lst)
        self.lst = None

    def cpustep_start(self):
        if self.type is None:
            self.type = "cpu"
        elif self.type != "cpu":
            assert False, "CPU TIMER ERR"

        self.lst = time.time()

    def cpustep_end(self):
        if self.type is None:
            self.type = "cpu"
        elif self.type != "cpu":
            assert False, "CPU TIMER ERR"

        self.gap.append(time.time() - self.lst)
        self.lst = None

    def eventstep_start(self):
        if self.type is None:
            self.type = "event"
        elif self.type != "event":
            assert False, "EVENT TIMER ERR"

        self.lst = torch.cuda.Event(enable_timing=True)
        self.lst.record()

    def eventstep_end(self):
        if self.type is None:
            self.type = "event"
        elif self.type != "event":
            assert False, "EVENT TIMER ERR"

        tmp = torch.cuda.Event(enable_timing=True)
        tmp.record()
        self.gap.append(self.lst.elapsed_time(tmp))
        self.lst = None

    def stats(self):
        return self.gap

    def clear(self):
        self.gap = []
        self.lst = None


class UATimer:
    fwd_timer = UTimer()
    bwd_timer = UTimer()

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            print("#!#building UniMoE Timer: ")
            cls._instance = object.__new__(cls)

        return cls._instance

    def __init__(self, *args, **kw):
        pass

    @staticmethod
    def track_fwd_start():
        UATimer.fwd_timer.syncstep_start()

    @staticmethod
    def track_fwd_end():
        UATimer.fwd_timer.syncstep_end()

    @staticmethod
    def track_bwd_start():
        UATimer.bwd_timer.syncstep_start()

    @staticmethod
    def track_bwd_end():
        UATimer.bwd_timer.syncstep_end()

    @staticmethod
    def stats():
        return UATimer.fwd_timer.stats(), UATimer.bwd_timer.stats()

    @staticmethod
    def clear():
        UATimer.fwd_timer.clear()
        UATimer.bwd_timer.clear()
