"""
Simple CUDA profiler, as an alternative to autograd's
Trying this first for three reasons:
 - I actually know what this does
 - Everyone on the top duck results says to do this and barely mention the autograd profiler
 - I could not find a clear analysis in autograd's favor.

https://discuss.pytorch.org/t/training-gets-slow-down-by-each-batch-slowly/4460
https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
https://github.com/Paxoo/PyTorch-Best_Practices/wiki/Correct-Way-To-Measure-Time
https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile
"""

import torch
import contextlib
import logging

# the only reason why there is this extra class is to make it possible to enable/disable by modifying one line in the source code
class SimpleCudaProfilerFactory:
    def __init__(self, enabled):
        self.enabled = enabled
    
    def profiler(self, log_header=""):
        if self.enabled:
            return SimpleCudaProfiler(log_header=log_header)
        else:
            return contextlib.nullcontext()



# I don't expect to use this directly, the factory is much easier to disable altogether
class SimpleCudaProfiler:
    def __init__(self, log_header=""):
        self.log_header = log_header
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed = None

    def __enter__(self):
        # fire start cuda event
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # fire end cuda event
        self.end_event.record()
        # Waits for everything to finish running (since cuda is asynchronous)
        torch.cuda.synchronize()
        # save elapsed time b/w events
        self.elapsed = self.start_event.elapsed_time(self.end_event)
        # autolog for ease of use (I don't plan to actually save and reference these instances)
        logger = logging.getLogger(__name__)
        logger.info(self.log_header + "  CUDA time: " + str(self.elapsed))
        return False
