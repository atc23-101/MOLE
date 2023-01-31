#adapted from fairseq
from functools import partial
import torch 
from UniMoE.native_atn_wrapper import native_relu_wrapper
from UniMoE.native_atn_wrapper import native_gelu_wrapper, native_gelu_backward_wrapper

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

def gelu_forward(x):
    return native_gelu_wrapper(x), x

def relu_forward(x):
    return native_relu_wrapper(x), x > 0

def gelu_backward(grad, x):
    return native_gelu_backward_wrapper(x, grad)

def relu_backward(grad, mask):
    return (mask )* grad

def build_activation_fwd(activation : str):
    if activation == "relu":
        return relu_forward
    elif activation == "gelu":
        return gelu_forward
    assert False, "activation not support"

def build_activation_bwd(activation , *args):
    if activation == "relu":
        return partial(relu_backward, mask = args[0])
    elif activation == "gelu":
        return partial(gelu_backward, x = args[0])
    assert False, "activation not support"

