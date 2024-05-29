"""
Credits to https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from omegaconf import OmegaConf
import inspect

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def _make_mha(config):
    return nn.MultiheadAttention(
        embed_dim=config.n_embd,
        num_heads=config.n_head,
        bias=config.bias,
        dropout=config.dropout,
        batch_first=True,
    )

# accepts a list of pairs (model, conf)
def configure_the_one_optimizer(list_of_model_conf_pairs, *, device_type):
    all_optim_groups = []
    for model, conf in list_of_model_conf_pairs:
        optim_groups = model.configure_optim_groups(**(OmegaConf.to_object(conf)))
        all_optim_groups.extend(optim_groups)
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer
