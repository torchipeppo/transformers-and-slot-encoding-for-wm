"""
a butchered Micheli2023 for non-RL
for comparison
observation prediction only
"""

from dataclasses import dataclass
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_module import TransformerDecoder, GPTConfig


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = TransformerDecoder(config)

        self.apply(init_weights)

    def forward(self, tokens: torch.LongTensor, for_the_loss = False) -> WorldModelOutput:

        x = self.transformer(tokens, None, for_the_loss=for_the_loss, skip_cross=True)

        # TransformerDecoder already has a final head that returns logits, so no need for one here
        # The dataclass remains as vestigial back-comp.
        return WorldModelOutput(x, x)

    def configure_optim_groups(self, weight_decay, **extra_conf):
        optim_groups = self.transformer.configure_optim_groups(weight_decay, **extra_conf)
        return optim_groups

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
