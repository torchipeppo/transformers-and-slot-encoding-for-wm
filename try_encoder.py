import torch
from transformer_module.decoder import GPTConfig
from transformer_module.encoder import TransformerEncoder

from omegaconf import OmegaConf
conf = OmegaConf.merge(
    OmegaConf.load("config/default.yaml"),
    OmegaConf.load("config/transformer/default.yaml"),
)

slots = torch.randn((conf.the_global.batch_size, conf.slots.slot_no, conf.slots.slot_dim))

config = GPTConfig(block_size=conf.the_global.tokimg_sequence_length)
kirby = TransformerEncoder(config)

for _ in range(32):
    print(_)
    slots = kirby(slots)
