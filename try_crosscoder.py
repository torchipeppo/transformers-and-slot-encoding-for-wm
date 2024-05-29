import torch
import torch.nn.functional as F
from transformer_module.decoder import GPTConfig
from transformer_module.crosscoder import TransformerCrosscoder

from omegaconf import OmegaConf
conf = OmegaConf.merge(
    OmegaConf.load("config/default.yaml"),
    OmegaConf.load("config/transformer/default.yaml"),
)


slots = torch.randn((conf.the_global.batch_size, conf.slots.slot_no, conf.slots.slot_dim))

config = GPTConfig(
    block_size = conf.crosscoder.block_size,
    vocab_size = conf.crosscoder.vocab_size,
    n_layer = conf.crosscoder.n_layer,
    n_head = conf.crosscoder.n_head,
    n_embd = conf.crosscoder.n_embd,
    dropout = conf.crosscoder.dropout,
    bias = conf.crosscoder.bias,
)
luma = TransformerCrosscoder(config)

idx = torch.randint(0, config.vocab_size, (conf.the_global.batch_size, conf.the_global.tokimg_sequence_length), dtype=torch.int)

for _ in range(32):
    print(_)
    # forward the model
    slots = luma(slots, idx)
