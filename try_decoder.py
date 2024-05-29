import torch
import torch.nn.functional as F
from transformer_module.decoder import TransformerDecoder, GPTConfig

from omegaconf import OmegaConf
conf = OmegaConf.merge(
    OmegaConf.load("config/default.yaml"),
    OmegaConf.load("config/transformer/default.yaml"),
)

slots = torch.randn((conf.the_global.batch_size, conf.slots.slot_no, conf.slots.slot_dim))

config = GPTConfig(
    block_size = conf.decoder.block_size,
    vocab_size = conf.decoder.vocab_size,
    n_layer = conf.decoder.n_layer,
    n_head = conf.decoder.n_head,
    n_embd = conf.decoder.n_embd,
    dropout = conf.decoder.dropout,
    bias = conf.decoder.bias,
)
megatron = TransformerDecoder(config)

idx = torch.zeros((conf.the_global.batch_size, 1), dtype=torch.int)

for _ in range(32):
    print(idx.shape)
    # forward the model to get the logits for the index in the sequence
    logits = megatron(idx, slots)
    # pluck the logits at the final step
    logits = logits[:, -1, :]
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    idx = torch.cat((idx, idx_next), dim=1)
