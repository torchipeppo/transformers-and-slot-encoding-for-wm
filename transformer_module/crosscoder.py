import math
import inspect

import torch
import torch.nn as nn

from .common import LayerNorm, MLP, _make_mha
from .decoder import GPTConfig

# Cross-attention like a decoder, bidirectional context like an encoder
class CrosscoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1s = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1f = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = _make_mha(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, slots, tokenized_frame, need_weights=False):
        x = slots

        f_normd = self.ln_1f(tokenized_frame)
        x_normd = self.ln_1s(x)
        x_attn, attn_weights = self.attn(query=x_normd, key=f_normd, value=f_normd, need_weights=need_weights)
        x = x + x_attn

        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights

class TransformerCrosscoder(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([CrosscoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, slots, tokenized_frame, *, need_weights=False, tokenized_frame_given_as_indices=True):
        device = slots.device
        if tokenized_frame_given_as_indices:
            b, t = tokenized_frame.size()
        else:
            b, t, c = tokenized_frame.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the model itself
        if tokenized_frame_given_as_indices:
            tok_emb = self.transformer.wte(tokenized_frame) # token embeddings of shape (b, t, n_embd)
        else:
            tok_emb = tokenized_frame # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = slots
        y = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, attn_weights = block(x, y, need_weights=need_weights)  # only the attn_weights of the final layer count

        if need_weights:
            return x, attn_weights
        else:
            return x

    def configure_optim_groups(self, weight_decay, **extra_conf):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, **extra_conf},
            {'params': nodecay_params, 'weight_decay': 0.0, **extra_conf}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        return optim_groups

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optim_groups = self.configure_optim_groups(weight_decay)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
