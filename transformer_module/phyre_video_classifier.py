import math
import inspect

import torch
import torch.nn as nn
import einops

from .decoder import GPTConfig
from .encoder import EncoderBlock

class PHYREVideoClassifier(nn.Module):

    def __init__(self, config: GPTConfig, *, permutation_invariant=True):
        super().__init__()
        self.config = config
        self.permutation_invariant = permutation_invariant
        if permutation_invariant:
            wpe_size = 2
        else:
            wpe_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(wpe_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
        ))
        self.classification_head_fc1     = nn.Linear(config.n_embd, config.n_embd//2, bias=False)
        self.classification_head_gelu    = nn.GELU()
        self.classification_head_fc2     = nn.Linear(config.n_embd//2, 1, bias=False)
        self.classification_head_dropout = nn.Dropout(config.dropout)
        self.classification_head_sigmoid = nn.Sigmoid()

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        self.CLS = nn.Parameter(torch.Tensor(1, 1, config.n_embd))
        nn.init.xavier_uniform_(self.CLS)

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
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        device = x.device
        b, s, e = x.size()
        s+=1  # account for CLS
        pos = torch.arange(0, s, dtype=torch.long, device=device) # shape (s)
        if self.permutation_invariant:
            pos = torch.where(pos==0, 0, 1)  # permutational invariance of slots
        clsbatch = einops.repeat(self.CLS, "b s e -> (repeat b) s e", repeat=b)

        x = torch.cat([clsbatch, x], dim=1)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(x + pos_emb)

        # forward the model itself
        for block in self.transformer.h:
            x = block(x)
        c = x[:, 0]  # focus exclusively on the classification spot now, nothing else matters
        c = self.classification_head_fc1(c)
        c = self.classification_head_gelu(c)
        c = self.classification_head_fc2(c)
        c = self.classification_head_sigmoid(c)
        return c

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
