from .utils import *

from .dvae import dVAE
from .transformer import TransformerEncoder, TransformerDecoder

import einops
from utils import steve_cross_entropy

from common_main_code import MyShape

IGNORE_IN_CROSS_ENTROPY = -100


class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if i < self.num_iterations - 1:
                    slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.savi = SlotAttentionVideo(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size,
            args.num_predictor_blocks, args.num_predictor_heads, args.predictor_dropout)

        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)


class STEVEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)


# added by me, in order to get more stuff out of the forward more orderly
from dataclasses import dataclass
@dataclass
class STEVEOutput:
    recon: torch.Tensor
    cross_entropy: torch.Tensor
    dvae_mse: torch.Tensor
    attns: torch.Tensor
    slots: torch.Tensor
    reconstructed_logits: torch.Tensor
    zq_in_shape: MyShape  # see common_main_code.py
    zq_out_shape: MyShape  # see common_main_code.py

    @classmethod
    def from_original_tuple(cls, of_tuple, *args, **kwargs):
        return cls(*of_tuple, *args, **kwargs)
    
    def get_original_tuple(self):
        return (self.recon, self.cross_entropy, self.dvae_mse, self.attns)

class STEVE(nn.Module):
    
    def __init__(self, *, args, tokenizer):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.img_channels = args.img_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        # dvae
        # self.dvae = dVAE(args.vocab_size, args.img_channels)
        self.tokenizer = tokenizer

        # encoder networks
        self.steve_encoder = STEVEEncoder(args)

        # decoder networks
        self.steve_decoder = STEVEDecoder(args)

    def forward(self, video, tau, hard):
        return self._forward(video, tau, hard, original_return=True)
    
    # "extended" forward, returns more stuff (no other changes)
    def forward_ext(self, video, tau, hard, * , target_video=None):
        return self._forward(video, tau, hard, original_return=False, target_video=target_video)

    def _forward(self, video, tau, hard, *, original_return=True, target_video=None):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W

        # dvae encode
        tokenizer_output = self.tokenizer.encode(video_flat)
        z_emb = tokenizer_output.z_quantized                                                        # B * T, d_model, H_enc, W_enc
        z_emb = einops.rearrange(z_emb, "b e h w -> b (h w) e")                                     # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.tokenizer.decode(tokenizer_output.z).reshape(B, T, C, H, W)               # B, T, C, H, W
        dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

        # decode
        slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
        pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
        pred_prehead = pred
        pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
        cross_entropy = steve_cross_entropy(pred, tokenizer_output.tokens,                                  # 1
                                            num_classes=pred.shape[2],
                                            ignore_index=IGNORE_IN_CROSS_ENTROPY,
                                            BT=pred.shape[0]*pred.shape[1])
        
        # addition for the eval_worldmodeling
        # now comes the hard part: continue predicting unknown frames in the future
        if target_video is not None:
            last_slot = slots[:, -1]
            blind_slots = []
            while len(blind_slots) < target_video.shape[1] - slots.shape[1]:
                last_slot = self.steve_encoder.savi.predictor(last_slot)
                blind_slots.append(last_slot[:, None])
            slots = torch.cat([slots] + blind_slots, dim=1)
            blind_preds = []
            pred_prehead_unflat = einops.rearrange(pred_prehead, "(b t) hw dmodel -> b t hw dmodel", b=B, t=T)
            last_pred_prehead = pred_prehead_unflat[:, -1]                                                                        # b hw dmodel
            while len(blind_preds) < target_video.shape[1] - pred_prehead_unflat.shape[1]:
                last_pred_prehead = self.steve_decoder.tf(last_pred_prehead, slots[:, pred_prehead_unflat.shape[1]+len(blind_preds)])     # b hw dmodel
                last_pred = self.steve_decoder.head(last_pred_prehead)                                           # b hw vocsz
                blind_preds.append(last_pred[:, None, :, :])  # add videotime axis to prepare for the cat
            pred_unflat = einops.rearrange(pred, "(b t) hw dmodel -> b t hw dmodel", b=B, t=T)
            pred_unflat = torch.cat([pred_unflat] + blind_preds, dim=1)
            pred = einops.rearrange(pred_unflat, "b t hw vocsz -> (b t) hw vocsz")

        ret = (dvae_recon.clamp(0., 1.),
                cross_entropy,
                dvae_mse,
                attns)
        if original_return:
            return ret
        else:
            zq_in_B, zq_in_T = video.shape[:2]
            zq_in_C, zq_in_H, zq_in_W = tokenizer_output.z_quantized.shape[-3:]
            zq_in_shape = MyShape(zq_in_B, zq_in_T, zq_in_C, zq_in_H, zq_in_W)
            if target_video is not None:
                zq_out_B, zq_out_T = target_video.shape[:2]
                zq_out_C, zq_out_H, zq_out_W = zq_in_C, zq_in_H, zq_in_W
                zq_out_shape = MyShape(zq_out_B, zq_out_T, zq_out_C, zq_out_H, zq_out_W)
            else:
                zq_out_shape = zq_in_shape
            return STEVEOutput.from_original_tuple(
                ret,
                slots=slots,
                reconstructed_logits=einops.rearrange(pred, "bt wh vocabsize -> (bt wh) vocabsize"),  # for compat/coherence w/ the rest of my code
                zq_in_shape=zq_in_shape,
                zq_out_shape=zq_out_shape,
            )

    def encode(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
                                                            # attns: B, T, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

        attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

        return slots, attns_vis, attns

    def decode(self, slots):
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.steve_encoder.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.steve_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.steve_decoder.tf(
                self.steve_decoder.pos(input),
                slots
            )
            z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.tokenizer.decode(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, video):
        """
        image: batch_size x img_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots, attns, _ = self.encode(video)
        recon_transformer = self.decode(slots.flatten(end_dim=1))
        recon_transformer = recon_transformer.reshape(B, T, C, H, W)

        return recon_transformer
