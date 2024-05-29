import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import utils.picking as picking
import dataset.common as datautils
import logging
import os, sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import einops
from dataclasses import dataclass


# PREP

def initial_boilerplate(conf, *, additional_log_dir=None):
    device = torch.device(conf.device)
    cpu = torch.device("cpu")
    if conf.device_type=="cuda":
        utils.force_cudnn_initialization()
    
    now = datetime.today().isoformat()

    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info("================================")
    logger.info("run id: " + now)
    logger.info("================================")

    is_rng = torch.Generator()
    is_rng.manual_seed(14383421)
    initial_slots = torch.randn(
        (conf.training.subbatch_size, conf.transformer.slots.slot_no, conf.transformer.slots.slot_dim),
        generator=is_rng
    )

    if additional_log_dir:
        now = os.path.join(additional_log_dir, now)
    log_dir = os.path.join(conf.path_constants.logs_dir, now)
    writer = SummaryWriter(log_dir)

    return device, cpu, logger, initial_slots, writer

def load_dataset(dataset, collate_fn, conf):
    # train-test split
    datasplit_rng = torch.Generator()
    datasplit_rng.manual_seed(14383421)
    train_len = int(len(dataset) * conf.training.train_split)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=datasplit_rng)
    # actual dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=conf.training.subbatch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = utils.infiniter(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=conf.eval.subbatch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    eval_iterator = utils.infiniter(eval_dataloader)

    return train_iterator, eval_iterator


# TOKENIZER

@dataclass
class MyShape:
    B: int
    T: int
    C: int
    H: int
    W: int

def pass_through_tokenizer(tokenizer, batch, IGNORE_IN_CROSS_ENTROPY):
    tokenizer_output = tokenizer.encode(batch.input_image)
    zq_in_B, zq_in_T, zq_in_C, zq_in_H, zq_in_W = tokenizer_output.z_quantized.shape
    input_tokimg = tokenizer_output.tokens

    tokenizer_output = tokenizer.encode(batch.target_image)
    zq_out_B, zq_out_T, zq_out_C, zq_out_H, zq_out_W = tokenizer_output.z_quantized.shape
    target_tokimg = tokenizer_output.tokens
    # mask out padding frames from the loss
    tokenmask = einops.repeat(batch.target_padding_mask, "batch videotime -> batch videotime token", token=target_tokimg.shape[2])
    target_tokimg = torch.where(tokenmask, IGNORE_IN_CROSS_ENTROPY, target_tokimg)

    return (
        input_tokimg,
        target_tokimg,
        MyShape(zq_in_B, zq_in_T, zq_in_C, zq_in_H, zq_in_W),
        MyShape(zq_out_B, zq_out_T, zq_out_C, zq_out_H, zq_out_W),
    )


# THREE TRANSFORMERS

@dataclass
class AttentionWeightsFromSlotEncoding:
    croc_attn_weights: torch.Tensor

def do_slot_encoding(crosscoder, encoder, slots, input_tokimg, input_time_dim, target_time_dim, *, need_croc_attn):
    slots_by_time = []
    attentions_by_time = []
    for t in range(input_time_dim):
        input_t = input_tokimg[:, t, :]   # this selects a time point/slice AND eliminates the time axis (notation [:, [t], :] would have kept it)
        if need_croc_attn:
            slots, attn_weights = crosscoder(slots, input_t, need_weights=True)
            attentions_by_time.append(attn_weights)
        else:
            slots = crosscoder(slots, input_t)
        slots = encoder(slots)
        slots_by_time.append(slots)  # collect post-predict-step slots b/c we're predicting the next frame for world modeling
    # now comes the hard part: continue predicting unknown frames in the future
    for t in range(input_time_dim, target_time_dim):
        slots = encoder(slots)
        slots_by_time.append(slots)

    # stack along the video time dimension
    slots = einops.rearrange(slots_by_time, "listaxis batch slot embedding -> batch listaxis slot embedding")
    if need_croc_attn:
        croc_attn_weights = einops.rearrange(attentions_by_time, "listaxis batch slot token -> batch listaxis slot token")

    if need_croc_attn:
        return slots, croc_attn_weights
    else:
        return slots, None

def pass_through_decoder(decoder, slots, input_tokimg, input_time_dim, target_time_dim, zq_in_shape):
    slots_given = einops.rearrange(slots[:, :input_time_dim], "batch videotime slot embedding -> (batch videotime) slot embedding")
    reconstructed_logits = decoder(input_tokimg, slots_given, for_the_loss=True)
    # shape is [batch, token, embedding]
    # remove the last token b/c it predicts outside of the original sequence (and thus is useless for cross-entropy loss)
    # (decoder adds BOS at the beginning of the sequence in order to make it possible to predict the first frame from nothing)
    reconstructed_logits = reconstructed_logits[:, :-1, :]

    reconstructed_logits = einops.rearrange(reconstructed_logits, "(batch videotime) token emb -> batch videotime token emb",
                                            batch=zq_in_shape.B, videotime=zq_in_shape.T)

    # now comes the hard part
    nongiven_recon_logits = []
    last_reconstructed_timestep = reconstructed_logits[:, -1, :, :]
    for t in range(input_time_dim, target_time_dim):
        last_reconstructed_timestep = last_reconstructed_timestep.argmax(dim=2)
        new_reconstructed_logits = decoder(last_reconstructed_timestep, slots[:, t], for_the_loss=True)
        new_reconstructed_logits = new_reconstructed_logits[:, :-1, :]  # as per the big comment above
        nongiven_recon_logits.append(new_reconstructed_logits[:, None, :, :])  # add videotime axis to prepare for the cat
        last_reconstructed_timestep = new_reconstructed_logits  # get ready for next iteration
    reconstructed_logits = torch.cat([reconstructed_logits] + nongiven_recon_logits, dim=1)
    reconstructed_logits = einops.rearrange(reconstructed_logits, "batch videotime token emb -> (batch videotime) token emb")

    return reconstructed_logits


# OTHER

def generate_video_for_tensorboard(reconstructed_logits, tokenizer, batch, zq_out_shape, *, croc_attn_weights, MAX_BATCH):
    reconstructed_probs = F.softmax(reconstructed_logits, dim=-1)
    reconstructed_tokens = torch.topk(reconstructed_probs, 1).indices
    reconstructed_tokens = einops.rearrange(reconstructed_tokens, "batch 1 -> batch")
    reconstructed_image = tokenizer.decode_from_tokenidx(reconstructed_tokens, h=zq_out_shape.H, w=zq_out_shape.W, should_postprocess=True)
    reconstructed_image = einops.rearrange(reconstructed_image, "(b t) c h w -> b t c h w", b=zq_out_shape.B, t=zq_out_shape.T)
    reconstructed_image = reconstructed_image.clamp(0., 1.)

    if croc_attn_weights is not None:
        croc_attn_2d = einops.rearrange(
            croc_attn_weights,
            "batch videotime slot (token_h token_w) -> (batch videotime) slot token_h token_w",
            token_h = zq_out_shape.H,
            token_w = zq_out_shape.W,
        )
        croc_attn_big = F.interpolate(croc_attn_2d, batch.input_image.shape[-2:], mode="bilinear")
        croc_attn_big = einops.rearrange(croc_attn_big, "(b t) slot h w -> b t slot h w", b=zq_out_shape.B, t=zq_out_shape.T)
        del croc_attn_2d
    else:
        croc_attn_big = None
    
    del reconstructed_probs
    del reconstructed_tokens

    return utils.visualize(batch.target_image, reconstructed_image, attns=croc_attn_big, MAX_BATCH=MAX_BATCH)
