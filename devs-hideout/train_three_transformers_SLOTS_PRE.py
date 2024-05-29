"""
loss peggiore
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
import einops
import os, sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset import make_batch
import transformer_module
import utils
import utils.picking as picking
import logging
from dataclasses import dataclass
# for typing
from tokenizer_module import Tokenizer
from transformer_module import TransformerCrosscoder, TransformerEncoder, TransformerDecoder

IGNORE_IN_CROSS_ENTROPY = -100

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper


# happened to get `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` once
# https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


@dataclass
class Batch():
    input_image: torch.Tensor
    target_image: torch.Tensor
    padding_mask: torch.Tensor
    input_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor

    # torch's stance is unclear, but this implementation ALWAYS SIDE-EFFECTS the batch.
    def to(self, *args, **kwargs):
        self.input_image = self.input_image.to(*args, **kwargs)
        self.target_image = self.target_image.to(*args, **kwargs)
        self.padding_mask = self.padding_mask.to(*args, **kwargs)
        self.input_padding_mask = self.input_padding_mask.to(*args, **kwargs)
        self.target_padding_mask = self.target_padding_mask.to(*args, **kwargs)
        return self

def collate_fn(list_of_samples):
    # video shape is [t c w h]   (or h w? doesn't matter here anyway)

    # eliminate last frame of the video from the input, b/c we wanna predict the NEXT frame
    # so we don't want to try and predict what comes after the final frame at training time
    # because we'd have no ground truth to compute the loss against
    input_samples = [video[:-1, :, :, :] for video in list_of_samples]
    # shift target vector 1 frame to the left, b/c we're gonna predict the NEXT frame
    target_samples = [video[:-1, :, :, :] for video in list_of_samples]

    input_samples, input_padding_mask = make_batch(input_samples)
    target_samples, target_padding_mask = make_batch(target_samples)
    assert torch.all(input_padding_mask==target_padding_mask)
    return Batch(input_samples, target_samples, input_padding_mask, input_padding_mask, target_padding_mask)

@hydra.main(config_path="config", config_name="default")
@hydraconf_autohandle_torch_device
def main(conf):
    device = torch.device(conf.device)
    cpu = torch.device("cpu")
    if conf.device_type=="cuda":
        force_cudnn_initialization()

    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info("================================")

    is_rng = torch.Generator()
    is_rng.manual_seed(14383421)
    initial_slots = torch.randn(
        (conf.training.batch_size, conf.transformer.slots.slot_no, conf.transformer.slots.slot_dim),
        generator=is_rng
    )

    log_dir = os.path.join(conf.path_constants.logs_dir, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(picking.pick_pretrained_tokenizer_path(conf), map_location=device))
    tokenizer.eval()

    crosscoder : TransformerCrosscoder = hydra.utils.instantiate(conf.transformer.crosscoder).to(device)
    encoder : TransformerEncoder = hydra.utils.instantiate(conf.transformer.encoder).to(device)
    decoder : TransformerDecoder = hydra.utils.instantiate(conf.transformer.decoder).to(device)

    # dataset
    dataset = picking.pick_dataset(conf)
    datasplit_rng = torch.Generator()
    datasplit_rng.manual_seed(14383421)
    train_len = int(len(dataset) * conf.training.train_split)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=datasplit_rng)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = utils.infiniter(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=conf.eval.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    eval_iterator = utils.infiniter(eval_dataloader)

    optimizer = transformer_module.configure_the_one_optimizer([
        (crosscoder, conf.training.crosscoder),
        (encoder, conf.training.encoder),
        (decoder, conf.training.decoder),
    ], device_type=conf.device_type)

    global_step = -1   # solely so that I can increment it at the beginning of the cycle
    for epoch_idx in range(conf.training.epochs):
        for batch_idx in range(conf.training.batches_per_epoch):
            print(f"epoch {epoch_idx}  batch {batch_idx}", end="")
            global_step += 1
            optimizer.zero_grad()

            if batch_idx == conf.training.batches_per_epoch-1:
                eval_mode = True
                batch : Batch = eval_iterator.next().to(device)
                crosscoder.eval()
                encoder.eval()
                decoder.eval()
                print(" (eval)")
            else:
                eval_mode = False
                batch : Batch = train_iterator.next().to(device)
                crosscoder.train()
                encoder.train()
                decoder.train()
                print()

            slots = initial_slots.detach().clone().to(device)

            tokenizer_output = tokenizer.encode(batch.input_image)
            zq_B, zq_T, zq_C, zq_H, zq_W = tokenizer_output.z_quantized.shape
            input_tokimg = tokenizer_output.tokens

            target_tokimg = tokenizer.encode(batch.target_image).tokens
            # mask out padding frames from the loss
            tokenmask = einops.repeat(batch.target_padding_mask, "batch videotime -> batch videotime token", token=target_tokimg.shape[2])
            target_tokimg = torch.where(tokenmask, IGNORE_IN_CROSS_ENTROPY, target_tokimg)

            time_dim = input_tokimg.shape[1]

            slots_by_time = []
            attentions_by_time = []
            for t in range(time_dim):
                input_t = input_tokimg[:, t, :]   # this selects a time point/slice AND eliminates the time axis (notation [:, [t], :] would have kept it)
                slots, attn_weights = crosscoder(slots, input_t, need_weights=True)
                slots_by_time.append(slots)
                slots = encoder(slots)
                attentions_by_time.append(attn_weights)
            # stack along the video time dimension
            slots = einops.rearrange(slots_by_time, "listaxis batch slot embedding -> batch listaxis slot embedding")
            croc_attn_weights = einops.rearrange(attentions_by_time, "listaxis batch slot token -> batch listaxis slot token")

            input_tokimg = einops.rearrange(input_tokimg, "batch videotime token -> (batch videotime) token")
            slots = einops.rearrange(slots, "batch videotime slot embedding -> (batch videotime) slot embedding")
            target_tokimg = einops.rearrange(target_tokimg, "batch videotime token -> (batch videotime) token")
            reconstructed_logits = decoder(input_tokimg, slots, for_the_loss=True)
            # shape is [batch, token, embedding]
            # remove the last token b/c it predicts outside of the original sequence (and thus is useless for cross-entropy loss)
            # (decoder adds BOS at the beginning of the sequence in order to make it possible to predict the first frame from nothing)
            reconstructed_logits = reconstructed_logits[:, :-1, :]

            reconstructed_logits = einops.rearrange(reconstructed_logits, "batch token emb -> (batch token) emb")
            target_tokimg = einops.rearrange(target_tokimg, "batch token -> (batch token)")
            loss = utils.steve_cross_entropy(reconstructed_logits, target_tokimg,
                                             num_classes=reconstructed_logits.shape[1],
                                             ignore_index=IGNORE_IN_CROSS_ENTROPY,
                                             BT=reconstructed_logits.shape[0])
            if not eval_mode:
                loss.backward()
                optimizer.step()
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)

            # end-of-epoch eval
            if eval_mode:
                writer.add_scalar('EVAL/loss', loss.item(), global_step)

                # visualize
                MAX_BATCH = 2
                reconstructed_probs = F.softmax(reconstructed_logits, dim=-1)
                reconstructed_tokens = torch.topk(reconstructed_probs, 1).indices
                reconstructed_tokens = einops.rearrange(reconstructed_tokens, "batch 1 -> batch")
                reconstructed_image = tokenizer.decode_from_tokenidx(reconstructed_tokens, h=zq_H, w=zq_W, should_postprocess=True)
                reconstructed_image = einops.rearrange(reconstructed_image, "(b t) c h w -> b t c h w", b=zq_B, t=zq_T)
                reconstructed_image = reconstructed_image.clamp(0., 1.)

                croc_attn_2d = einops.rearrange(
                    croc_attn_weights,
                    "batch videotime slot (token_h token_w) -> (batch videotime) slot token_h token_w",
                    token_h = zq_H,
                    token_w = zq_W,
                )
                croc_attn_big = F.interpolate(croc_attn_2d, batch.input_image.shape[-2:], mode="bilinear")
                croc_attn_big = einops.rearrange(croc_attn_big, "(b t) slot h w -> b t slot h w", b=zq_B, t=zq_T)

                frames = utils.visualize(batch.target_image, reconstructed_image, attns=croc_attn_big, MAX_BATCH=MAX_BATCH)
                writer.add_video('EVAL_recons/epoch={:03}/i={:05}'.format(epoch_idx, batch_idx), frames)

                logger.info(f"========= EVAL attention epoch={epoch_idx:03}/i={batch_idx:05} =========")
                assert torch.all(0 <= croc_attn_weights) and torch.all(croc_attn_weights <= 1)
                croc_attn_to_log = einops.rearrange(croc_attn_2d, "(b t) s th tw -> b t s th tw", b=zq_B, t=zq_T)
                croc_attn_to_log = croc_attn_to_log[:MAX_BATCH]
                logger.info(croc_attn_to_log)
                logger.info("")

                checkpoint = {
                    'epoch': epoch_idx,
                    'global_step': global_step,
                    'crosscoder': crosscoder.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, "three_transformers.pt")

                del reconstructed_probs
                del reconstructed_tokens
                del reconstructed_image
                del frames





main()
