"""

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
import einops
import os, sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import dataset.common as datautils
import transformer_module
import utils
import utils.picking as picking
import logging
from dataclasses import dataclass
# for typing
from tokenizer_module import Tokenizer
from transformer_module import TransformerCrosscoder, TransformerEncoder, TransformerDecoder

import common_main_code as common

IGNORE_IN_CROSS_ENTROPY = -100

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper


@hydra.main(config_path="config", config_name="default")
@hydraconf_autohandle_torch_device
def main(conf):
    device, cpu, logger, initial_slots, writer = common.initial_boilerplate(conf, additional_log_dir="3transf")

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(picking.pick_pretrained_tokenizer_path(conf), map_location=device))
    tokenizer.eval()

    crosscoder : TransformerCrosscoder = hydra.utils.instantiate(conf.transformer.crosscoder).to(device)
    encoder : TransformerEncoder = hydra.utils.instantiate(conf.transformer.encoder).to(device)
    decoder : TransformerDecoder = hydra.utils.instantiate(conf.transformer.decoder).to(device)

    # dataset
    dataset = picking.pick_dataset(conf)
    collate_fn = datautils.collate_fn_factory(datautils.collate_with_future_targets, datautils.make_batch_padded, labeled=conf.dataset.is_labeled)
    train_iterator, eval_iterator = common.load_dataset(dataset, collate_fn, conf)

    optimizer = transformer_module.configure_the_one_optimizer([
        (crosscoder, conf.training.crosscoder),
        (encoder, conf.training.encoder),
        (decoder, conf.training.decoder),
    ], device_type=conf.device_type)

    subbatches_per_batch = conf.training.batch_size // conf.training.subbatch_size
    loss_accumulation = []

    global_step = 0

    for epoch_idx in range(conf.training.epochs):
        if epoch_idx == 2 and conf.report_memory:
            with open("memory_summary.txt", "w") as f:
                f.write(torch.cuda.memory_summary(device))
                print("Memory summary written, kthxbye")
                break

        for batch_idx in range(conf.training.batches_per_epoch):
            print(f"epoch {epoch_idx}  batch {batch_idx}", end="")

            optimizer.zero_grad()

            if batch_idx == conf.training.batches_per_epoch-1:
                eval_mode = True
                print(" (eval)")
            else:
                eval_mode = False
                print()
            
            for subbatch_idx in range(subbatches_per_batch):
                print(f"epoch {epoch_idx}  batch {batch_idx}  sub {subbatch_idx}")

                global_step += conf.training.subbatch_size

                if batch_idx == conf.training.batches_per_epoch-1:
                    batch : datautils.Batch = eval_iterator.next().to(device)
                    crosscoder.eval()
                    encoder.eval()
                    decoder.eval()
                else:
                    batch : datautils.Batch = train_iterator.next().to(device)
                    crosscoder.train()
                    encoder.train()
                    decoder.train()

                slots = initial_slots.detach().clone().to(device)

                input_tokimg, target_tokimg, zq_in_shape, zq_out_shape = common.pass_through_tokenizer(tokenizer, batch, IGNORE_IN_CROSS_ENTROPY)

                slots, croc_attn_weights = common.do_slot_encoding(crosscoder, encoder, slots, input_tokimg, zq_in_shape.T, zq_out_shape.T, need_croc_attn=True)

                input_tokimg = einops.rearrange(input_tokimg, "batch videotime token -> (batch videotime) token")
                target_tokimg = einops.rearrange(target_tokimg, "batch videotime token -> (batch videotime) token")

                reconstructed_logits = common.pass_through_decoder(decoder, slots, input_tokimg, zq_in_shape.T, zq_out_shape.T, zq_in_shape)

                reconstructed_logits = einops.rearrange(reconstructed_logits, "batch token emb -> (batch token) emb")
                target_tokimg = einops.rearrange(target_tokimg, "batch token -> (batch token)")

                loss = utils.steve_cross_entropy(reconstructed_logits, target_tokimg,
                                                 num_classes=reconstructed_logits.shape[1],
                                                 ignore_index=IGNORE_IN_CROSS_ENTROPY,
                                                 BT=reconstructed_logits.shape[0])
                
                loss_accumulation.append(loss)
            
            loss = torch.mean(torch.stack(loss_accumulation))
            loss_accumulation.clear()
            
            if not eval_mode:
                loss.backward()
                optimizer.step()
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)

            # end-of-epoch eval
            if eval_mode:
                writer.add_scalar('EVAL/loss', loss.item(), global_step)

                # visualize
                if conf.enable_videos:
                    frames = common.generate_video_for_tensorboard(reconstructed_logits, tokenizer, batch, zq_out_shape, croc_attn_weights=None, MAX_BATCH=2)
                    writer.add_video('EVAL_recons/epoch={:03}/i={:05}'.format(epoch_idx, batch_idx), frames)
                    del frames

                checkpoint = {
                    'epoch': epoch_idx,
                    'global_step': global_step,
                    'crosscoder': crosscoder.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, "three_transformers.pt")

                del checkpoint





main()
