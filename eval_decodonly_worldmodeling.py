"""
for comparison
represents something close to Micheli2023's architecture
(but simplified b/c this is not RL) 
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
from other_archs.worldmodel_micheli import WorldModel, WorldModelOutput
from transformer_module import PHYREVideoClassifier

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
    device, cpu, logger, _, writer = common.initial_boilerplate(conf)

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(picking.pick_pretrained_tokenizer_path(conf), map_location=device))
    tokenizer.eval()

    worldmodel = WorldModel(conf.transformer.decoder.config).to(device)
    ckpt_decodonly = torch.load(picking.pick_pretrained_decodonly_path(conf), map_location=device)
    worldmodel.load_state_dict(ckpt_decodonly["worldmodel"])
    worldmodel.eval()

    # dataset
    dataset = picking.pick_dataset(conf)
    collate_fn = datautils.collate_fn_factory(datautils.collate_with_future_targets, datautils.make_batch_padded, conf.task_worldmodeling.given_frames, conf.dataset.is_labeled)
    train_iterator, eval_iterator = common.load_dataset(dataset, collate_fn, conf)

    # classifier
    classifier : PHYREVideoClassifier = hydra.utils.instantiate(conf.task_worldmodeling.classifier, permutation_invariant=False).to(device)

    optimizer = transformer_module.configure_the_one_optimizer([
        (classifier, conf.task_worldmodeling.training)
    ], device_type=conf.device_type)

    subbatches_per_batch = conf.training.batch_size // conf.training.subbatch_size
    loss_accumulation = []
    predicted_probs_accumulation = []
    labels_accumulation = []

    global_step = 0

    for epoch_idx in range(conf.training.epochs):
        for batch_idx in range(conf.training.batches_per_epoch):
            
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

                if eval_mode:
                    batch : datautils.Batch = eval_iterator.next().to(device)
                    classifier.eval()
                else:
                    batch : datautils.Batch = train_iterator.next().to(device)
                    classifier.train()

                input_tokimg, target_tokimg, zq_in_shape, zq_out_shape = common.pass_through_tokenizer(tokenizer, batch, IGNORE_IN_CROSS_ENTROPY)

                recon_tokimg = input_tokimg
                target_tokimg = einops.rearrange(target_tokimg, "batch videotime token -> (batch videotime) token")

                while recon_tokimg.shape[1] <= zq_out_shape.T:
                    recon_tokimg_semiflat = einops.rearrange(recon_tokimg, "batch videotime token -> (batch videotime) token")

                    reconstructed_logits = worldmodel(recon_tokimg_semiflat, for_the_loss=True).logits_observations
                    # shape is [batch, token, embedding]
                    # remove the last token b/c it predicts outside of the original sequence
                    # (decoder adds BOS at the beginning of the sequence in order to make it possible to predict the first token from nothing)
                    reconstructed_logits = reconstructed_logits[:, :-1, :]

                    # get from logits to tokens, to prepare for the next step
                    reconstructed_probs = F.softmax(reconstructed_logits, dim=-1)
                    reconstructed_tokens = torch.topk(reconstructed_probs, k=1, dim=-1).indices
                    reconstructed_tokens = einops.rearrange(
                        reconstructed_tokens,
                        "(batch videotime) token 1 -> batch videotime token",
                        batch=recon_tokimg.shape[0],
                        videotime=recon_tokimg.shape[1],
                    )
                    new_reconstructed_timestep = reconstructed_tokens[:, [-1]]

                    recon_tokimg = torch.cat([recon_tokimg, new_reconstructed_timestep], dim=1)

                reconstructed_logits = einops.rearrange(reconstructed_logits, "batch token emb -> (batch token) emb")
                target_tokimg = einops.rearrange(target_tokimg, "batch token -> (batch token)")

                recon_cross_entropy = utils.steve_cross_entropy(reconstructed_logits, target_tokimg,
                                                                num_classes=reconstructed_logits.shape[1],
                                                                ignore_index=IGNORE_IN_CROSS_ENTROPY,
                                                                BT=reconstructed_logits.shape[0])
                
                # NOW for the new stuff
                reconstructed_embeddings = worldmodel.transformer.transformer.wte(recon_tokimg)  # shape: (batch videotime token embedding)
                last_frames = batch.get_last_frames(reconstructed_embeddings)  # (eliminates time axis)
                predicted_probs = classifier(last_frames)  # only last timestep, to supply only the necessary info
                loss = F.binary_cross_entropy(predicted_probs, batch.labels)

                loss_accumulation.append(loss)
                predicted_probs_accumulation.append(predicted_probs)
                labels_accumulation.append(batch.labels)

            loss = torch.mean(torch.stack(loss_accumulation))
            predicted_probs = torch.cat(predicted_probs_accumulation, dim=0)
            the_labels = torch.cat(labels_accumulation, dim=0)
            loss_accumulation.clear()
            predicted_probs_accumulation.clear()
            labels_accumulation.clear()

            if not eval_mode:
                loss.backward()
                optimizer.step()
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)

                cm = utils.BinaryConfusionMatrix(predicted_probs, the_labels, input_is_probs=True)
                writer.add_scalar('TRAIN/accuracy', cm.accuracy.item(), global_step)
                writer.add_scalar('TRAIN/precision', cm.precision.item(), global_step)
                writer.add_scalar('TRAIN/recall', cm.recall.item(), global_step)
                writer.add_scalar('TRAIN/f1_score', cm.f1_score.item(), global_step)
                writer.add_scalar('TRAIN/yes_rate', cm.yes_rate.item(), global_step)

            # end-of-epoch eval
            if eval_mode:
                writer.add_scalar('EVAL/loss', loss.item(), global_step)
                
                cm = utils.BinaryConfusionMatrix(predicted_probs, the_labels, input_is_probs=True)
                writer.add_scalar('EVAL/accuracy', cm.accuracy.item(), global_step)
                writer.add_scalar('EVAL/precision', cm.precision.item(), global_step)
                writer.add_scalar('EVAL/recall', cm.recall.item(), global_step)
                writer.add_scalar('EVAL/f1_score', cm.f1_score.item(), global_step)
                writer.add_scalar('EVAL/yes_rate', cm.yes_rate.item(), global_step)

                # visualize
                frames = common.generate_video_for_tensorboard(reconstructed_logits, tokenizer, batch, zq_out_shape, croc_attn_weights=None, MAX_BATCH=2)
                writer.add_video('EVAL_recons/epoch={:03}/i={:05}'.format(epoch_idx, batch_idx), frames)

                checkpoint = {
                    'epoch': epoch_idx,
                    'global_step': global_step,
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, "eval_decodonly_worldmodeling.pt")

                del frames





main()
