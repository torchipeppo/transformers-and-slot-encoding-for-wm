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
from transformer_module import PHYREVideoClassifier
from other_archs.steve_singh import STEVE

import common_main_code as common

IGNORE_IN_CROSS_ENTROPY = -100

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper

@hydra.main(config_path="config", config_name="STEVE")
@hydraconf_autohandle_torch_device
def main(conf):
    device, cpu, logger, initial_slots, writer = common.initial_boilerplate(conf)

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(picking.pick_pretrained_tokenizer_path(conf), map_location=device))
    tokenizer.eval()

    model = STEVE(args=conf.steve_conf, tokenizer=tokenizer)
    ckpt_steve = torch.load(picking.pick_pretrained_steve_path(conf), map_location=device)
    model.load_state_dict(ckpt_steve['model'])
    model = model.to(device)
    model.eval()

    # dataset
    dataset = picking.pick_dataset(conf)
    collate_fn = datautils.collate_fn_factory(datautils.collate_with_future_targets, datautils.make_batch_padded, conf.task_worldmodeling.given_frames, conf.dataset.is_labeled)
    train_iterator, eval_iterator = common.load_dataset(dataset, collate_fn, conf)

    # classifier
    classifier : PHYREVideoClassifier = hydra.utils.instantiate(conf.task_worldmodeling.classifier).to(device)

    optimizer = transformer_module.configure_the_one_optimizer([
        (classifier, conf.task_worldmodeling.training)
    ], device_type=conf.device_type)

    profact = utils.SimpleCudaProfilerFactory(enabled=conf.enable_profiling)

    subbatches_per_batch = conf.training.batch_size // conf.training.subbatch_size
    loss_accumulation = []
    predicted_probs_accumulation = []
    labels_accumulation = []

    global_step = 0

    for epoch_idx in range(conf.training.epochs):
        for batch_idx in range(conf.training.batches_per_epoch):
            
            with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  optim 0grad"):
                optimizer.zero_grad()

            for subbatch_idx in range(subbatches_per_batch):
                print(f"epoch {epoch_idx}  batch {batch_idx}  sub {subbatch_idx}", end="")
                log_super_header = f"epo {epoch_idx}  bat {batch_idx}  sub {subbatch_idx}"

                global_step += conf.training.subbatch_size

                with profact.profiler(log_super_header + "  batch extract"):
                    if batch_idx == conf.training.batches_per_epoch-1:
                        eval_mode = True
                        with profact.profiler(log_super_header + "  batch extract - next()"):
                            batch : datautils.Batch = eval_iterator.next().to(device)
                        with profact.profiler(log_super_header + "  batch extract - eval()"):
                            classifier.eval()
                        print(" (eval)")
                    else:
                        eval_mode = False
                        with profact.profiler(log_super_header + "  batch extract - next()"):
                            batch : datautils.Batch = train_iterator.next().to(device)
                        with profact.profiler(log_super_header + "  batch extract - train()"):
                            classifier.train()
                        print()

                with profact.profiler(log_super_header + "  steve"):
                    steve_output = model.forward_ext(batch.input_image, conf.tau_final, conf.hard, target_video=batch.target_image)
                
                # NOW for the new stuff
                with profact.profiler(log_super_header + "  classifier"):
                    last_slots = batch.get_last_frames(steve_output.slots)  # (eliminates time axis)
                    predicted_probs = classifier(last_slots)  # only last timestep, to supply only the necessary info
                    loss = F.binary_cross_entropy(predicted_probs, batch.labels)

                loss_accumulation.append(loss)
                predicted_probs_accumulation.append(predicted_probs)
                labels_accumulation.append(batch.labels)

            with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  subbatch mean"):
                loss = torch.mean(torch.stack(loss_accumulation))
                predicted_probs = torch.cat(predicted_probs_accumulation, dim=0)
                the_labels = torch.cat(labels_accumulation, dim=0)
            loss_accumulation.clear()
            predicted_probs_accumulation.clear()
            labels_accumulation.clear()

            if not eval_mode:
                with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  loss bkwd"):
                    loss.backward()
                with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  optim step"):
                    optimizer.step()
                
                with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  tensorboard"):
                    writer.add_scalar('TRAIN/loss', loss.item(), global_step)

                    cm = utils.BinaryConfusionMatrix(predicted_probs, the_labels, input_is_probs=True)
                    writer.add_scalar('TRAIN/accuracy', cm.accuracy.item(), global_step)
                    writer.add_scalar('TRAIN/precision', cm.precision.item(), global_step)
                    writer.add_scalar('TRAIN/recall', cm.recall.item(), global_step)
                    writer.add_scalar('TRAIN/f1_score', cm.f1_score.item(), global_step)
                    writer.add_scalar('TRAIN/yes_rate', cm.yes_rate.item(), global_step)

            # end-of-epoch eval
            if eval_mode:
                with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  tensorboard (eval)"):
                    writer.add_scalar('EVAL/loss', loss.item(), global_step)
                    
                    cm = utils.BinaryConfusionMatrix(predicted_probs, the_labels, input_is_probs=True)
                    writer.add_scalar('EVAL/accuracy', cm.accuracy.item(), global_step)
                    writer.add_scalar('EVAL/precision', cm.precision.item(), global_step)
                    writer.add_scalar('EVAL/recall', cm.recall.item(), global_step)
                    writer.add_scalar('EVAL/f1_score', cm.f1_score.item(), global_step)
                    writer.add_scalar('EVAL/yes_rate', cm.yes_rate.item(), global_step)

                # visualize
                with profact.profiler(f"epo {epoch_idx}  bat {batch_idx}  visualize (eval)"):
                    frames = common.generate_video_for_tensorboard(steve_output.reconstructed_logits, tokenizer, batch, steve_output.zq_out_shape, croc_attn_weights=None, MAX_BATCH=2)
                    writer.add_video('EVAL_recons/epoch={:03}/i={:05}'.format(epoch_idx, batch_idx), frames)

                checkpoint = {
                    'epoch': epoch_idx,
                    'global_step': global_step,
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, "eval_3transf_worldmodeling.pt")





main()
