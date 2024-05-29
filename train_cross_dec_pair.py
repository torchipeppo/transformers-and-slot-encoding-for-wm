"""
Test minimale coi transformer che abbiamo fatto:
li trattiamo alla stregua di un autoencoder per fare

tokenized_image ---crosscoder---> slots ---decoder---> tokens

e verificare se i tokens ricostruiscono la tokenized_image originale.

Idealmente testerei ciascun transformer separatamente, ma questa è la minima architettura che mi garantisce un obiettivo di training.
Così verifico che ho fatto dei transformer capaci di imparare QUALCOSA, anche se il collegamento col vero obietivo è vago.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
import einops
import os, sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset import PhyreVideoDataset, make_batch
import transformer_module
import utils
import logging
from dataclasses import dataclass

IGNORE_IN_CROSS_ENTROPY = -100

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper


@dataclass
class Batch():
    image: torch.TensorType
    padding_mask: torch.Tensor

    # torch's stance is unclear, but this implementation ALWAYS SIDE-EFFECTS the batch.
    def to(self, *args, **kwargs):
        self.image = self.image.to(*args, **kwargs)
        self.padding_mask = self.padding_mask.to(*args, **kwargs)
        return self

def collate_fn(list_of_samples):
    samples, padding_mask = make_batch(list_of_samples)
    return Batch(samples, padding_mask)

@hydra.main(config_path="config", config_name="default")
@hydraconf_autohandle_torch_device
def main(conf):
    device = torch.device(conf.device)

    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info("================================")

    is_rng = torch.Generator()
    is_rng.manual_seed(14383421)
    initial_slots = torch.randn(
        (conf.transformer.slots.slot_no, conf.transformer.slots.slot_dim),
        generator=is_rng
    ).to(device)

    log_dir = os.path.join(conf.path_constants.logs_dir, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)

    tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(conf.path_constants.pretrained_tokenizer, map_location=device))
    tokenizer.eval()

    crosscoder = hydra.utils.instantiate(conf.transformer.crosscoder).to(device)
    decoder = hydra.utils.instantiate(conf.transformer.decoder).to(device)

    # dataset
    datasplit_rng = torch.Generator()
    datasplit_rng.manual_seed(14383421)
    dataset = PhyreVideoDataset(conf.path_constants.phyre_video_dataset)
    train_len = int(len(dataset) * conf.training.train_split)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=datasplit_rng)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=True, collate_fn=collate_fn)
    train_iterator = utils.infiniter(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=conf.eval.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_iterator = utils.infiniter(eval_dataloader)

    optimizer = transformer_module.configure_the_one_optimizer([
        (crosscoder, conf.training.crosscoder),
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
                decoder.eval()
                print(" (eval)")
            else:
                eval_mode = False
                batch : Batch = train_iterator.next().to(device)
                crosscoder.train()
                decoder.train()
                print()

            slots = initial_slots.detach().clone()

            tokenizer_output = tokenizer.encode(batch.image)
            zq_B, zq_T, zq_C, zq_H, zq_W = tokenizer_output.z_quantized.shape
            input_tokimg = tokenizer_output.tokens

            target_tokimg = input_tokimg.detach().clone()
            # mask out padding frames from the loss
            tokenmask = einops.repeat(batch.padding_mask, "batch videotime -> batch videotime token", token=target_tokimg.shape[2])
            target_tokimg = torch.where(tokenmask, IGNORE_IN_CROSS_ENTROPY, target_tokimg)

            time_dim = input_tokimg.shape[1]

            input_tokimg = einops.rearrange(input_tokimg, "batch videotime token -> (batch videotime) token")
            target_tokimg = einops.rearrange(target_tokimg, "batch videotime token -> (batch videotime) token")
            slots = einops.repeat(slots, "slot embedding -> repeat slot embedding", repeat=input_tokimg.shape[0])

            slots = crosscoder(slots, input_tokimg)
            reconstructed_logits = decoder(input_tokimg, slots, for_the_loss=True)
            # shape is [batch, token, embedding]
            # remove the last token b/c it predicts outside of the original sequence (and thus is useless for cross-entropy loss)
            # (decoder adds BOS at the beginning of the sequence in order to make it possible to predict the first frame from nothing)
            reconstructed_logits = reconstructed_logits[:, :-1, :]

            reconstructed_logits = einops.rearrange(reconstructed_logits, "batch token emb -> (batch token) emb")
            target_tokimg = einops.rearrange(target_tokimg, "batch token -> (batch token)")
            loss = F.cross_entropy(reconstructed_logits, target_tokimg, ignore_index=IGNORE_IN_CROSS_ENTROPY)
            if not eval_mode:
                loss.backward()
                optimizer.step()
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)

            # end-of-epoch eval
            if eval_mode:
                writer.add_scalar('EVAL/loss', loss.item(), global_step)

                # visualize
                reconstructed_probs = F.softmax(reconstructed_logits, dim=-1)
                reconstructed_tokens = torch.topk(reconstructed_probs, 1).indices
                reconstructed_tokens = einops.rearrange(reconstructed_tokens, "batch 1 -> batch")
                reconstructed_image = tokenizer.decode_from_tokenidx(reconstructed_tokens, h=zq_H, w=zq_W, should_postprocess=True)
                reconstructed_image = einops.rearrange(reconstructed_image, "(b t) c h w -> b t c h w", b=zq_B, t=zq_T)
                reconstructed_image = reconstructed_image.clamp(0., 1.)

                frames = utils.visualize(batch.image, reconstructed_image, MAX_BATCH=2)
                writer.add_video('EVAL_recons/epoch={:03}/i={:05}'.format(epoch_idx, batch_idx), frames)

                checkpoint = {
                    'epoch': epoch_idx,
                    'global_step': global_step,
                    'crosscoder': crosscoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, "cross_dec_pair.pt")





main()
