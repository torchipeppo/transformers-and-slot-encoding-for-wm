from tokenizer_module import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
from torch.utils.data import DataLoader
import hydra
import torch
import einops
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import utils
import utils.picking as picking
from pathlib import Path

def hydraconf_autohandle_torch_device(f):
    def wrapper(conf):
        if conf.device == "CUDA_IF_AVAILABLE":
            conf.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        conf.device_type = 'cuda' if 'cuda' in conf.device else 'cpu'
        f(conf)
    return wrapper

def collate_fn(list_of_videos):
    # I want to do this:
    # return einops.rearrange(list_of_videos, "b t dunno1 dunno2 dunno3 -> (b t) dunno1 dunno2 dunno3")
    # unfortunately einops attempts to stack first, which still breaks if videos have different lengths like PHYRE
    b = torch.cat(list_of_videos, dim=0)
    # restore batch dimension
    return einops.rearrange(b, "t dunno1 dunno2 dunno3 -> 1 t dunno1 dunno2 dunno3")

@hydra.main(config_path="config", config_name="default")
@hydraconf_autohandle_torch_device
def main(conf):
    device = torch.device(conf.device)
    Path("ckpt").mkdir()

    log_dir = os.path.join(conf.path_constants.logs_dir, "tokenizer", datetime.today().isoformat())
    writer = SummaryWriter(log_dir)

    dataset = picking.pick_dataset(conf)
    dataloader = DataLoader(dataset, batch_size = 10, shuffle=True, collate_fn=collate_fn)

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)

    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=0.0001)

    i = 0

    for epoch in range(1):
        it = iter(dataloader)

        for batch in it:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, losses = tokenizer.compute_loss(batch)
            loss_total_step = losses.loss_total
            loss_total_step.backward()
            optimizer.step()

            print(f"i: {i:05} (ep{epoch})    loss: {loss_total_step.item():>7f}")

            writer.add_scalar('TRAIN/ae/loss', loss_total_step.item(), i)
            for k, v in losses.intermediate_losses.items():
                writer.add_scalar(f'TRAIN/ae/{k}', v, i)

            if i>=100 and i % 20 == 0:
                torch.save(tokenizer.state_dict(), "ckpt/tokenizer_last.pt")
                with torch.no_grad():
                    test_batch = next(it).to(device)
                    B, T, _, _, _ = test_batch.size()
                    images = einops.rearrange(test_batch, "b t c w h -> (b t) c w h")

                    outputs, losses = tokenizer.compute_loss(test_batch)
                    loss_total_eval = losses.loss_total
                    reconstruct = outputs.get_good_recons()

                    writer.add_scalar('EVAL/ae/loss', loss_total_eval.item(), i)
                    for k, v in losses.intermediate_losses.items():
                        writer.add_scalar(f'EVAL/ae/{k}', v, i)

                    images = einops.rearrange(images, "(b t) c w h -> b t c w h", b=B, t=T).clamp(0., 1.)
                    reconstruct = einops.rearrange(reconstruct, "(b t) c w h -> b t c w h", b=B, t=T).clamp(0., 1.)
                    frames = utils.visualize(images, reconstruct, MAX_BATCH=8)
                    writer.add_video('EVAL/ae_recons/epoch={:03}/i={:05}'.format(0, i), frames)

            i += 1



main()
