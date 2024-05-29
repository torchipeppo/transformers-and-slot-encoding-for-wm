from dvae_steve.dvae import dVAE
import dvae_steve.utils as steve_utils
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import einops
import matplotlib.pyplot as plt
from dataset import PhyreVideoDataset
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import utils

from omegaconf import OmegaConf
conf = OmegaConf.merge(
    OmegaConf.load("config/default.yaml"),
    OmegaConf.load("config/path_constants/path_constants.yaml"),
    OmegaConf.load("config/dvae/default.yaml"),
)


def encode_decode(dvae, video, tau, hard):
    B, T, C, H, W = video.size()

    video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W

    # dvae encode
    z_logits = F.log_softmax(dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
    z_soft = steve_utils.gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc

    # dvae recon
    dvae_recon = dvae.decoder(z_soft).reshape(B, T, C, H, W)               # B, T, C, H, W
    dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1

    return dvae_recon.clamp(0., 1.), dvae_mse


log_dir = os.path.join("logs/", datetime.today().isoformat())
writer = SummaryWriter(log_dir)

dataset = PhyreVideoDataset(conf.phyre_video_dataset)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

dvae = dVAE(conf.vocab_size, conf.img_channels)

optimizer = torch.optim.Adam((x[1] for x in dvae.named_parameters()), lr=conf.lr)

i = 0
it = iter(dataloader)

dvae.train()

for video in it:

    global_step = i

    tau = steve_utils.cosine_anneal(
        global_step,
        conf.SCHEDULE_ARGS.tau_start,
        conf.SCHEDULE_ARGS.tau_final,
        0,
        conf.SCHEDULE_ARGS.tau_steps)

    optimizer.zero_grad()

    _, loss = encode_decode(dvae, video, tau, conf.HARD)

    loss.backward()
    clip_grad_norm_(dvae.parameters(), 0.05, 'inf')
    optimizer.step()

    print(f"i={i:05}  loss: {loss.item():>7f}")

    with torch.no_grad():
        if i>=100 and i % 20 == 0:
            writer.add_scalar('TRAIN/loss', loss.item(), global_step)
            writer.add_scalar('TRAIN/tau', tau, global_step)

            images = next(it)

            reconstruct, _ = encode_decode(dvae, images, tau, conf.HARD)
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(einops.rearrange(images[0][0].detach(), "c w h -> w h c"))
            axarr[1].imshow(einops.rearrange(reconstruct[0][0].detach(), "c w h -> w h c"))
            plt.show()

            frames = utils.visualize(images, reconstruct, MAX_BATCH=8)
            writer.add_video('TRAIN_recons/epoch={:03}/i={:05}'.format(0, i), frames)

    i += 1
