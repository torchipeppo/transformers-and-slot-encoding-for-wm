import torch
import torchvision.utils as vutils
import einops

def visualize(video, recon, attns=None, MAX_BATCH=8):
    B, T, C, H, W = video.size()

    if attns is not None: 
        slot_videos = make_masked_videos(video, attns, MAX_BATCH)

    frames = []
    for t in range(T):
        video_t = video[:MAX_BATCH, t, None, :, :, :]
        recon_t = recon[:MAX_BATCH, t, None, :, :, :]
        to_cat = [video_t, recon_t]
        nrow = 2

        if attns is not None:
            slot_videos_t = slot_videos[:MAX_BATCH, t, :, :, :, :]
            to_cat.append(slot_videos_t)
            nrow += slot_videos.shape[2]

        # tile
        tiles = torch.cat(to_cat, dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=nrow, pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames

def make_masked_videos(video, attns, MAX_BATCH=8):
    # unsqueeze an extra axis to align slot and channel as different dimensions,
    # allowing to broadcast both tensors in a single operation later.
    # Credits to the STEVE code [Singh2022] for the idea
    video = einops.rearrange(video[:MAX_BATCH], "b t c h w -> b t 1 c h w")
    attns = einops.rearrange(attns[:MAX_BATCH], "b t s h w -> b t s 1 h w")
    masked = video*attns + 0.5*(1.0-attns)
    return masked
