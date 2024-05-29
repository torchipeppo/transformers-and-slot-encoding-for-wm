import math
import os.path

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import einops

from dataset import PhyreVideoDataset
import dataset.common as datautils
from tokenizer_module import Tokenizer
import utils.picking as picking

from .steve import STEVE
from .utils import cosine_anneal, linear_warmup

import hydra

"""
cose che non posso fare :(

implementare batching e sub-batching (nope, va in out of memory)
implementare mascheramento in loss e rimettere make_batch_padded  (ma anche no, gli autori hanno previsto un troncamento molto più brutale, e questo è solo un confronto)
"""

def main(conf):
    device = torch.device(conf.device)
    torch.manual_seed(14383421)

    log_dir = os.path.join(conf.path_constants.logs_dir, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)

    train_dataset = picking.pick_dataset(conf)
    val_dataset = train_dataset

    loader_kwargs = {
        'batch_size': conf.training.subbatch_size,
        'shuffle': True,
        'num_workers': conf.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }

    collate_fn = datautils.collate_fn_factory(datautils.collate_with_same_targets, datautils.make_batch_truncated, labeled=conf.dataset.is_labeled)
    train_loader = DataLoader(train_dataset, sampler=None, collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=None, collate_fn=collate_fn, **loader_kwargs)

    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)

    log_interval = 10 # train_epoch_size // 5

    tokenizer : Tokenizer = hydra.utils.instantiate(conf.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(picking.pick_pretrained_tokenizer_path(conf), map_location=device))
    tokenizer.eval()

    model = STEVE(args=conf.steve_conf, tokenizer=tokenizer)

    if os.path.isfile(conf.checkpoint_path):
        checkpoint = torch.load(conf.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0

    model = model.cuda()
    if conf.use_dp:
        model = DP(model)

    optimizer = Adam([
        {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': conf.lr_dvae},
        {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
        {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
    ])

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])


    def visualize(video, recon_dvae, recon_tf, attns, N=8):
        B, T, C, H, W = video.size()

        frames = []
        for t in range(T):
            video_t = video[:N, t, None, :, :, :]
            recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
            recon_tf_t = recon_tf[:N, t, None, :, :, :]
            attns_t = attns[:N, t, :, :, :, :]

            # tile
            tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

            # grid
            frame = vutils.make_grid(tiles, nrow=(conf.steve_conf.num_slots + 3), pad_value=0.8)
            frames += [frame]

        frames = torch.stack(frames, dim=0).unsqueeze(0)

        return frames


    for epoch in range(start_epoch, conf.epochs):
        model.train()
        tokenizer.eval()
        
        for bat_idx, batch in enumerate(train_loader):
            global_step = epoch * train_epoch_size + bat_idx
            video = batch.input_image

            tau = cosine_anneal(
                global_step,
                conf.tau_start,
                conf.tau_final,
                0,
                conf.tau_steps)

            lr_warmup_factor_enc = linear_warmup(
                global_step,
                0.,
                1.0,
                0.,
                conf.lr_warmup_steps)

            lr_warmup_factor_dec = linear_warmup(
                global_step,
                0.,
                1.0,
                0,
                conf.lr_warmup_steps)

            lr_decay_factor = math.exp(global_step / conf.lr_half_life * math.log(0.5))

            optimizer.param_groups[0]['lr'] = conf.lr_dvae
            optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * conf.lr_enc
            optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * conf.lr_dec

            video = video.cuda()

            optimizer.zero_grad()
            
            (recon, cross_entropy, mse, attns) = model(video, tau, conf.hard)

            if conf.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            loss = mse + cross_entropy

            loss.backward()
            clip_grad_norm_(model.parameters(), conf.clip, 'inf')
            optimizer.step()
            
            with torch.no_grad():
                if bat_idx % log_interval == 0:
                    print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                        epoch+1, bat_idx, train_epoch_size, loss.item(), mse.item()))
                    
                    writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                    writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                    writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                    writer.add_scalar('TRAIN/tau', tau, global_step)
                    writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                    writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

    #     with torch.no_grad():
                    # gen_video = (model.module if conf.use_dp else model).reconstruct_autoregressive(video[:8])
                    # frames = visualize(video, recon, gen_video, attns, N=8)
                    # writer.add_video('TRAIN_recons/epoch={:03}/bat_idx={:05}'.format(epoch+1, bat_idx), frames)
        
        with torch.no_grad():
            model.eval()
            tokenizer.eval()

            val_cross_entropy = 0.
            val_mse = 0.

            for bat_idx, batch in enumerate(val_loader):
                video = batch.input_image
                video = video.cuda()

                (recon, cross_entropy, mse, attns) = model(video, tau, conf.hard)

                if conf.use_dp:
                    mse = mse.mean()
                    cross_entropy = cross_entropy.mean()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()

                if bat_idx % log_interval == 0:
                    print('Val Epoch: {:3} [{:5}/{:5}]'.format(
                        epoch+1, bat_idx, val_epoch_size))

            val_cross_entropy /= (val_epoch_size)
            val_mse /= (val_epoch_size)

            val_loss = val_mse + val_cross_entropy

            writer.add_scalar('VAL/loss', val_loss, epoch+1)
            writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
            writer.add_scalar('VAL/mse', val_mse, epoch+1)

            print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(model.module.state_dict() if conf.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

                if global_step < conf.steps:
                    torch.save(model.module.state_dict() if conf.use_dp else model.state_dict(), os.path.join(log_dir, f'best_model_until_{conf.steps}_steps.pt'))

                if 50 <= epoch:
                    # gen_video = (model.module if conf.use_dp else model).reconstruct_autoregressive(video[:8])
                    # frames = visualize(video, recon, gen_video, attns, N=8)
                    # writer.add_video('VAL_recons/epoch={:03}'.format(epoch + 1), frames)
                    pass

            writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model': model.module.state_dict() if conf.use_dp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

    writer.close()
