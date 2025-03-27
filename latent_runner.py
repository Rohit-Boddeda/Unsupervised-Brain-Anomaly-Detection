# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
import torch.nn as nn
import distributed_util as dist_util

from . import util
from .latent_network import DiffusionWrapper, DiscriminativeSubNetwork
from .diffusion import Diffusion
from .loss import FocalLoss, SSIM
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import Module

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def re_scale_mask(arr, dtype=torch.uint8):
    # arr = arr.clamp_(-1, 1)
    rescale_arr = (arr + 1) / 2
    rescale_arr = torch.round(rescale_arr).to(dtype)
    return rescale_arr

def re_scale_img(img_arr):
    rescale_img = (img_arr + 1) / 2
    rescale_img = rescale_img.to(torch.float)
    return rescale_img
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = re_scale_mask(inputs, dtype=torch.uint8)
        targets = re_scale_mask(targets, dtype=torch.uint8)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def build_optimizer_sched(opt, net, model_seg, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    
    optimizer = torch.optim.AdamW([
                                    {"params": net.parameters(), "lr": opt.lr, 'weight_decay': opt.l2_norm},
                                    {"params": model_seg.parameters(), "lr": opt.lr, 'weight_decay': opt.l2_norm}])
     
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()

class LatentRunner():
    def __init__(self, opt, log, save_opt=True):
        super(LatentRunner, self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(
            n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval
        )
        betas = np.concatenate(
            [betas[: opt.interval // 2], np.flip(betas[: opt.interval // 2])]
        )
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = (
            torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device)
            * opt.interval
        )
        self.net = DiffusionWrapper(
            log,
            noise_levels=noise_levels,
            ae_path="/home/lalith/Latent_SB/autoencoder/results/ae/latest.pt",
            use_fp16=opt.use_fp16,
            cond=opt.cond_x1,
        )

        self.net.instantiate_first_stage()
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=3)
        self.model_seg.apply(weights_init)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint["net"])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.model_seg.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """Eq 12"""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    def sample_batch(self, opt, loader):

        inp_img, gt_mask, aug_image, ano_mask, has_ano, class_label = next(loader)

        x0 = inp_img.to(opt.device, non_blocking=True)
        x1 = aug_image.to(opt.device, non_blocking=True)
        
        gt_mask = gt_mask.to(opt.device, non_blocking=True)
        ano_mask = ano_mask.to(opt.device, non_blocking=True)
        has_ano = has_ano.to(opt.device, non_blocking=True)
        class_label = class_label.to(opt.device, non_blocking=True)

        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, cond, gt_mask, ano_mask, has_ano, class_label

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log
        
        net = DDP(self.net, device_ids=[opt.device])
        model_seg = DDP(self.model_seg, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, model_seg, log)
        
        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader = util.setup_loader(val_dataset, batch_size=4)
        
        loss_ssim = SSIM()
        loss_dice = DiceLoss()
        
        net.train()
        model_seg.train()
        
        # n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        n_inner_loop = 8
        for it in range(opt.num_itr):
            optimizer.zero_grad()
            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, aug_img, cond, clean_mask, ano_mask, has_ano, class_label = self.sample_batch(opt, train_loader)

                # ===== get latent encoding pair =====
                x0_posterior = self.net.encode_first_stage(x0)
                x1_posterior = self.net.encode_first_stage(aug_img)
                cond_posterior = self.net.encode_first_stage(cond)

                x0 = self.net.get_first_stage_encoding(x0_posterior)
                x1 = self.net.get_first_stage_encoding(x1_posterior)
                cond = self.net.get_first_stage_encoding(cond_posterior)
                
                # ===== timestep =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                # ===== forward sample i2sb eq:11 =====
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                # ===== network input=====
                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape
                
                # ===== decode the output from latent unet =====
                pred_dec = self.net.decode_first_stage(pred)
                
                # ===== send the decoded output with augmented image to segmentation =====
                joined_in = torch.cat((re_scale_img(pred_dec), re_scale_img(aug_img)), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                
                # ===== compute loss for latent unet =====
                l2_loss = F.mse_loss(pred, label)
                ssim_loss = loss_ssim(pred, x0)
                # ===== compute loss for segmentation =====
                seg_loss = F.mse_loss(out_mask_sm, re_scale_mask(ano_mask, dtype=torch.float))
                dice_loss = loss_dice(out_mask_sm, ano_mask)
                # ===== compute total loss for latent unet and segmentation =====
                loss = l2_loss + ssim_loss + seg_loss + dice_loss
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None:
                sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{} | l2_loss:{} | ssim_loss:{} | seg_loss:{} | dice_loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.3f}".format(loss.item()),
                "{:+.3f}".format(l2_loss.item()),
                "{:+.3f}".format(ssim_loss.item()),
                "{:+.3f}".format(seg_loss.item()),
                "{:+.3f}".format(dice_loss.item()),
            ))
            
            if it % 10 == 0:
                self.writer.add_scalar(it, "total_loss", loss.detach())
                self.writer.add_scalar(it, "l2_loss", l2_loss.detach())
                self.writer.add_scalar(it, "ssim_loss", ssim_loss.detach())
                self.writer.add_scalar(it, "seg_loss", seg_loss.detach())
                self.writer.add_scalar(it, "dice_loss", dice_loss.detach())
                self.writer.add_scalar(it, "lr", optimizer.param_groups[0]["lr"])

            if it % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        'seg': self.model_seg.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()
            if it == 500 or it % 1000 == 0:  # 0, 0.5k, 3k, 6k 9k
                net.eval()
                self.evaluation(opt, it, val_loader, val_dataset)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(
        self,
        opt,
        x1,
        mask=None,
        cond=None,
        clip_denoise=False,
        nfe=10,
        log_count=10,
        verbose=True,
    ):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe + 1)

        # create log steps
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None:
            cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1.0 - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full(
                    (xt.shape[0],), step, device=opt.device, dtype=torch.long
                )
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps,
                pred_x0_fn,
                x1,
                mask=mask,
                ot_ode=opt.ot_ode,
                log_steps=log_steps,
                verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)
        return xs, pred_x0
    
    def all_cat_cpu(self, opt, log, t, decode=False):
        if not opt.distributed and decode == False:
            return t
        elif not opt.distributed and decode == True:
            return t
        elif opt.distributed and decode == False:
            return dist_util.all_gather(t.to(opt.device), log=log)
        elif opt.distributed and decode == True:
            gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
            return gathered_t
        else: 
            assert f"Something is wrong with the all_cat_cpu function"


    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, val_dataset):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt_org, cond, clean_mask, ano_mask, has_ano, class_label = self.sample_batch(opt, val_loader)

        # ===== get latent encoding pair =====
        img_corrupt_posterior = self.net.encode_first_stage(img_corrupt_org)
        cond_posterior = self.net.encode_first_stage(cond)

        img_corrupt = self.net.get_first_stage_encoding(img_corrupt_posterior)
        cond = self.net.get_first_stage_encoding(cond_posterior)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt,
            x1,
            mask=None,
            cond=cond,
            clip_denoise=opt.clip_denoise,
            verbose=opt.global_rank == 0,
        )

        log.info("Collecting tensors ...")
        img_clean = self.all_cat_cpu(opt, log, img_clean)
        img_corrupt_org = self.all_cat_cpu(opt, log, img_corrupt_org)
        xs = self.all_cat_cpu(opt, log, xs, decode=True)
        pred_x0s = self.all_cat_cpu(opt, log, pred_x0s, decode=True)
        
        batch, len_t, *xdim = xs.shape
        # assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        log.info(f"Generated recon trajectories: size={xs.shape}")
        
        # ===== do segmentation =====
        img_recon = self.net.decode_first_stage(xs[:, 0, ...].to(opt.device))
        joined_in = torch.cat((re_scale_img(img_recon).to(opt.device), 
                               re_scale_img(img_corrupt_org).to(opt.device)), dim=1).to(opt.device)
        out_mask = self.model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        
        out_mask_cv = out_mask_sm[..., 1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy()

        # ===== logging for tensorboard =====
        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img + 1) / 2, nrow=nrow))  # [1,1] -> [0,1]
            
        def log_seg_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid(img, nrow=nrow))

        log.info("Logging images ...")
        log_image("image/CleanImage", img_clean)
        log_image("image/AugmentedImage", img_corrupt_org)
        log_image("image/UnetRecon", img_recon)
        log_image("image/GtMask", clean_mask)
        log_image("image/AnoMask", ano_mask)
        log_seg_image("image/SegRecon", out_mask_sm)
        log_seg_image("image/Sigmod_SegRecon", F.sigmoid(torch.tensor(out_mask_sm)))
        log_seg_image("image/out_mask_cv", torch.tensor(out_mask_cv))
        log_seg_image("image/out_mask_averaged", torch.tensor(out_mask_averaged))
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj", xs.reshape(-1, *xdim), nrow=len_t)

        log.info(f"========== Evaluation finished: iter={it} ==========")

        torch.cuda.empty_cache()
