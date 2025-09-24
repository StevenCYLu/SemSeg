import argparse
from copy import deepcopy
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus_wass import DeepLabV3Plus
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from util.wass import FastApproxSWDLoss


def create_criterion(cfg, local_rank):
    """Create loss criteria based on configuration"""
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError(f"{cfg['criterion']['name']} criterion is not implemented")

    if cfg["unsup_loss"] == "CE":
        criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
        criterion_swd = None
    elif cfg["unsup_loss"] == "SWD_FM":
        criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
        criterion_swd = FastApproxSWDLoss(reduction="mean").cuda(local_rank)
    else:
        criterion_u = FastApproxSWDLoss(reduction="mean").cuda(local_rank)
        criterion_swd = None

    return criterion_l, criterion_u, criterion_swd


def create_dataloaders(cfg, args):
    """Create training and validation dataloaders"""
    trainset_u = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"],
        args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg["batch_size"], pin_memory=True,
        num_workers=2, drop_last=True, sampler=trainsampler_l
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg["batch_size"], pin_memory=True,
        num_workers=2, drop_last=True, sampler=trainsampler_u
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True,
        num_workers=2, drop_last=False, sampler=valsampler
    )

    return trainloader_l, trainloader_u, valloader


def update_learning_rate(optimizer, cfg, iters, total_iters):
    """Update learning rate with polynomial decay"""
    lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]
    return lr


def compute_unsupervised_loss(pred_u_s, pred_u_s_ft, pred_u_w_ft, mask_u_w_cutmixed,
                             conf_u_w_cutmixed, ignore_mask_cutmixed, cfg,
                             criterion_u, criterion_swd, args):
    """Compute unsupervised loss based on configuration"""
    loss_u_swd = torch.tensor(0.0).cuda()

    if cfg["unsup_loss"] == "SWD":
        pred_u_s = pred_u_s.softmax(dim=1)
        pred_u_w_cutmixed = pred_u_w_cutmixed.softmax(dim=1)

        mask_confident = ((conf_u_w_cutmixed >= cfg["conf_thresh"]) &
                         (ignore_mask_cutmixed != 255)).unsqueeze(1)
        pred_u_s = pred_u_s * mask_confident
        pred_u_w_cutmixed = pred_u_w_cutmixed * mask_confident

        loss_u_s = criterion_u(pred_u_s, pred_u_w_cutmixed.detach()) * args.unsup_weight

    elif cfg["unsup_loss"] == "SWD_FM":
        mask_confident = ((conf_u_w_cutmixed >= cfg["conf_thresh"]) &
                         (ignore_mask_cutmixed != 255))

        loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed.detach())
        loss_u_s = loss_u_s * mask_confident
        loss_u_s = torch.sum(loss_u_s) / torch.sum(ignore_mask_cutmixed != 255).item() * 1.5

        loss_u_swd = criterion_swd(pred_u_s_ft, pred_u_w_ft.detach()) * cfg["unsup_weight"]
        loss_u_s += loss_u_swd

    else:
        mask_confident = ((conf_u_w_cutmixed >= cfg["conf_thresh"]) &
                         (ignore_mask_cutmixed != 255))

        loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed.detach())
        loss_u_s = loss_u_s * mask_confident
        loss_u_s = torch.sum(loss_u_s) / torch.sum(ignore_mask_cutmixed != 255).item()

    return loss_u_s, loss_u_swd


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--unsup_weight", type=float)
parser.add_argument("--temp", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--DA", type=bool, default=False)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--resume", default=False, type=bool)
parser.add_argument("--ema", default=0.9996, type=float)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    # initialize the process group
    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": cfg["lr"]},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name
                ],
                "lr": cfg["lr"] * cfg["lr_multi"],
            },
        ],
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )
    local_rank = int(os.environ["LOCAL_RANK"])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    sup_only = True if cfg["sup_epochs"] > 0 else False

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=sup_only,
    )


    criterion_l, criterion_u, criterion_swd = create_criterion(cfg, local_rank)
    if cfg["unsup_loss"] == "SWD_FM":
        criterion_ce = criterion_u

    trainloader_l, trainloader_u, valloader = create_dataloaders(cfg, args)

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best
                )
            )

        total_loss, total_loss_x, total_loss_s, total_loss_swd = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        if epoch < cfg["sup_epochs"]:
            for i, (img_x, mask_x) in enumerate(trainloader_l):
                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                model.train()

                pred_x = model(img_x)
                loss = criterion_l(pred_x, mask_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters = epoch * len(trainloader_l) + i
                update_learning_rate(optimizer, cfg, iters, total_iters)

                if (i % (len(trainloader_l) // 8) == 0) and (rank == 0):
                    logger.info(
                        "Iters: {:}, Total loss: {:.3f}".format(i, total_loss / (i + 1))
                    )

        else:
            for i, (
                (img_x, mask_x),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
                (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _),
            ) in enumerate(loader):

                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
                ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
                img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
                ignore_mask_mix = ignore_mask_mix.cuda()

                with torch.no_grad():
                    pred_u_w_mix = model(img_u_w_mix).detach()
                    conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = (
                    img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
                )

                model.train()

                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

                pred_x = model(img_x)
                pred_u_s, pred_u_s_ft = model(img_u_s, need_ft=True, pred=True)

                with torch.no_grad():
                    pred_u_w, pred_u_w_ft = model(img_u_w, need_ft=True)
                pred_u_w_ft = pred_u_w_ft.detach()

                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

                mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = (
                    mask_u_w.clone(),
                    conf_u_w.clone(),
                    ignore_mask.clone(),
                )

                mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
                conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
                ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

                loss_x = criterion_l(pred_x, mask_x)

                loss_u_s, loss_u_swd = compute_unsupervised_loss(
                    pred_u_s, pred_u_s_ft, pred_u_w_ft, mask_u_w_cutmixed,
                    conf_u_w_cutmixed, ignore_mask_cutmixed, cfg,
                    criterion_u if cfg["unsup_loss"] != "SWD_FM" else criterion_ce,
                    criterion_swd, args
                )

                loss = loss_x + loss_u_s
                torch.distributed.barrier()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_x += loss_x.item()
                total_loss_s += loss_u_s.item()
                total_loss_swd += loss_u_swd.item()
                mask_ratio = ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum().item()
                total_mask_ratio += mask_ratio

                iters = epoch * len(trainloader_u) + i
                update_learning_rate(optimizer, cfg, iters, total_iters)

                if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                    logger.info(
                        "Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, "
                        "Loss s: {:.3f}, Loss swd: {:.3f} Mask: {:.3f}".format(
                            i,
                            total_loss / (i + 1),
                            total_loss_x / (i + 1),
                            total_loss_s / (i + 1),
                            total_loss_swd / (i + 1),
                            total_mask_ratio / (i + 1),
                        )
                    )
        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        if rank == 0:
            logger.info(
                "***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n".format(
                    eval_mode, mIOU
                )
            )

        is_best = mIOU > previous_best
        previous_best = max(mIOU, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
