import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from supervised import *

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--port', default=None, type=int)
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

logger = init_log('global', logging.INFO)
logger.propagate = 0

rank, world_size = setup_distributed(port=args.port)

if rank == 0:
    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    writer = SummaryWriter(args.save_path)
    
    os.makedirs(args.save_path, exist_ok=True)

cudnn.enabled = True
cudnn.benchmark = True

model = DeepLabV3Plus(cfg)
if rank == 0:
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                    output_device=local_rank, find_unused_parameters=False)

# model = torch.load_state_dict(args.checkpoint)
valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

valsampler = torch.utils.data.distributed.DistributedSampler(valset)
valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                        drop_last=False, sampler=valsampler)


if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    previous_best = checkpoint['previous_best']
    
    if rank == 0:
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
eval_mode = "center_crop"

mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

if rank == 0:
    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
    writer.add_scalar('eval/mIoU', mIoU, epoch)
    for i, iou in enumerate(iou_class):
        writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
