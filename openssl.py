import argparse
import logging
import os
import pprint
from itertools import cycle

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter, MarginLoss, cluster_acc, accuracy, Logger, setup_seed, proto_graph, graph_cluster, reknn_graph
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='OpenSSL')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--reg', type=float, nargs='+', default=[1, 1], help='loss weights')
parser.add_argument('--labeled_num', default=10, type=int)
parser.add_argument('--nn', default=3, type=int)
parser.add_argument('--unknown_n_cls', action='store_true', help='action if n_classes is unknown')
parser.add_argument('--min_count', default=0, type=int)
parser.add_argument('--group_method', default='spectral', help='[louvain/connected/propagation] if unknown_n_cls')
parser.add_argument('--lamda_graph', default=1, type=float)

def flatten(x):
    return x.view(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous().view(-1, x.size(1))

def train(cfg, args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch):
    model.train()

    ent_losses = AverageMeter('ent_loss', ':.4e')
    cls_losses = AverageMeter('cls_loss', ':.4e')
    group_losses = AverageMeter('group_loss', ':.4e')
    proto_losses = AverageMeter('proto_loss', ':.4e')
    
    loader = zip(train_label_loader, train_unlabel_loader)
    total_iters = len(train_unlabel_loader) * cfg['epochs']
    for i, ((img_x, img_x2, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box, _)) in enumerate(loader):

        img_x, img_x2, mask_x = img_x.cuda(), img_x2.cuda(), mask_x.cuda()
        img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
        ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
                    
        model.train()
        
        loss_dic = model.module.loss(img_x, img_x2.clone(), mask_x, img_u_s1, img_u_s2)
        loss = loss_dic['proto'] + loss_dic['group'] +  args.reg[0] * loss_dic['ent']\
            + args.reg[1] * loss_dic['cls']
            
        proto_losses.update(loss_dic['proto'].item(), cfg['batch_size'])
        group_losses.update(loss_dic['group'].item(), cfg['batch_size'])
        cls_losses.update(args.reg[1] * loss_dic['cls'].item(), cfg['batch_size'])
        ent_losses.update(args.reg[0] * loss_dic['ent'].item(), cfg['batch_size'])

        torch.distributed.barrier()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iters = epoch * len(train_unlabel_loader) + i
        lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
        
        if i % 200 == 0:
            print('Train loss = protosim({:.3f}) + groupsim({:.3f}) + cls({:.3f}) + ent({:.3f})'\
                  .format(proto_losses.avg,  group_losses.avg, cls_losses.avg, ent_losses.avg))



def val(args, model, device, train_label_loader, train_unlabel_loader, epoch, n_cls, old_graph):
    model.eval()
    features_l = []
    features_u = []
    targets_l = []
    # targets_u = []
    unlabel_loader_iter = cycle(train_unlabel_loader)
    # print(len(train_label_loader))
    with torch.no_grad():
        for batch_idx, (x_l, _, y_l) in enumerate(train_label_loader):
            (_, x_u, _, _, _, _) = next(unlabel_loader_iter)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            
            _, feature_l  = model(x_l)
            _, feature_u  = model(x_u)
            
            y_l = F.interpolate(y_l.unsqueeze(1).float(), size=(feature_l.size(2), feature_l.size(3)), mode="nearest")
            feature_l = flatten(feature_l)
            feature_u = flatten(feature_u)
            features_l.append(feature_l.cpu())
            features_u.append(feature_u.cpu())
            
            y_l = y_l.view(-1).long()
            targets_l.append(y_l.cpu())
            if batch_idx == 200:
                break

        # targets_u = torch.hstack(targets_u)
        features_l = torch.vstack(features_l)
        features_u = torch.vstack(features_u)
        targets_l = torch.hstack(targets_l)
        features = torch.cat((features_l,features_u),0)
        # targets = torch.cat((targets_l,targets_u),0)

        prototypes = model.module.prototypes[model.module.proto_ind]
        prototypes = F.normalize(prototypes, dim=1).cpu()
        features = F.normalize(features, dim=1)

        dist_matrix = torch.mm(features, prototypes.t())
        edge_graph = proto_graph(dist_matrix, args.nn)

        ind = edge_graph.diagonal() >= args.min_count
        edge_graph = edge_graph[ind,:][:,ind]

        dist_matrix = dist_matrix[:,ind]
        proto_ind = model.module.proto_ind.clone()
        proto_ind[proto_ind==True] = ind
        prototypes = prototypes[ind]
        
        seed = 0
        edge_graph = reknn_graph(dist_matrix, args.nn, mode='min')

        def group_discovery(edge_graph, prototypes, args, targets_l, eps=0, n_cls=10):
            proto_label, proto_mask = graph_cluster(edge_graph, prototypes, lamda=args.lamda_graph, method=args.group_method, seed=seed, n_cls=n_cls, eps=eps)
            group_label, group_index = np.unique(proto_label, return_index=True)
            group_mask = proto_mask[group_index]

            # match y_l to proto group label
            preds_proto = np.argmax(dist_matrix.cpu().numpy(),1)
            preds_group = proto_label[preds_proto]
            
            preds_group_l = preds_group[:len(features_l)]
            contingency_matrix = metrics.cluster.contingency_matrix(targets_l, preds_group_l)
            target_ind, group_label_new_l = linear_sum_assignment(contingency_matrix.max() - contingency_matrix)
            
            group_label_new = np.r_[group_label[group_label_new_l], np.delete(group_label, group_label_new_l, axis=0)]
            group_mask_new = group_mask[group_label_new]

            mp = group_label.copy()
            mp[group_label_new] = group_label
            pred_ordered = mp[preds_group_l]
            acc = accuracy(pred_ordered, targets_l)
            return acc,  preds_proto, proto_mask, group_mask, group_mask_new
        
        acc_best = 0
        n_cls_best = n_cls
        proto_mask_best, group_mask_new_best = None, None

        if args.unknown_n_cls:
            if args.group_method in ['propagation', 'connected'] and args.dataset=='cifar10':
                eps_min = 0.5
                eps_max = 0.99
            elif args.group_method in ['louvain'] and args.dataset=='cifar10':
                eps_min = 2 # 0.5
                eps_max = 12
            elif args.group_method in ['louvain'] and args.dataset=='cifar100':
                eps_min = 6
                eps_max = 12
            else:
                raise Exception("Invalid clustering method. ['propagation', 'connected', 'louvain'] for cifar10, ['louvain'] for cifar100.")          
            grouping_epochs = args.fix_epoch - args.warm_epoch
            eps_array = [((eps_min/eps_max)**((i) / (grouping_epochs-1)))*eps_max for i in range(grouping_epochs)]
            
            for eps_t in eps_array[:(epoch-args.warm_epoch+1)]:
                acc, _, proto_mask, _, group_mask_new = group_discovery(edge_graph, prototypes, args, targets_l, eps_t)
                print('EPS:{:.2f}, NUM:{:d}, ACC:{:.4f}'.format(eps_t, group_mask_new.shape[0], acc))
                if acc > acc_best:
                    acc_best = acc
                    n_cls_best = group_mask_new.shape[0]
                    proto_mask_best = proto_mask
                    group_mask_new_best = group_mask_new
            print('** Best Group_num {} **'.format(n_cls_best))
        else:
            acc_best, _, proto_mask_best, _, group_mask_new_best = group_discovery(edge_graph, prototypes, args, targets_l, n_cls=n_cls)

        return edge_graph, proto_mask_best, group_mask_new_best, proto_ind, acc_best

def test(args, model, labeled_num, device, test_loader, epoch):
    model.eval()
    preds = np.array([])
    features = []
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, id) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            pred, _, feature = model.module.pred(x)
            # print(label.shape)

            features.append(feature)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            # confs = np.append(confs, conf.cpu().numpy())

    features = torch.cat(features, 0)
    targets = targets.astype(int)
    preds = preds.astype(int)
    features = features.cpu().numpy()

    # print(preds.shape, targets.shape, features.shape)

    known_mask = targets < labeled_num
    unknown_mask = ~known_mask

    all_acc, row_acc_all = cluster_acc(preds, targets)
    known_acc = accuracy(preds[known_mask], targets[known_mask])
    unknown_acc, row_acc = cluster_acc(preds[unknown_mask], targets[unknown_mask])
    
    print('Test All ACC {:.4f}, Known ACC {:.4f}, Unknown ACC {:.4f}'.format(all_acc, known_acc, unknown_acc))
    print(row_acc_all)
    return all_acc, known_acc, unknown_acc


def main():
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
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'MarginLoss':
        criterion_l = MarginLoss(m=-0.5).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    n_classes = cfg['nclass']
    
    criterion_bce = nn.BCELoss().cuda()
    device = torch.device('cuda')

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        
        if epoch < cfg["warm_epoch"]: # warm up
            print('===========> Epoch {} Warming stage'.format(epoch))
            args.reg[1] = 0
        elif epoch < cfg["fix_epoch"]:
            print('===========> Epoch {} Grouping stage'.format(epoch))
            args.reg[1] = 0.0 if args.unknown_n_cls else 1
            proto_graph, proto_mask, group_mask, proto_ind, acc = val(args, model, device, trainloader_l, trainloader_u, epoch, n_classes, model.module.proto_mask)
            model.module.proto_graph = proto_graph.to(device)
            model.module.proto_mask = torch.tensor(proto_mask).to(device)
            model.module.group_mask = torch.tensor(group_mask).to(device)
            model.module.proto_ind = proto_ind
            print('Current GroupNum:{}'.format(model.module.group_mask.shape[0]))  
        else:
            print('===========> Epoch {} Fixing stage'.format(epoch)) 
            args.reg[1] = 1
            pass

            
        train(cfg, args, model, device, trainloader_l, trainloader_u, optimizer, epoch)
        all_acc, known_acc, unknown_acc = \
            test(args, model, args.labeled_num, device, valloader, epoch)
        
        # scheduler.step()

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                

        
if __name__ == '__main__':
    main()
