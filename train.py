""" Trains """
#In[]: Bibliotecas
import math
import sys
import os
# import urllib.request
# from functools import partial
# from urllib.error import HTTPError
# import random
# import sparse
import argparse
# from enum import Enum
import time
from datetime import datetime
import warnings
import pickle
# import builtins
import shutil
import pandas as pd
import csv

# Images
# from PIL import Image
# import imageio.v2 as imageio

# Plotting
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib_inline.backend_inline
import numpy as np
# import seaborn as sns
import torch.distributed

# PyTorch Lightning
# import lightning as L

# PyTorch
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
# from torch.autograd import Variable
# from torch.cuda.amp import autocast,GradScaler
import pytorch_msssim

# Torchvision
# import torchvision
# from lightning.pytorch.callbacks import ModelCheckpoint
# from torchvision import datasets, transforms
# from torchvision.datasets import CIFAR100
# from tqdm.notebook import tqdm
# from torchvision.models import efficientnet_b4, resnet50
# from torchvision.models.feature_extraction import create_feature_extractor

# timm
# from timm import layers, models

# from models.ViT_Basis import *
from datasets.Brooklyn_n_Queens import *
from models.transformer_model import Create_Model
from models.unet import UNet
from optimizers.make_optimizer import *
from criterion.soft_triplet import *
from criterion.dice_loss import Dice_Loss
# from losses.cal_loss import cal_kl_loss,cal_loss,cal_triplet_loss

# from scipy.stats import wasserstein_distance

#In[]: Parser-
parser = argparse.ArgumentParser(description='CVSegGuide')

parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load_ckpt', default='/saved_models/', type=str, metavar='PATH',
                    help='load checkpoint')
parser.add_argument('--save_path', default='/saves/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--autocast', action='store_true',
                    default=True, help='use mix precision' )
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--cross', action='store_true',
                    help='use cross area')
parser.add_argument('--dataset', default='brooklyn&queens', type=str,
                    help='vigor, cvusa, cvact')
parser.add_argument('--op', default='adam', type=str,
                    help='sgd, adam, adamw')
parser.add_argument('--share', action='store_true',
                    help='share fc')
parser.add_argument('--mining', action='store_true',
                    help='mining')
parser.add_argument('--asam', action='store_true',
                    help='asam')
parser.add_argument('--rho', default=0.05, type=float,
                    help='rho for sam')
parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')
parser.add_argument('--segmentation', action='store_true',
                    help='instance segmentation')
parser.add_argument('--fov', default=0, type=int,
                    help='Fov')
parser.add_argument('--root', default="/Brooklyn_n_Queens/brooklyn_queens/", type=str,
                    help='Dataset Root')
parser.add_argument('--img_dict_path', default="/datasets/", type=str,
                    help='Aerial Heatmaps')
parser.add_argument('--sample', default=1, type=int,
                    help='Number of samples')
parser.add_argument('--val_split', default=-1, type=float,
                    help='Number of samples')
parser.add_argument('--city', type=str,
                    help='City')
parser.add_argument('--heat_blocks', default=4, type=float,
                    help='Number of samples')
parser.add_argument('--dim', default=1024, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--bins', default=1000, type=int,
                    help='Bins')
parser.add_argument('--patch-size', default=16, type=int,
                    help='Patch Size')
parser.add_argument('--aerial-patch-size', default=16, type=int,
                    help='Aerial Patch Size')
parser.add_argument('--street-patch-size', default=16, type=int,
                    help='Street Patch Size')
parser.add_argument('--embed-dim', default=384, type=int,
                    help='Embedding Dimension')
parser.add_argument('--aerial-embed-dim', default=384, type=int,
                    help='Aerial Embedding Dimension')
parser.add_argument('--street-embed-dim', default=384, type=int,
                    help='Street Embedding Dimension')
parser.add_argument('--classes', default=4, type=int,
                    help='Number of classes for segmentation')
parser.add_argument('--total-classes', default=8, type=int,
                    help='Number of Classes at Landcover')
parser.add_argument('--segments', default=[0,4,5,6], type=int,
                    help='Segmented Classes')
parser.add_argument('--base', default="featup", type=str,
                    help='Transformer Base: featup or mst')
parser.add_argument('--ckpt', default="_model_best.pth.tar", type=str,
                    help='Checkpoint')


#In[]: main()
best_dist = 0
def main():
  base = "CVSegGuide"
  print("\n\n{}\n\n".format(base))
  global best_dist
  warnings.filterwarnings("ignore")

  args = parser.parse_args()
  print(args)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ngpus_per_node = torch.cuda.device_count()
  print("\nUse GPU: {} for training".format(args.gpu))
  print("Num GPUs available: {}".format(ngpus_per_node))

  # Create model
  print("\n-> Creating model")
  model = Create_Model(args=args).cuda(args.gpu)
  # model = UNet(n_channels=3, n_classes=args.classes, bilinear=True).cuda(args.gpu)

  # DataParallel will divide and allocate batch_size to all available GPUs
  # model = torch.nn.DataParallel(model).cuda(args.gpu)

  parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

  # Criterion
  # compute_complexity(model, args)  # uncomment to see detailed computation cost
  criterion_triplet = SoftTripletBiLoss().cuda(args.gpu)
  criterion_BCE = nn.BCEWithLogitsLoss().cuda(args.gpu) #Num Class = 1
  criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
  criterion_SSIM = pytorch_msssim.SSIM(data_range=1.0, channel=1)
  criterion_Dice = Dice_Loss().cuda(args.gpu)
  criterion_MSE = nn.MSELoss().cuda(args.gpu)

  # Optimizer
  optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
  # optimizer = torch.optim.RMSprop(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, foreach=True)

  # Load checkpoint
  if(f"{base}_{args.base}{args.ckpt}" in os.listdir(args.load_ckpt)):
    checkpoint = torch.load(f"{args.load_ckpt}/{base}_{args.base}{args.ckpt}")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"\nLoaded checkpoint: Epoch {checkpoint['epoch']}")

  # Street-Aerial Matches Brooklyn
  img_dict_path = os.path.join(f"{args.img_dict_path}brooklyn_dict.pkl")
  if os.path.exists(img_dict_path):
    with open(img_dict_path, 'rb') as fp:
        train_dict = pickle.load(fp)
    print('Image dictionary loaded successfully')
  else:
    train_dict = None
  
  # Training Dataset -> Brooklyn
  args.size_sat = (256, 256)
  args.size_sat_default = (256, 256)
  args.size_gnd = (128, 512)
  args.size_gnd_default = (1664, 3328)
  args.city = "brooklyn"
  _, train_loader, train_sampler, _, _, _, dataset_sizes = make_dataset(args,
                                                                        img_dict=train_dict,
                                                                        labels=["overhead", "streetview", "landcover", "depth"])

  print("\nTraining Dataset:\n", dataset_sizes)
  args.train_aerial_length = dataset_sizes["overhead"]
  args.train_street_length = dataset_sizes["streetview"]

  # Street-Aerial Matches Queens
  img_dict_path = os.path.join(f"{args.img_dict_path}queens_dict.pkl")
  if os.path.exists(img_dict_path):
    with open(img_dict_path, 'rb') as fp:
        val_dict = pickle.load(fp)
    print('Image dictionary loaded successfully')
  else:
    val_dict = None
  
  # Validation Dataset
  args.city = "queens"
  _, val_loader, _, _, _, _, dataset_sizes = make_dataset(args,
                                                          img_dict=val_dict,
                                                          labels=["overhead", "streetview", "landcover", "depth"])
  print("\nValidation Dataset:\n", dataset_sizes)
  args.valid_aerial_length = dataset_sizes["overhead"]
  args.valid_street_length = dataset_sizes["streetview"]

  args.time = time.strftime("%Y-%m-%d", time.gmtime())
  for epoch in range(args.start_epoch, args.epochs):
    print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
    adjust_learning_rate(optimizer, epoch, args)

    print("\n\nTrain")
    loss = train(train_loader, model, criterion_MSE, [criterion_CE, criterion_Dice], [criterion_BCE, criterion_Dice], optimizer, epoch, args, train_sampler=train_sampler)

    print("\n\nValidate")
    evaluation, dist = validate(val_loader, model, criterion_MSE, criterion_Dice, criterion_BCE, args)
    is_best = dist < best_dist
    best_dist = max(dist, best_dist)

    save_csv(epoch+1, loss, evaluation, 
             filename=f"{args.base}_{args.time}.csv", args=args)
  
    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc1': best_dist,
      'optimizer': optimizer.state_dict(),
    }, is_best, filename=f"{args.base}_{args.time}_checkpoint_{epoch}.pth.tar", args=args) # 'checkpoint_{:04d}.pth.tar'.format(epoch)

#In[]: train
def train(train_loader, model, criterion_pos, criterion_lab, criterion_dep, optimizer, epoch, args, train_sampler=None):
  batch_time = AverageMeter('Time:', ':.2f')
  data_time = AverageMeter('Data:', ':.2f')
  losses_ar_heat = AverageMeter('Loss_ar:', ':.2e')
  losses_st_heat = AverageMeter('Loss_st:', ':.2e')
  losses_pos = AverageMeter('Loss_pos:', ':.2e')
  losses = AverageMeter('Total Loss:', ':.4e')
  progress = ProgressMeter(
      args.train_aerial_length//args.batch_size,
      [losses_ar_heat, losses_st_heat, losses_pos, losses],
      prefix="Epoch: [{}]".format(epoch))

  # switch to train mode
  model.train()

  end = time.time()

  args.max_loss_pos = 1
  for i, image_data in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    indexes = image_data[0].cuda(non_blocking=True)
    aerial_img = image_data[1].cuda(non_blocking=True)
    # aerial_box = image_data[2].cuda(non_blocking=True)
    street_img = image_data[3].cuda(non_blocking=True)
    # street_pos = image_data[4].cuda(non_blocking=True)
    label_data = []
    for label in image_data[5:]:
      if not torch.all(label==0):
        label_data += [label]
    
    del image_data

    """Compute Output"""
    aerial_labelmap, street_depthmap = model(aerial_img, street_img)
    real_mask = nn.functional.one_hot(label_data[0][:,0].to(torch.long), args.total_classes).permute(0, 3, 1, 2).float()
    real_mask = real_mask[:, args.segments, :]
    pos_pred = torch.stack([torch.stack([(street_depthmap[k][j]==torch.max(street_depthmap[k][j])).nonzero()[0] for j in range(street_depthmap.size()[1])]) for k in range(street_depthmap.size()[0])])
    pos_pred = pos_pred[:,0,:]
    pos_real = torch.stack([torch.stack([(label_data[-1][k][j]==torch.max(label_data[-1][k][j])).nonzero()[0] for j in range(label_data[-1].size()[1])]) for k in range(label_data[-1].size()[0])])
    pos_real = pos_real[:,0,:]
    
    """Loss Position"""
    loss_pos = criterion_pos(pos_pred.to(torch.float).cuda(args.gpu),
                             pos_real.to(torch.float).cuda(args.gpu))
    args.max_loss_pos = loss_pos if loss_pos>args.max_loss_pos else args.max_loss_pos
    loss_pos = loss_pos/args.max_loss_pos

    """Loss Aerial Image"""
    loss_ar_heat = 0
    loss_ar_heat = criterion_lab[0](aerial_labelmap.cuda(args.gpu),
                                    real_mask.cuda(args.gpu))
    loss_ar_heat += criterion_lab[1](aerial_labelmap.cuda(args.gpu),
                                    real_mask.cuda(args.gpu))
    
    """Loss Street Image"""
    loss_st_heat = 0
    loss_st_heat = criterion_dep[0](street_depthmap.cuda(args.gpu),
                                    label_data[-1].cuda(args.gpu))
    loss = (loss_ar_heat +5*loss_st_heat +10*loss_pos)

    losses_ar_heat.update(loss_ar_heat, street_img.size(0))
    losses_st_heat.update(loss_st_heat, street_img.size(0))
    losses_pos.update(loss_pos, street_img.size(0))
    losses.update(loss, street_img.size(0))

    """Compute Gradient"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      progress.display(i)
      print(f"\t\t\tLosses Ar [{criterion_lab[0](aerial_labelmap.cuda(args.gpu),real_mask.cuda(args.gpu)):.3f}, {criterion_lab[1](aerial_labelmap.cuda(args.gpu),real_mask.cuda(args.gpu)):.3f}]",
            f"\tLosses St [{criterion_dep[0](street_depthmap.cuda(args.gpu),label_data[-1].cuda(args.gpu)):.3f}, {criterion_dep[1](street_depthmap.cuda(args.gpu),label_data[-1].cuda(args.gpu)):.3f}]",
            f"\tLosses Pos [{loss_pos:.3f}]")
      
      print(f"\t\t\tAerial [{aerial_labelmap.min().item():.3f}, {aerial_labelmap.max().item():.3f}]",
            f"\t\tStreet [{street_depthmap.min().item():.3f}, {street_depthmap.max().item():.3f}]")

    del loss
    del indexes
    del aerial_img
    del street_img
    del label_data
    del real_mask
    del aerial_labelmap
    del street_depthmap

  return [losses_ar_heat.save().detach().cpu().numpy(),
          losses_st_heat.save().detach().cpu().numpy(),
          losses_pos.save().detach().cpu().numpy(),
          losses.save().detach().cpu().numpy()]

#In[]: validate
def validate(val_loader, model, criterion_pos, criterion_lab, criterion_dep, args):
  batch_time = AverageMeter('Time', ':6.3f')
  map_k = AverageMeter('Segmented Map:', ':.2f')
  progress_k = ProgressMeter(
    args.valid_aerial_length//args.batch_size,
    [batch_time, map_k],
    prefix='Test_reference: ')
  map_q = AverageMeter('Depth Map:', ':.2e')
  progress_q = ProgressMeter(
    args.valid_aerial_length//args.batch_size,
    [batch_time, map_q],
    prefix='Test_query: ')

  model.cuda(args.gpu).eval()

  reference_pos = np.zeros([args.valid_aerial_length, 2])
  query_pos = np.zeros([args.valid_aerial_length, 2])

  with torch.no_grad():
    end = time.time()

    # reference features
    for i, image_data in enumerate(val_loader):
      indexes = image_data[0].cpu().numpy().astype(int)
      aerial_img = image_data[1].cuda(non_blocking=True)
      street_img = image_data[3].cuda(non_blocking=True)
      street_pos = image_data[4]
      label_data = []
      for label in image_data[5:]:
        if not torch.all(label==0):
          label_data += [label]

      del image_data

      """Compute Output"""
      _, street_depthmap = model(aerial_img, street_img)
      pos_pred = torch.stack([torch.stack([(street_depthmap[k][j]==torch.max(street_depthmap[k][j])).nonzero()[0] for j in range(street_depthmap.size()[1])]) for k in range(street_depthmap.size()[0])])
      pos_pred = pos_pred[:,0,:]
      pos_real = torch.stack([torch.stack([(label_data[-1][k][j]==torch.max(label_data[-1][k][j])).nonzero()[0] for j in range(label_data[-1].size()[1])]) for k in range(label_data[-1].size()[0])])
      pos_real = pos_real[:,0,:]

      for j, index in enumerate(indexes):
        reference_pos[index] = pos_real.detach().cpu().numpy()[j]
        query_pos[index] = pos_pred.detach().cpu().numpy()[j]        

      # measure elapsed time
      batch_time.update(time.time() - end)

      end = time.time()

      del indexes
      del aerial_img
      del street_img
      del label_data
      del street_depthmap
    dist = np.linalg.norm(reference_pos -query_pos, axis=1)
  return [np.mean(dist), np.median(dist)], dist.mean()

#In[]: scan
# save all the attention map
def scan(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_q, images_k, _, indexes , delta, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_k, embed_q = model(im_q =images_q, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

#In[]: save_csv
def save_csv(epoch, loss, evaluation, filename='checkpoint.csv', args=None):
  filename = f"{args.save_path}/{filename}"
  if (not os.path.exists(filename)):
    header = []
    header += ["epoch"]
    for i in range(len(loss)):
      header += [f"loss {i}"]
    for i in range(len(evaluation)):
      header += [f"evaluate {i}"]
    with open(filename, "w") as f:
      writer = csv.writer(f)
      writer.writerow(header)
  
  data = []
  data += [epoch]
  for i in range(len(loss)):
    data += [loss[i]]
  for i in range(len(evaluation)):
    data += [evaluation[i]]
  with open(filename, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(data)

#In[]: save_checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    torch.save(state, os.path.join(args.save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(args.save_path,filename), os.path.join(args.save_path,f'Depth_{args.base}_model_best.pth.tar'))

#In[]: AverageMeter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def save(self):
       return self.avg

    def __str__(self):
        fmtstr = '{name} {val'+self.fmt+'} ({avg'+self.fmt+'})'
        return fmtstr.format(**self.__dict__)

#In[]: ProgressMeter
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:'+str(num_digits)+'d}'
        return '['+fmt+'/'+fmt.format(num_batches)+']'

#In[]: adjust_learning_rate
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#In[]: accuracy
def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 80000:
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2]

#In[]: clip
def clip(images, classes=[1, 7, 9, 10]):
  n = images.size(0)
  clip = torch.zeros(n, len(classes), 256,256).cuda()
  for n_img in range(n):
    for index, class_value in enumerate(classes):
      pos_i, pos_j = torch.where(images[n_img] == class_value)
      clip[n_img, index, pos_i, pos_j] = 1
  return clip

#In[]: main
if __name__ == '__main__':
    main()
    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
		"""
		Performs all_gather operation on the provided tensors.
		*** Warning ***: torch.distributed.all_gather has no gradient.
		"""
		tensors_gather = [torch.ones_like(tensor)
				for _ in range(torch.distributed.get_world_size())]
		torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

		output = torch.cat(tensors_gather, dim=0)
		return output
