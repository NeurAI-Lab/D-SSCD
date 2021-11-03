import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from transforms.simclr_transform import SimCLRTransform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.utils import save_checkpoint, log
from criterion.ntxent import NTXent, BarlowTwinsLoss, BarlowTwinsLoss_CD
from torchvision.utils import make_grid
import cv2
import torchvision.datasets as dataset
import dataset.CMU as CMU
import dataset.PCD as PCD
import util.utils as utils


def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    if args.barlow_twins :
        criteria = {'Barlow': [BarlowTwinsLoss_CD(args.device)]}   #BarlowTwinsLoss
    else:
        criteria = {'ntxent': [NTXent(args), args.criterion_weight[0]]}

    return criteria


def write_scalar(writer, total_loss, loss_p_c, leng, epoch):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss / leng, epoch)
    for k in loss_p_c:
        writer.add_scalar("{}_Loss/train".format(k), loss_p_c[k] / leng, epoch)


def trainloaderSimCLR(args):
    """
    Load training data through DataLoader
    """
    transform = SimCLRTransform(args.img_size)
    if args.ssl_dataset == 'CIFAR100':
        train_dataset = CIFAR100(args.data_dir, train=True, download=True, transform=transform)
    elif args.ssl_dataset == 'CMU':
        DATA_PATH = os.path.join(args.data_dir)

        VAL_DATA_PATH = os.path.join(args.val_data_dir)

        train_dataset = CMU.Dataset(DATA_PATH,
                                         'train', 'ssl', transform= False,     #ssl
                                         transform_med = transform)
    elif args.ssl_dataset == 'PCD':
        print('PCD dataset loaded')
        DATA_PATH = os.path.join(args.data_dir)

        VAL_DATA_PATH = os.path.join(args.val_data_dir)

        train_dataset = PCD.Dataset(DATA_PATH,
                                    'train', 'ssl', transform=False,  # ssl
                                    transform_med=transform)
    #  Data Loader
    if args.distribute:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.ssl_batchsize,sampler=train_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.ssl_batchsize, shuffle=True, drop_last=True)

    log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader

def various_distance( out_vec_t0, out_vec_t1, dist_flag='l2'):

    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
    if dist_flag == 'cos':
        similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
        distance = 1 - 2 * similarity / np.pi
    return distance

def train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch):
    """
    Train one epoch of SSL model

    """
    # torch.autograd.set_detect_anomaly(True)
    loss_per_criterion = {}
    total_loss = 0
    total_sup_loss = 0
    total_barlow_loss = 0

    for i, batch in enumerate(train_loader):
        p1, p2, n1, n2, f1,f2, label = batch    # p1, p2 = positive pair belonging to t0 images ; n1,n2 = positive pair belonging to t1 images
        p1 = p1.cuda(device=args.device)
        p2 = p2.cuda(device=args.device)
        n1 = n1.cuda(device=args.device)
        n2 = n2.cuda(device=args.device)
        label = label.cuda(device=args.device)
        label = label.float()

        optimizer.zero_grad()




        if args.dsscd_barlow_twins == True or args.sscd_barlow_twins == True :
            # fx, fy, zx, zy = model(p1, n1) # pos and neg-1st iteration
            ##### with difference layer used with double loss func
            _, _, zx, zy = model(p1, p2)
            _, _, zx1, zy1 = model(n1, n2)


            ## simple diff layer to get change map
            diff_feat0 = torch.abs(zx - zx1)
            diff_feat1 = torch.abs(zy - zy1)
            diff_feat2 = torch.abs(zx - zy1)
            diff_feat3 = torch.abs(zy - zx1)

        else:
            _, _, zx, zy = model(p1, p2)
            _, _, zx1, zy1 = model(n1, n2)

        # Multiple loss aggregation
        loss = torch.tensor(0).to(args.device)
        for k in criteria:
            global_step = epoch * len(train_loader) + i

            if args.dsscd_barlow_twins == True:
                loss = criteria[k][0](diff_feat0, diff_feat1, diff_feat2, diff_feat3 )
            elif args.sscd_barlow_twins == True:
                loss = criteria[k][0](zx, zy, zx1, zy1)
            else:
                criterion_loss = criteria[k][0](zx, zy, zx1, zy1, global_step)
                if k not in loss_per_criterion:
                    loss_per_criterion[k] = criterion_loss
                else:
                    loss_per_criterion[k] += criterion_loss
                loss = torch.add(loss, torch.mul(criterion_loss, criteria[k][1]))

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 50 == 0:

            log("Batch {}/{}. Loss: {}.  Time elapsed: {} ".format(i, len(train_loader), loss.item(),
                                                                                      datetime.now() - args.start_time))
        total_loss += loss.item()


    return total_loss, loss_per_criterion





def trainSSL(args, model, optimizer, criteria, writer, scheduler=None):
    """
    Train a SSL model
    """

    model.train()
    # Data parallel Functionality
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log('Model converted to DataParallel model with {} cuda devices'.format(torch.cuda.device_count()))
    model = model.to(args.device)

    train_loader = trainloaderSimCLR(args)

    for epoch in tqdm(range(1, args.ssl_epochs + 1)):
        total_loss, loss_per_criterion = train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch)

        write_scalar(writer, total_loss, loss_per_criterion, len(train_loader), epoch)
        log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
            format(epoch, args.ssl_epochs, total_loss / len(train_loader), datetime.now() - args.start_time))


        # Save checkpoint after every epoch
        path = save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch, filename='checkpoint.pth'.format(epoch))
        if os.path.exists:
            state_dict = torch.load(path, map_location=args.device)
            model.load_state_dict(state_dict)

        # Save the model at specific checkpoints
        if epoch % 10 == 0:

            if torch.cuda.device_count() > 1:
                save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}_model1.pth'.format(epoch))

            else:
                save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}_model1.pth'.format(epoch))

    log("Total training time {}".format(datetime.now() - args.start_time))
    writer.close()
