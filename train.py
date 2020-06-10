from __future__ import division

from fastai import *
from fastai.vision import *

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as net
import params
import cv2
bs = 12
parser = argparse.ArgumentParser()
parser.add_argument('-train', '--trainroot', required=True, help='path to train dataset')
parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
# np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
def data_loader():
    # train
    tfms = get_transforms(do_flip = False)
    # tfms = transforms.Compose([transforms.Resize((32,100)), transforms.ToTensor()])
    train_dataset = dataset.lmdbDataset(root=args.trainroot, transform = tfms)
    # assert train_dataset
    # if not params.random_sample:
    #     sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    # else:
    #     sampler = None
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
    #         shuffle=True, sampler=sampler, num_workers=int(params.workers), \
    #         collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    # # dataset.resizeNormalize((params.imgW, params.imgH))
    # # val
    val_dataset = dataset.lmdbDataset(root=args.valroot, transform= None)
    # assert val_dataset
    # val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    return train_dataset, val_dataset
    # return train_loader, val_loader


# train_loader, val_loader = data_loader()
def collation(samples:BatchSamples, pad_idx:int=0):
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    # if isinstance(samples[0][1], int): return data_collate(samples)
    max_len = max([len(s[1][0].data) for s in samples])
    # print('max_len', max_len)
    labels = torch.zeros(len(samples), max_len).long()
    imgs = []
    lengths = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        lengths.append(s[1][1].data)
        lbls = s[1][0].data
        labels[i,:len(lbls)] = tensor(lbls)
    # print(torch.tensor(lengths))
    return torch.cat(imgs,0), (labels,torch.tensor(lengths))
def loss_func(preds, text, length):
    # print(preds.shape)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * bs))
    # preds = preds.permute(2,1,0)
    # print('preds shape', preds.shape)
    # print("shape of ips/ preds", preds.shape)
    # print("shape of trgt/trgt", text.shape)
    # print("shape of ip_len/pred_size", preds_size.shape)
    # print("shape of op_len/length", length.shape)
    return criterion(preds, text, preds_size, length)

if __name__ == "__main__":
    train_ds, val_ds = data_loader()
    img, _ = train_ds[0]
    # print(type(img))
    # print(type(train_loader))
    # data = DataBunch.create(train_ds, val_ds, bs = bs, collate_fn= collation)
    # idx = np.random.randint(0,bs)
    # sample = val_ds[idx]
    # sample = data.one_batch()
    # print(sample[0].shape, sample[1][0].shape)
    # criterion = CTCLoss()
    # nclass = len(params.alphabet) + 1
    # crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    # learn = Learner(data, crnn, loss_func = loss_func)
    # print(learn)
    # learn.fit_one_cycle(5, 1e-3)
    # print(type(data.one_batch()[0]))
    # learn = Learner()