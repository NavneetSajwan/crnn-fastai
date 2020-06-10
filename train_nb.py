from __future__ import print_function
from __future__ import division

from fastai import *
from fastai.vision import *

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
def data_loader(train_root, val_root):
    tfms = get_transforms(do_flip = False)
    # train
    # tfms = transforms.Compose([transforms.Resize((32,100)), transforms.ToTensor()])
    train_dataset = dataset.lmdbDataset(root = train_root, transform = tfms)
    val_dataset = dataset.lmdbDataset(root = val_root, transform = tfms)
    return train_dataset, val_dataset


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
    return torch.cat(imgs,0), (labels,torch.tensor(lengths))

def loss_func(preds, text, length):
    preds_size = Variable(torch.LongTensor([preds.size(0)] * bs))
    return criterion(preds, text, preds_size, length)

# if __name__ == "__main__":
#     train_ds, val_ds = data_loader()
#     # print(type(train_loader))
#     data = DataBunch.create(train_ds, val_ds, bs = bs, collate_fn= collation)
#     # print(type(data.show_batch()))
#     idx = np.random.randint(0,1000)
#     # print(train_ds[idx])
#     sample = data.one_batch()
#     # print(sample)
#     criterion = CTCLoss()
#     nclass = len(params.alphabet) + 1
#     crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
#     learn = Learner(data, crnn, loss_func = loss_func)
#     # print(learn)
#     learn.lr_find()
#     # print(type(data.one_batch()[0]))
#     # learn = Learner()