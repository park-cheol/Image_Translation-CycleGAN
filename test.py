import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import *
from dataset import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import matplotlib.image as img

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()

# load state dicts
G_AB.load_state_dict(torch.load("saved_models/%s/G_A2B_%d.pth" % (opt.dataset_name, opt.epoch)))
G_BA.load_state_dict(torch.load("saved_models/%s/G_B2A_%d.pth" % (opt.dataset_name, opt.epoch)))

G_AB.eval()
G_BA.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

transforms_ = [transforms.Resize((int(opt.img_height * 1.12), int(opt.img_width * 1.12)), Image.BICUBIC),
               transforms.RandomCrop((opt.img_height, opt.img_width)),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]
dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode='train'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu
)

if not os.path.exists("test_img/A"):
    os.makedirs("test_img/A")
if not os.path.exists("test_img/B"):
    os.makedirs("test_img/B")

for i, batch in enumerate(dataloader):
    real_A = Variable(batch['A'].type(Tensor))
    real_B = Variable(batch['B'].type(Tensor))

    fake_B = G_AB(real_A)
    cycle_A = G_BA(fake_B)

    fake_A = G_BA(real_B)
    cycle_B = G_AB(fake_A)

    real_A = make_grid(real_A, nrow=1, normalize=True)
    fake_B = make_grid(fake_B, nrow=1, normalize=True)
    cycle_A = make_grid(cycle_A, nrow=1, normalize=True)

    real_B = make_grid(real_B, nrow=1, normalize=True)
    fake_A = make_grid(fake_A, nrow=1, normalize=True)
    cycle_B = make_grid(cycle_B, nrow=1, normalize=True)

    image_grid1 = torch.cat((real_A, fake_B, cycle_A, real_A), 1)
    image_grid2 = torch.cat((real_B, fake_A, cycle_B, real_B), 1)

    save_image(fake_B, "test_img/A/%04d.png" % (i + 1), normalize=False)
    save_image(fake_A, "test_img/B/%04d.png" % (i + 1), normalize=False)
    print(f"[{i}/{len(dataloader)}")