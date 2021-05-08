import argparse
import warnings
import os
import random
import numpy as np
import math
import itertools
# 효율적인 반복을 위한 함수 https://kimdoky.github.io/python/2019/11/24/python-itertools/
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import *
from dataset import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="korean_drawing", help="name of the data")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
# lr decay 시작할 epoch
parser.add_argument("-j", "--workers", type=int, default=8, help="number of cpu threads to use during batch generation")
#TODO n_cpu?
parser.add_argument("--seed", type=int, default=None, help="seed for initializing training")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
# n_residual_blocks : 6 아니면 9 사용
# lambda값 논문 그대로 적용
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

#cuda = torch.cuda.is_available()
#x.cuda() pytorch 초기버전에서 이렇게 많이쓰이고 이후 버전에서는 to(device)사용 차이는 없음
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True #TODO 원리
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if not os.path.isdir('images/%s' % args.dataset_name) and not os.path.isdir('saved_models/%s' % args.dataset_name):
        # images, saved_models 디렉터리 없을 시 생성
        os.makedirs('images/%s' % args.dataset_name, exist_ok=True)
        os.makedirs('saved_models/%s' % args.dataset_name, exist_ok=True)

    input_shape = (args.channels, args.img_height, args.img_width)

    main_worker(input_shape, args)

def main_worker(input_shape, args):
    # Loss 설정
    # L1 L2 상관없지만 L1이 좀 더 경미하게 실험적으로 좋음
    criterion_GAN = nn.MSELoss().to(device) # lsGAn
    criterion_cycle = nn.L1Loss().to(device) # L1 loss Low frequency
    criterion_idt = nn.L1Loss().to(device)

    # initial Generator and Discriminator
    # c7s1-64, d128, d256, R256(x6,x9), u128, u64, c7s1-3
    generator_A2B = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)
    generator_B2A = GeneratorResNet(input_shape, args.n_residual_blocks).to(device)

    # c64-c128-c256-c512
    discriminator_A = Discriminator(input_shape).to(device)
    discriminator_B = Discriminator(input_shape).to(device)

    if args.start_epoch != 0: # 저장해놓은 checkpoint가 있으면 실행
        generator_A2B.load_state_dict(torch.load('saved_models/%s/G_A2B_%d.pth' % (args.dataset_name, args.epoch)))
        generator_B2A.load_state_dict(torch.load('saved_models/%s/G_B2A_%d.pth' % (args.dataset_name, args.epoch)))
        discriminator_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (args.dataset_name, args.epoch)))
        discriminator_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (args.dataset_name, args.epoch)))

    else: # weight 초기화
        generator_A2B.apply(weights_init_normal)
        generator_B2A.apply(weights_init_normal)
        discriminator_A.apply(weights_init_normal)
        discriminator_B.apply(weights_init_normal)

    # optimizers
    # itertools.chain("ABC", "DEF") = A B C D E F
    generator_optimizer = torch.optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
        lr=args.lr,
        betas=(args.b1, args.b2)
    )

    discriminator_A_optimizer = torch.optim.Adam(discriminator_A.parameters(),
                                                 lr=args.lr,
                                                 betas=(args.b1, args.b2))
    discriminator_B_optimizer = torch.optim.Adam(discriminator_B.parameters(),
                                                 lr=args.lr,
                                                 betas=(args.b1, args.b2))

    # learning rate update schedulers
    lr_scheduler_generator = adjust_learning_rate(generator_optimizer, args)

    lr_scheduler_discriminaotr_A = adjust_learning_rate(discriminator_A_optimizer, args)
    lr_scheduler_discriminaotr_B = adjust_learning_rate(discriminator_B_optimizer, args)

    # initial ReplayBuffer
    fake_A_buffer = ReplayBuffer() # max_size default=50
    fake_B_buffer = ReplayBuffer()

    # image transform
    transforms_ = [
        transforms.Resize(int(args.img_height * 1.12), Image.BICUBIC),#TODO Interpolation
        transforms.RandomCrop((args.img_height, args.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Dataset loader(train)
    dataloader = torch.utils.data.DataLoader(
        ImageDataset('data/%s' % args.dataset_name, transforms_=transforms_, unaligned=True, mode='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    for epoch in range(args.start_epoch, args.epochs):


        # train
        train(epoch, dataloader, discriminator_A, discriminator_B, generator_A2B, generator_B2A,
          generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer, criterion_GAN,
          criterion_cycle, criterion_idt, lr_scheduler_generator, lr_scheduler_discriminaotr_A,
          lr_scheduler_discriminaotr_B, fake_A_buffer, fake_B_buffer, args)


def train(epoch, dataloader, discriminator_A, discriminator_B, generator_A2B, generator_B2A,
          generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer, criterion_GAN,
          criterion_cycle, criterion_idt, lr_scheduler_generator, lr_scheduler_discriminaotr_A,
          lr_scheduler_discriminaotr_B, fake_A_buffer, fake_B_buffer, args):
    start_time_epoch = time.time()

    for i, batch in enumerate(dataloader): # dataloader return {"A": item_A, "B": item_B}
        real_A = Variable(batch['A'].type(Tensor)) # [batch:1, channel:3, 256, 256]
        real_B = Variable(batch['B'].type(Tensor))

        # x.output_shape = (1, height // 16, width // 16): patch size
        # [1, 1, 16, 16]
        valid = Variable(Tensor(np.ones((real_A.size(0), *discriminator_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *discriminator_A.output_shape))), requires_grad=False)

        #####################
        #   Generator loss
        #####################
        generator_A2B.train()
        generator_B2A.train()

        generator_optimizer.zero_grad()

        # GAN MSELoss
        fake_B = generator_A2B(real_A)
        fake_A = generator_B2A(real_B)
        loss_GAN_A2B = criterion_GAN(discriminator_B(fake_B), valid) # |D(G(x))-1|^2
        loss_GAN_B2A = criterion_GAN(discriminator_A(fake_A), valid) # |D(F(y))-1|^2
        loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2

        # Cycle L1 loss
        cycle_A = generator_B2A(fake_B)
        cycle_B = generator_A2B(fake_A)
        loss_cycle_A = criterion_cycle(cycle_A, real_A) # || F(G(x))-x ||
        loss_cycle_B = criterion_cycle(cycle_B, real_B) # || G(F(y))-y ||
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Identity L1 loss
        # 필수는 아니지만 두 도메인의 특징이 비슷할 때 넣어주면 더 좋은 성능을 보임
        loss_id_A = criterion_idt(generator_B2A(real_A), real_A) # || F(x)-x ||
        loss_id_B = criterion_idt(generator_A2B(real_B), real_B) # || G(y)-y ||
        loss_id = (loss_id_A + loss_id_B) / 2

        # Generator total loss
        generator_loss = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_id
        generator_loss.backward()

        generator_optimizer.step()

        #####################
        #   Discriminator A
        #####################

        discriminator_A_optimizer.zero_grad()
        # ReplayBuffer fake image A
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)

        # Real loss
        loss_real_A = criterion_GAN(discriminator_A(real_A), valid) # |D_A(x)-1|^2
        # fake loss
        loss_fake_A = criterion_GAN(discriminator_A(fake_A_.detach()), fake) # | D_A(F(y))-0|^2

        loss_discriminator_A = (loss_real_A + loss_fake_A) / 2

        loss_discriminator_A.backward()
        discriminator_A_optimizer.step()

        #####################
        #   Discriminator B
        #####################

        discriminator_B_optimizer.zero_grad()
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)

        # Real loss
        loss_real_B = criterion_GAN(discriminator_B(real_B), valid) # |D_B(y)-1| ^2
        # fake loss
        loss_fake_B = criterion_GAN(discriminator_B(fake_B_.detach()), fake) # |D_B(G(x))-0|^2

        loss_discriminator_B = (loss_real_B + loss_fake_B) / 2

        loss_discriminator_B.backward()
        discriminator_B_optimizer.step()

        # Discriminator total loss
        discriminator_loss = (loss_discriminator_B + loss_discriminator_A) / 2


        # log
        if (i) % 100 == 0:
            pirnt_log(epoch, args.epochs, i, len(dataloader), discriminator_loss, generator_loss, loss_GAN, loss_cycle, loss_id)

            # save sample
        if i % args.sample_interval == 0:
            images = next(iter(dataloader))
            generator_A2B.eval() # dropout이나 정규화 X
            generator_B2A.eval()

            sample_real_A = Variable(images['A'].type(Tensor))
            sample_real_B = Variable(images['B'].type(Tensor))
            sample_fake_A = generator_B2A(sample_real_B)
            sample_fake_B = generator_A2B(sample_real_A)

            sample_real_A = torchvision.utils.make_grid(sample_real_A, nrow=3, normalize=True)
            sample_real_B = torchvision.utils.make_grid(sample_real_B, nrow=3, normalize=True)
            sample_fake_A = torchvision.utils.make_grid(sample_fake_A, nrow=3, normalize=True)
            sample_fake_B = torchvision.utils.make_grid(sample_fake_B, nrow=3, normalize=True)

            image_grid = torch.cat((sample_real_A, sample_fake_B, sample_real_B, sample_fake_A), 1)
            torchvision.utils.save_image(image_grid, "images/%s/%s_epoch_%s.png" % (args.dataset_name, epoch, i), normalize=False)

    # Update learning rate
    lr_scheduler_generator.step()
    lr_scheduler_discriminaotr_A.step()
    lr_scheduler_discriminaotr_B.step()

    # save model checkpoint
    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        torch.save(generator_A2B.state_dict(), "saved_models/%s/G_A2B_%d.pth" % (args.dataset_name, epoch+1))
        torch.save(generator_B2A.state_dict(), "saved_models/%s/G_B2A_%d.pth" % (args.dataset_name, epoch+1))
        torch.save(discriminator_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (args.dataset_name, epoch+1))
        torch.save(discriminator_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (args.dataset_name, epoch+1))

    # elapsed time
    sec = time.time() - start_time_epoch
    print('%d epoch time: ', datetime.timedelta(seconds=sec), '\n')





def pirnt_log(epoch, total_epoch, iter, total_iter, D_loss ,G_loss, G_adv_loss, G_cycle_loss, G_id_loss):
    print("\r[Epoch %d/%d] [Iter %d/%d] [D loss: %f] [G loss: %f -> adv: %f, cycle: %f, identity: %f]"
          % (epoch, total_epoch, iter, total_iter, D_loss, G_loss, G_adv_loss, G_cycle_loss, G_id_loss))



def adjust_learning_rate(optimizer, args):
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step
    )

    return lr_scheduler
# LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
# => Lambda 표현식으로 작성한 함수를 통해 lr 조절, 초기lr에 lambda함수에서 나온 value을 곱해서 lr계산
# lr_epoch = lr_initial * Lambda(epoch)

# MultiplicativeLR: LambdaLR은 초기 lr에만 곱해줘서 조절해주고 이 모듈은 전lr_epoch-1에 lambda를
#                   곱해주어 더욱 빠르게 떨어짐

# stepLR: gamma와 step_size 인자를 바탕으로 step_size마다 gamma를 곱해줌
#         (if epoch % step_size = 0) Gamma * lr_epoch -1

# MultistepLR: step_size가 아니라 milestones(epoch)를 따로 지정하는 것만 빼고 stepLR와 동일


if __name__ == '__main__':
    main()






