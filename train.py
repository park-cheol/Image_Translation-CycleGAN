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
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

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
# distributed training
parser.add_argument("--world-size", default=-1, type=int,
                    help='number of nodes for distributed training ')
parser.add_argument("--rank", default=-1, type=int,
                    help='node rank for distributed training ') # node의 아이디
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training ')

# Gloo backend 형식 device('cuda')
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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1: #TODO env:// 와 tcp
        args.world_size = int(os.environ["WORLD_SIZE"])
        # 전체 프로세스 수 - 마스터가 얼마나 많은 worker들을 기다릴지 알 수 있다.

    # node(server, 본체) 2개 이상 이거나 multiprocessing_distributed가 TURE
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() # node: server(기계)라고 생각
    print("ngpus 개수 확인", ngpus_per_node)
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("wolrd_size확인: ", args.world_size)
        # world_size: 총 processes 크기 (usually 1gpu = 1process)
        # world_size => 서버(기계)1개당 gpu 수 * 기계 수
        # args.world_size help='총 노드 수'
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # torch.multiprocessing.spawn
        # (fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
        # Spawns nprocs processes that run fn with args: n개의 processes로 fn 실행
        # args: fn에 넘겨줄 인자들
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    input_shape = (args.channels, args.img_height, args.img_width)
    # PatchGan 70 x 70
    output_shape = (1, args.img_height // 2 ** 4, args.img_width // 2 ** 4)

    args.gpu = gpu

    print("distributed 확인: ", args.distributed)
    # initial Generator and Discriminator
    # c7s1-64, d128, d256, R256(x6,x9), u128, u64, c7s1-3
    generator_A2B = GeneratorResNet(input_shape, args.n_residual_blocks)
    generator_B2A = GeneratorResNet(input_shape, args.n_residual_blocks)

    # c64-c128-c256-c512
    discriminator_A = Discriminator(input_shape)
    discriminator_B = Discriminator(input_shape)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed: # 초기화
        if args.dist_url == "env://" and args.rank == -1: # RANK 지정 X 경우
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # 멀티 프로세싱을 하기 위해서 rank는 모든 프로세스 중의 global rank를 필요함
            args.rank = args.rank * ngpus_per_node + gpu
            print("rank 확인: ", args.rank)
            # 순서대로 rank를 지정해줌 e.g)node 2 with 4gpu
            # node=0 -> rank[0, 1, 2, 3] node=1 -> rank[4, 5, 6, 7]
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        # torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name=''
        # distributed process group을 초기화, distributed package또한


    if not torch.cuda.is_available(): # GPU가 없을 시
        print('using CPU, this will be slow')
    elif args.distributed:
        # 멀티프로세싱 distributed 경우, 반드시 단일 장치범위(single device scope) 지정해야함
        # 그렇게 안하면 사용 가능한 모든 device를 사용
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu) # 특정 gpu 설정
            generator_A2B.cuda(args.gpu)
            generator_B2A.cuda(args.gpu)
            discriminator_A.cuda(args.gpu)
            discriminator_B.cuda(args.gpu)

            # per process와 DistributedDataparallel=> single GPU 사용 할 때
            # batchsize를 가지고 있는 총 gpu을 기점으로 나눠 줄 필요가 있다.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # workers: batch generation 때 사용 할 cpu thread 수
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            generator_A2B = nn.parallel.DistributedDataParallel(generator_A2B, device_ids=[args.gpu])
            generator_B2A = nn.parallel.DistributedDataParallel(generator_B2A, device_ids=[args.gpu])
            discriminator_A = nn.parallel.DistributedDataParallel(discriminator_A, device_ids=[args.gpu])
            discriminator_B = nn.parallel.DistributedDataParallel(discriminator_B, device_ids=[args.gpu])

        else:
            # device_ids를 설정해주지 않으면 모든 사용가능한 GPU로 batchsize 나누고 할당
            generator_A2B.cuda()
            generator_B2A.cuda()
            discriminator_A.cuda()
            discriminator_B.cuda()

            generator_A2B = nn.parallel.DistributedDataParallel(generator_A2B)
            generator_B2A = nn.parallel.DistributedDataParallel(generator_B2A)
            discriminator_A = nn.parallel.DistributedDataParallel(discriminator_A)
            discriminator_B = nn.parallel.DistributedDataParallel(discriminator_B)

    elif args.gpu is not None:
        # distributed X, Dataparallel X
        torch.cuda.set_device(args.gpu)
        generator_A2B = generator_A2B.cuda(args.gpu)
        generator_B2A = generator_B2A.cuda(args.gpu)
        discriminator_A = discriminator_A.cuda(args.gpu)
        discriminator_B = discriminator_B.cuda(args.gpu)

    else:
        # DataParallel은 사용가능한 gpu에다가 batchsize을 나누고 할당
        generator_A2B = nn.DataParallel(generator_A2B).cuda(args.gpu)
        generator_B2A = nn.DataParallel(generator_B2A).cuda(args.gpu)
        discriminator_A = nn.DataParallel(discriminator_A).cuda(args.gpu)
        discriminator_B = nn.DataParallel(discriminator_B).cuda(args.gpu)

    # Loss 설정
    # L1 L2 상관없지만 L1이 좀 더 경미하게 실험적으로 좋음
    criterion_GAN = nn.MSELoss().cuda(args.gpu) # lsGAn
    criterion_cycle = nn.L1Loss().cuda(args.gpu) # L1 loss Low frequency
    criterion_idt = nn.L1Loss().cuda(args.gpu)


    if args.start_epoch != 0: # 저장해놓은 checkpoint가 있으면 실행
        generator_A2B.load_state_dict(torch.load('saved_models/%s/G_A2B_%d.pth' % (args.dataset_name, args.start_epoch)))
        generator_B2A.load_state_dict(torch.load('saved_models/%s/G_B2A_%d.pth' % (args.dataset_name, args.start_epoch)))
        discriminator_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (args.dataset_name, args.start_epoch)))
        discriminator_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (args.dataset_name, args.start_epoch)))

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
        transforms.Resize((int(args.img_height * 1.12),int(args.img_width * 1.12)), Image.BICUBIC),#TODO Interpolation
        transforms.RandomCrop((args.img_height, args.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    print(args.distributed)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            ImageDataset('data/%s' % args.dataset_name, transforms_=transforms_, unaligned=True, mode='train')
        )

    else:
        train_sampler = None

    # Dataset loader(train)
    # Dataset에는 index로 data를 return (__len__, __getitem__)
    # sampler : __len__, __iter__로 호출시 가져올 data idx 반환
    # 이것을 dataloader가 인자로 받는 형식
    dataloader = torch.utils.data.DataLoader(
        ImageDataset('data/%s' % args.dataset_name, transforms_=transforms_, unaligned=True, mode='train'),
        batch_size=args.batch_size,
        shuffle=(train_sampler is None), # 어차피 sampler로 섞기 때문에 shuffled할 필요가없는듯
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # num_worker = 4 * num_GPU  실험적으로 좋다 GPU memory 상관 X
    # pin_memory 언제 쓰는게 좋은가 https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    # GPU와 통신하기 위해 CPU의 메모리공간이 강제로 할당되어서 데이터 통신 속도 향상
    # 이 때 pinned memory와 pageable memory가 있는데 pinned memory가 제일 좋더라
    # https://cvml.tistory.com/24 (단 시스템 메모리가 넉넉해야함)
    # pinned memory에 고정시켜 전송하는 방법

    # sampler: index를 컨트롤 하는 방법
    # https://subinium.github.io/pytorch-dataloader/
    # 데이터의 index를 원하는 방식대로 조정 => Shuffle=False여야합니다.
    # 다양한 option들이 존재
    # https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        train(epoch, dataloader, discriminator_A, discriminator_B, generator_A2B, generator_B2A,
              generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer, criterion_GAN,
              criterion_cycle, criterion_idt, lr_scheduler_generator, lr_scheduler_discriminaotr_A,
              lr_scheduler_discriminaotr_B, fake_A_buffer, fake_B_buffer, output_shape, args)


def train(epoch, dataloader, discriminator_A, discriminator_B, generator_A2B, generator_B2A,
          generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer, criterion_GAN,
          criterion_cycle, criterion_idt, lr_scheduler_generator, lr_scheduler_discriminaotr_A,
          lr_scheduler_discriminaotr_B, fake_A_buffer, fake_B_buffer, output_shape, args):
    start_time_epoch = time.time()

    for i, batch in enumerate(dataloader): # dataloader return {"A": item_A, "B": item_B}
        real_A = Variable(batch['A']).cuda(args.gpu, non_blocking=True) # [batch:1, channel:3, 256, 256]
        real_B = Variable(batch['B']).cuda(args.gpu, non_blocking=True)
        # pin_memory=False, non_blocking=True: 안좋을수 있다. https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/13
        # Host에서 GPU로 복사할 때 고정된(pinned) 메모리를 사용하면더 빠름
        # pin_memory를 통해 고정된 영역의 데이터 복사본을 얻고 일단 고정시키면 non_blokcing=True
        # 넣어줌으로써 비동기적으로 GPU 복사본을 사용 가능
        # Dataloader 생성자에 pn_memory=True를 넣어주어서 고정된 메모리에서 배치 생산
        # https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader


        # x.output_shape = (1, height // 16, width // 16): patch size
        # [1, 1, 16, 16]
        valid = Variable(Tensor(np.ones((real_A.size(0), *output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *output_shape))), requires_grad=False)

        #####################
        #   Generator loss
        #####################
        generator_A2B.train()
        generator_B2A.train()

        generator_optimizer.zero_grad()

        # Identity L1 loss
        # 필수는 아니지만 두 도메인의 특징이 비슷할 때 넣어주면 더 좋은 성능을 보임
        loss_id_A = criterion_idt(generator_B2A(real_A), real_A) # || F(x)-x ||
        loss_id_B = criterion_idt(generator_A2B(real_B), real_B) # || G(y)-y ||
        loss_id = (loss_id_A + loss_id_B) / 2

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
        # save sample
        if i % len(dataloader)-1 == 0:
            sample_images(epoch, real_A, real_B, generator_A2B, generator_B2A)

        if i % len(dataloader)-1 ==0:
            pirnt_log(epoch, args.epochs, i, len(dataloader), discriminator_loss, generator_loss, loss_GAN, loss_cycle, loss_id)


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
    print('%d epoch time: '.format(epoch), datetime.timedelta(seconds=sec), '\n')


def sample_images(epoch, real_A, real_B, generator_A2B, generator_B2A):
    generator_A2B.eval()
    generator_B2A.eval()

    fake_B = generator_A2B(real_A)
    fake_A = generator_B2A(real_B)

    cycle_A = generator_B2A(fake_B)
    cycle_B = generator_A2B(fake_A)

    real_A = torchvision.utils.make_grid(real_A, nrow=4, normalize=True)
    fake_B = torchvision.utils.make_grid(fake_B, nrow=4, normalize=True)
    cycle_A = torchvision.utils.make_grid(cycle_A, nrow=4, normalize=True)

    real_B = torchvision.utils.make_grid(real_B, nrow=4, normalize=True)
    fake_A = torchvision.utils.make_grid(fake_A, nrow=4, normalize=True)
    cycle_B = torchvision.utils.make_grid(cycle_B, nrow=4, normalize=True)

    print("A",real_A.size())
    image_grid1 = torch.cat((real_A, fake_B, cycle_A, real_A), 1)
    image_grid2 = torch.cat((real_B, fake_A, cycle_B, real_B), 1)
    print(image_grid1.size())

    torchvision.utils.save_image(image_grid1, "output/A/%03d.png" % (epoch), normalize=False)
    torchvision.utils.save_image(image_grid2, "output/B/%03d.png" % (epoch), normalize=False)
    #04d: 4자리 숫자를 표현하는데 4자리가 안되면 0으로 채워라


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



