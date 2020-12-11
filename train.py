from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from models.model import get_GAN


# 为再现性设置随机seem
manualSeed = 999
# manualSeed = random.randint(1, 10000) 
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def save_example_imgs(dataloader, device):
    # 绘制部分我们的输入图像
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('example_imgs.jpg')


def train(args):
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    #
    save_example_imgs(dataloader, device)

    # set models
    G, D = get_GAN(args)

    # criterion
    criterion = nn.BCELoss()

    # set real label and fake label
    real_label = 1
    fake_label = 0

    # set optimizer
    optimizerD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 创建一批潜在的向量，我们将用它来可视化生成器的进程
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):

            ########################################################
            # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ########################################################
            # 1.1 Train with all-real batch
            D.zero_grad()  # 等价于optimizerD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = D(real_data).view(-1)
            loss_D_real = criterion(output, label)
            loss_D_real.backward()
            D_x = output.mean().item()
            # 1.2 Train with all-fake batch
            noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)
            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            loss_D = loss_D_real + loss_D_fake
            # update D
            optimizerD.step()

            ########################################################
            # 2. Update G network: maximize log(D(G(z)))
            ########################################################
            optimizerG.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # display info
            if (i+1) % 50 == 0:
                print("epoch[{}/{}]\t step[{}]\tLoss_D:{}\tLoss_G:{}\tD(x):{}\tD(G(z)):{}/{}\t".format(
                    epoch, args.num_epochs, i+1, loss_D, loss_G, D_x, D_G_z1, D_G_z2
                ))

            # Save Losses for plotting later
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                g_out = vutils.make_grid(fake, padding=2, normalize=True)
                vutils.save_image(g_out, "G_out/epoch{}_iter{}.jpg".format(epoch, iters))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('loss_curve.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN train')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--epochs_D', default=4, type=int, help='epochs of Discriminator')
    parser.add_argument('--epochs_G', default=3, type=int, help='epochs of Generator')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of train')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--dataroot', default="D:/dataset/celeba", type=str, help='data set path')
    parser.add_argument('--image_size', default=64, type=int, help='scale of input')
    parser.add_argument('--nc', default=3, type=int, help='channel of input')
    parser.add_argument('--nz', default=100, type=int, help='scale of latent vector')
    parser.add_argument('--ngf', default=64, type=int, help='The size of the feature map in the generator')
    parser.add_argument('--ndf', default=64, type=int, help='The size of the feature map in the discriminator')
    parser.add_argument('--vis_path', default="Generate_results", type=str, help='generate data save path')
    args = parser.parse_args()
    train(args)
