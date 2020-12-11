import torch.nn as nn
import torch


# weights initialization
def weight_init(m:nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    input: [bs, 100, 1, 1]
    output: [bs, nc, 64, 64]
    """
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state shape: [bs, ngf*8, 4, 4]

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state shape: [bs, ngf*4, 8, 8]

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state shape: [bs, ngf*2, 16, 16]

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state shape: [bs, ngf, 32, 32]

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state shape: [bs, nc, 64, 64]
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input shape: [bs, nc, 64, 64]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf, 32, 32]

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*2, 16, 16]

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*4, 8, 8]

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state shape: [bs, ndf*8, 4, 4]

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state shape: [bs, 1, 1, 1]
        )

    def forward(self, x):
        return self.net(x)


def get_GAN(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    gpus = torch.cuda.device_count()
    gpu_list = list(range(gpus))

    G = torch.nn.DataParallel(Generator(nz=args.nz, ngf=args.ngf, nc=args.nc), device_ids=gpu_list)
    D = torch.nn.DataParallel(Discriminator(ndf=args.ndf, nc=args.nc), device_ids=gpu_list)
    G.apply(weight_init)
    D.apply(weight_init)

    return G, D


if __name__ == '__main__':
    device = 'cuda:0'
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    G = Generator(nz=100, ngf=64, nc=3).to(device)
    D = Discriminator(ndf=64, nc=3).to(device)
    G.apply(weight_init)
    D.apply(weight_init)
    # print(G)
    g_out = G(fixed_noise)
    d_out = D(g_out)
    print("Generator output shape: ", g_out.shape)
    print("Discriminator output shape: ", d_out.shape)

