"""
gan.py
Dominick Taylor
Fall 2019
Define the architecture for a vanilla GAN, designed for CIFAR10. The tutorial DCGAN on the Pytorch website
was used as a ference for much of this code.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class GAN:

    def __init__(self, nz=100, epochs=5, batch_size=100, dimensions=64):
        # Size of the noise vector
        self.nz = nz

        # Number of output channels
        self.channels = 3

        # Number of feature maps
        self.features = 8

        # 64 for CIFAR10. 32 x 32 x 3. Channels handled by convolution.
        self.dimensions = dimensions

        self.epochs = epochs
        self.batch_size = batch_size

        self.G = self.Generator(self.nz, self.features, self.channels)

        self.D = self.Discriminator(self.channels, self.features)

        self.G.apply(self.weight_init)
        self.D.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        # An ultimately useless attempt at fixing a problem. Leaving it for now
        if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif type(m) is nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, 1.0, 0.2)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):

        def __init__(self, nz, features, channels):
            super(GAN.Generator, self).__init__()
            # super().__init__()

            bias = False

            # act = nn.LeakyReLU(0.2, inplace=True)
            act = nn.ReLU(True)

            # W - K + 2P
            # ---------- + 1
            #     S

            self.main1 = nn.Sequential(
                nn.ConvTranspose2d(nz, 32, kernel_size=13, stride=5, padding=0, bias=bias),
                nn.BatchNorm2d(32),
                act,
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=3, padding=4, bias=bias),
                nn.Tanh()
            )

            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, kernel_size=5, stride=3, padding=0, bias=bias),
                nn.BatchNorm2d(512),
                # activation(),
                act,
                nn.ConvTranspose2d(512, 256, kernel_size=5, stride=3, padding=1, bias=bias),
                nn.BatchNorm2d(256),
                # activation(),
                act,
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=bias),
                nn.BatchNorm2d(128),
                # activation(),
                act,
                nn.ConvTranspose2d(128, channels, kernel_size=3, stride=1, padding=0, bias=bias),
                nn.Tanh(),
            )

            self.main2 = nn.Sequential(
                nn.Linear(nz, 3*64**2, bias=bias),
                nn.BatchNorm1d(3*64**2),
                act,
                nn.Linear(3*64**2, 3*48**2, bias=bias),
                nn.BatchNorm1d(3*48**2),
                act,
                nn.Linear(3*48**2, 3*36**2, bias=bias),
                nn.BatchNorm1d(3*36**2),
                act,
                nn.Linear(3*36**2, 3*32**2, bias=bias),
                nn.Sigmoid(),
            )

        def forward(self, z):
            return self.main(z)
            # return self.main(z).view(-1, 3, 32, 32)

    class Discriminator(nn.Module):

        def __init__(self, channels, features):
            super(GAN.Discriminator, self).__init__()
            # super().__init__()

            activation = nn.ReLU
            bias = False
            act = nn.LeakyReLU(0.2, inplace=True)

            self.main = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=5, padding=0, stride=2, bias=bias),
                nn.BatchNorm2d(4),
                act,
                nn.Conv2d(4, 8, kernel_size=5, padding=0, stride=2, bias=bias),
                nn.BatchNorm2d(8),
                act,
                # nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=2, bias=bias),
                # nn.BatchNorm2d(128),
                # act,
                nn.Conv2d(8, 1, kernel_size=5, padding=0, stride=1, bias=bias),
                nn.Sigmoid()
            )

        def forward(self, x):
            # .view reshapes the 4 dimensional output to 2 dimensional
            return self.main(x)

