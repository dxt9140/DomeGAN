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

        # self.G.apply(self.weight_init)
        # self.D.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        # An ultimately useless attempt at fixing a problem. Leaving it for now
        if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    class Generator(nn.Module):

        def __init__(self, nz, features, channels):
            super(nn.Module).__init__()
            super().__init__()

            activation = nn.Sigmoid

            self.main = nn.Sequential(
                # nz x 1 x 1
                nn.ConvTranspose2d(1, 96, kernel_size=3, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(features * 32),
                activation(),
                # (features * 5) x 3 x 3
                nn.Conv2d(96, 64, kernel_size=5, stride=2, padding=0, bias=False),
                # nn.BatchNorm2d(features * 12),
                activation(),
                # (features * 10) x 5 x 5
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=2, bias=False),
                # nn.BatchNorm2d(features * 4),
                activation(),
                # (features * 20) x 7 x 7)
                nn.Conv2d(32, channels, kernel_size=5, stride=1, padding=1, bias=False),
                nn.Sigmoid(),
                # (features * 32) x 32 x 32
                # nn.ConvTranspose2d(features * 32, channels, kernel_size=4, stride=2, padding=1, bias=False),
                # nn.Sigmoid()
                # 3 x 32 x 32
            )

        def forward(self, z):
            return self.main(z)

    class Discriminator(nn.Module):

        def __init__(self, channels, features):
            super(nn.Module, self).__init__()
            super().__init__()

            activation = nn.Sigmoid

            self.main = nn.Sequential(
                # 3 x 32 x 32
                nn.Conv2d(channels, 24, kernel_size=3, padding=0, stride=2),
                # nn.BatchNorm2d(features * 4),
                activation(),
                # (dimensions * 3) x 16 x 16
                nn.Conv2d(24, 48, kernel_size=3, padding=0, stride=2),
                # nn.BatchNorm2d(features * 12),
                activation(),
                # (dimensions * 6) x 8 x 8)
                nn.Conv2d(48, 64, kernel_size=3, padding=0, stride=2),
                # nn.BatchNorm2d(features * 32),
                activation(),
                # (dimensions * 16) x 2 x 2
                nn.Conv2d(64, 1, kernel_size=3, padding=0, stride=2),
                nn.Sigmoid()
                # 1 x 1 x 1
            )

        def forward(self, x):
            # .view reshapes the 4 dimensional output to 2 dimensional
            return self.main(x)

