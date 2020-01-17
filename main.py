"""
main.py
Dominick Taylor
Fall 2019
Main program file for training a series of models on the CIFAR10 dataset.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.sgd as sgd
import PIL.Image
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torchvision.transforms import ToPILImage
import numpy as np
from gan import GAN
import cv2


def load_model(argl):
    if argl.model == 'gan':
        return GAN(epochs=argl.epochs, batch_size=argl.batch_size, nz=argl.noise)
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="batch_size", type=int, help="Specify the batch size.",
                        default=100)
    parser.add_argument("-e", dest="epochs", type=int, help="Specify the number of epochs to run.",
                        default=1)
    parser.add_argument("-m", dest="model", type=str, help="Specify the model to use for generation.",
                        choices=['gan'], default='gan')
    parser.add_argument("-n", dest="noise", type=int, help="Specify the size of the latent noise vector to use.",
                        default=100)
    parser.add_argument("-r", "--reuse-model",  dest="reuse_model", action='store_true', default=False,
                        help="Specify if you'd like to load the previously saved model.")
    parser.add_argument("--num-gpus", dest="gpus", type=int, default=0, help="Specify the number of GPUs available.")
    argl = parser.parse_args()

    # Select the device to use. Assuming we use CPU unless specified otherwise.
    device = torch.device("cuda:0" if argl.gpus > 0 else "cpu")

    # Set up datasets for loading
    training_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                 transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                             transform=transforms.ToTensor())

    train_loader = dataloader.DataLoader(training_data, shuffle=True, batch_size=argl.batch_size)

    # This is used below. We cannot pre-form valid and fake tensors like in tensorflow. We must fill a label
    # tensor later with the appropriate values in between each step.
    real = 1
    fake = 0

    loss_func = nn.BCELoss()

    # Load the model. This exists to provide extensibility in case I want to add further models.
    model = load_model(argl)
    if model is None:
        raise RuntimeError("Failed to load model %s" % argl.model)

    # print("Generator state_dict:")
    # for param_tensor in model.G.state_dict():
    #    print(param_tensor, "\t", model.G.state_dict()[param_tensor].size())

    # modules = model.G.modules()
    # print(modules)
    # for mod in modules:
    #    # print(mod)
    #    pass

    if argl.reuse_model and os.path.exists("GAN.model"):
        model_dict = torch.load('GAN.model')

        D_dict = model_dict["Discriminator"]
        G_dict = model_dict["Generator"]

        # print(D_dict)
        # print("Model's state_dict:")
        # for param_tensor in model.D.state_dict():
        #    print(param_tensor, "\t", model.D.state_dict()[param_tensor].size())

        model.D.load_state_dict(D_dict)
        model.G.load_state_dict(G_dict)

    else:
        D_optimizer = sgd.SGD(model.D.parameters(), lr=1e-2)
        G_optimizer = sgd.SGD(model.G.parameters(), lr=1e-2)

        for e in range(argl.epochs):

            for i, data in enumerate(train_loader):

                # Zero the gradients
                model.D.zero_grad()

                images, classes = data
                # print(images.shape)

                # Train the Discriminator

                labels = torch.full((argl.batch_size,), real)
                # print(labels.shape)

                # Feed real data into the discriminator and compute gradients
                # d_forward_real = model.D.forward(images).view(-1)
                d_forward_real = model.D(images).view(-1)
                # d_forward_real = model.D(images.to(device))
                # print("D_forward_real shape: ", str(d_forward_real.shape))
                d_error_real = loss_func(d_forward_real, labels)
                d_error_real.backward()
                print("D_x:\t\t", torch.mean(d_forward_real).item())

                labels.fill_(fake)

                # Generate some random noise
                noise_batch = torch.rand(argl.batch_size, 1, argl.noise, argl.noise)
                # Shape is batch_size x nz  x 1 x 1
                # print("Noise shape: " + str(noise_batch.shape))

                # Feed fake data into Generator to output an image
                # Gz = model.G.forward(noise_batch)
                Gz = model.G(noise_batch)
                print("Generated shape: ", str(Gz.shape))

                # blargh = Gz.detach().numpy()[0].transpose(1, 2, 0)
                # blargh *= 255
                # blargh = blargh.astype(np.uint8)
                # blargh = cv2.resize(blargh, (300, 300))
                # cv2.imshow("bana", blargh)
                # cv2.waitKey()

                # d_forward_fake = model.D.forward(Gz).view(-1)
                d_forward_fake = model.D(Gz.detach()).view(-1)
                # print("D_forward_fake shape: ", str(d_forward_fake.shape))
                d_error_fake = loss_func(d_forward_fake, labels)
                d_error_fake.backward()
                print("D(G(z)):\t", torch.mean(d_forward_fake).item())
                # print(noise_batch)

                total_real = d_error_real + d_error_fake

                D_optimizer.step()
                model.G.zero_grad()

                # Train the Generator
                labels.fill_(real)

                # g_forward_fake = model.D.forward(Gz.detach()).view(-1)
                g_forward_fake = model.D(Gz).view(-1)
                # g_forward_fake = model.D(Gz.to(device))
                g_error_real = loss_func(g_forward_fake, labels)
                g_error_real.backward()

                # labels.fill_(fake)

                # g_forward_real = model.D.forward(images).view(-1)
                # g_error_fake = loss_func(g_forward_real, labels)
                # g_error_fake.backward()

                # g_error_total = g_error_real + g_error_fake

                G_optimizer.step()

                print("[%d] %d/%d: D_loss_real [%.4f] D_loss_fake [%.4f] G_loss [%.4f]"
                      % (e+1, i+1, len(train_loader), d_error_real, d_error_fake, g_error_real))

        torch.save({
            "Discriminator": model.D.state_dict(),
            "Generator": model.G.state_dict()
        }, "GAN.model")

    while 1:
        # Generate some random noise
        noise_batch = torch.rand(1, argl.noise, 1, 1)
        generated = model.G.forward(noise_batch).detach()

        D_gen = model.D(generated)
        print("D output of generated image:\t", D_gen.mean().item())

        # img = torchvision.utils.make_grid(generated, normalize=True)
        # plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.show()

        # im = ToPILImage()(generated)
        im = generated.numpy()[0].transpose(1, 2, 0)
        im *= 255
        im = im.astype(np.uint8)

        win = cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 300, 300)

        cv2.imshow("Image", im)
        if cv2.waitKey(0) == ord('q'):
            break
