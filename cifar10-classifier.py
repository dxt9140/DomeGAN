"""
CIFAR-10 Classifier
Dominick Taylor
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.optim.sgd as sgd
import torch.nn as nn


def weight_init(m):
    if type(m) in [torch.nn.ConvTranspose2d, torch.nn.Conv2d]:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)


class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.num_classes = num_classes

        activation = nn.LeakyReLU(0.2, inplace=True)
        activation2 = nn.Tanh()
        bias = False

        self.main = torch.nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(128),
            activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(256),
            activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(512),
            activation,
            # torch.nn.Flatten(1, -1),
            # torch.nn.Conv2d(128, num_classes, kernel_size=3, stride=2, padding=0, bias=bias),
            # nn.MaxPool2d(kernel_size=3),
            nn.Flatten(1, -1),
            # nn.Linear(256, num_classes, bias=bias),
            nn.Linear(512*3*3, num_classes, bias=bias),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)

    # def fit(self):


def main():
    epochs = 5
    batch_size = 1000

    training_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                 transform=transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                             transform=transforms.ToTensor())
    train_loader = dataloader.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    test_loader = dataloader.DataLoader(test_data, shuffle=True, batch_size=len(test_data))

    num_classes = len(training_data.classes)

    model = CNN(num_classes)
    model.apply(weight_init)
    # print(model)

    reuse = False
    if not reuse:

        optimizer = sgd.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        loss_function = nn.NLLLoss()

        for e in range(epochs):

            for i, data in enumerate(train_loader):

                image, label = data
                # print("Image shape: ", image.shape)
                # print("Labels shape: ", label.shape)

                model.zero_grad()

                prediction = model(image)
                # print("Prediction shape: ", prediction.shape)

                loss = loss_function(prediction, label)
                loss.backward()

                optimizer.step()

                print("[%d/%d][%d/%d] %.4f" % (e+1, epochs, i+1, len(train_loader), loss.mean().item()))

        torch.save(model.state_dict(), "CNN.model")

    else:
        model_dict = torch.load("CNN.model")
        model = CNN(num_classes)
        model.load_state_dict(model_dict)

    for b in test_loader:

        image, label = b

        with torch.no_grad():
            output = model(image)
            predicted = torch.max(output, 1)

            correct = (predicted.indices == label).nonzero().squeeze()
            # print(correct.shape)

            print("%d/%d = %.2f%%" % (len(correct), len(test_data), len(correct) / len(test_data) * 100))


if __name__ == '__main__':
    main()
