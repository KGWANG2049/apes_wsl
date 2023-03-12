# CRITIC NETWORK
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.dirichlet import Dirichlet


# def LeakyReLU(x, x_max=1, hard_slope=1e-2):
# return (x <= x_max) * x + (x > x_max) * (x_max + hard_slope * (x - x_max))


class APESCriticNet(nn.Module):
    def __init__(self):
        super(APESCriticNet, self).__init__()
        self.mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.LeakyReLU = torch.nn.LeakyReLU(inplace=False)
        self.fc1 = nn.Linear(1654, 512)  # 1728改为1728+start+goal
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x, s, g, coefficients):
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = torch.flatten(x)
        s = torch.flatten(s)
        g = torch.flatten(g)
        coefficients = torch.flatten(coefficients)
        x = torch.cat((x, s, g, coefficients), 0)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)
        # x = x.cpu()
        # x = x.detach()
        # x = x.numpy().tolist()

        return x


# GENERATOR NETWORK
class APESGeneratorNet(nn.Module):
    def __init__(self):
        super(APESGeneratorNet, self).__init__()
        self.mp = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.LeakyReLU = torch.nn.LeakyReLU(inplace=False)
        self.fc1 = nn.Linear(1604, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 50)

    def forward(self, x, s, g):
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.mp(x)
        x = torch.flatten(x)
        s = torch.flatten(s)
        g = torch.flatten(g)
        x = torch.cat((x, s, g), 0)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)
        x = torch.exp(x)
        print("x", x)
        dist = Dirichlet(x)

        return dist

    def rsample(self, x, s, g):
        dist = self.forward(x, s, g)
        dirchlet_sample = dist.rsample()
        dirchlet_entropy = dist.entropy()

        return dirchlet_sample, dirchlet_entropy

    # def sample(self, x, s, g):
    # coefficients_dist = self.forward(x, s, g)
    # coefficients_rs = coefficients_dist.rsample()
    # coefficients_entropy = -coefficients_dist.prob(coefficients_rs) * coefficients_dist.log_prob(coefficients_rs)

    # return coefficients_entropy
    # def sample(self, x, s, g):
    # coefficients_dist = self.forward(x, s, g)
    # coefficients_rs = coefficients_dist.rsample()
    # coefficients_entropy = -coefficients_dist.prob(coefficients_rs) * coefficients_dist.log_prob(coefficients_rs)

    # return coefficients_entropy
