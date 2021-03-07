import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T


import numpy as np

class BaselineResNet(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.resizing = nn.UpsamplingBilinear2d(size=(254, 254))
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        self.feature_extractor.fc =nn.Linear(self.feature_extractor.fc.in_features, num_output)

    def forward(self, x):
        x = self.resizing(x)
        x = self.feature_extractor(x)
        return x

class ResNetMainBranch(nn.Module):
    def __init__(self):
        self.resnet = BaselineResNet(1000)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1000, 56)

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class ResNetTwoBranch(nn.Module):
    def __init__(self):
        self.resnet = BaselineResNet(1000)
        self.relu = nn.ReLU()
        self.fc_main = nn.Linear(1000, 56)
        self.fc_auxillary = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        main = self.fc_main(x)
        auxillary = self.fc_auxillary(x)
        return main, auxillary


    

