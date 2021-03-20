import torch
import torch.nn as nn
from torch.nn.modules.normalization import GroupNorm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.models.resnet import resnet18

import numpy as np

def init_layers(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        # nn.init.zeros_(layer.bias)

class BaselineResNet(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        # self.resizing = nn.UpsamplingBilinear2d(size=(224, 224))
        # self.feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        self.feature_extractor = resnet18(norm_layer=Norm_Layer)
        self.feature_extractor.fc =nn.Linear(self.feature_extractor.fc.in_features, num_output)
        self.apply(init_layers)

    def forward(self, x):
        # x = self.resizing(x)
        x = self.feature_extractor(x)
        return x

class ResNetMainBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = BaselineResNet(1000)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1000, 43)
        self.apply(init_layers)


    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class ResNetTwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = BaselineResNet(1000)
        self.relu = nn.ReLU()
        self.fc_main = nn.Linear(1000, 43)
        self.fc_auxillary = nn.Linear(1000, 4)
        self.apply(init_layers)


    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        main = self.fc_main(x)
        auxillary = self.fc_auxillary(x)
        return main, auxillary

class Norm_Layer(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.group_norm = GroupNorm(num_channel // 2, num_channel)

    def forward(self, x):
        return self.group_norm(x)


    

