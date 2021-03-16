import torch
import torch.nn as nn
from torch.nn.modules.normalization import GroupNorm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch.nn.functional as F  # useful stateless functions

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class BaselineCNN(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)
        
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.zeros_(self.conv6.bias)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(18432, 9216)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(9216, num_output)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = flatten(x)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        
class CNNMainBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = BaselineCNN(1000)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1000, 43)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
        
class CNNTwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = BaselineCNN(1000)
        self.relu = nn.ReLU()
        self.fc_main = nn.Linear(1000, 43)
        self.fc_auxillary = nn.Linear(1000, 4)



    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        main = self.fc_main(x)
        auxillary = self.fc_auxillary(x)
        return main, auxillary