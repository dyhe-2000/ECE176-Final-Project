import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
import pickle
import numpy as np
import cv2 

train_x = torch.load('train_x.pt', map_location=torch.device('cpu'))
train_y = torch.load('train_y.pt', map_location=torch.device('cpu'))
test_x = torch.load('test_x.pt', map_location=torch.device('cpu'))
test_y = torch.load('test_y.pt', map_location=torch.device('cpu'))
val_x = torch.load('val_x.pt', map_location=torch.device('cpu'))
val_y = torch.load('val_y.pt', map_location=torch.device('cpu'))

train_x_rotate = torch.load('train_x_rotate.pt', map_location=torch.device('cpu'))
train_y_rotate = torch.load('train_y_rotate.pt', map_location=torch.device('cpu'))
test_x_rotate = torch.load('test_x_rotate.pt', map_location=torch.device('cpu'))
test_y_rotate = torch.load('test_y_rotate.pt', map_location=torch.device('cpu'))
val_x_rotate = torch.load('val_x_rotate.pt', map_location=torch.device('cpu'))
val_y_rotate = torch.load('val_y_rotate.pt', map_location=torch.device('cpu'))

train_x = train_x.numpy().astype('float32')
train_y = train_y.numpy().astype('float32')
test_x = test_x.numpy().astype('float32')
test_y = test_y.numpy().astype('float32')
val_x = val_x.numpy().astype('float32')
val_y = val_y.numpy().astype('float32')

train_x_rotate = train_x_rotate.numpy().astype('float32')
train_y_rotate = train_y_rotate.numpy().astype('float32')
test_x_rotate = test_x_rotate.numpy().astype('float32')
test_y_rotate = test_y_rotate.numpy().astype('float32')
val_x_rotate = val_x_rotate.numpy().astype('float32')
val_y_rotate = val_y_rotate.numpy().astype('float32')

train_x = np.swapaxes(train_x,3,1)
train_x = np.swapaxes(train_x,2,1)
test_x = np.swapaxes(test_x,3,1)
test_x = np.swapaxes(test_x,2,1)
val_x = np.swapaxes(val_x,3,1)
val_x = np.swapaxes(val_x,2,1)

train_x_rotate = np.swapaxes(train_x_rotate,3,1)
train_x_rotate = np.swapaxes(train_x_rotate,2,1)
test_x_rotate = np.swapaxes(test_x_rotate,3,1)
test_x_rotate = np.swapaxes(test_x_rotate,2,1)
val_x_rotate = np.swapaxes(val_x_rotate,3,1)
val_x_rotate = np.swapaxes(val_x_rotate,2,1)

print("train_x shape: ", train_x.shape)
print("train_y shape: ", train_y.shape)
print("test_x shape: ", test_x.shape)
print("test_y shape: ", test_y.shape)
print("val_x shape: ", val_x.shape)
print("val_y shape: ", val_y.shape)

print("train_x_rotate shape: ", train_x_rotate.shape)
print("train_y_rotate shape: ", train_y_rotate.shape)
print("test_x_rotate shape: ", test_x_rotate.shape)
print("test_y_rotate shape: ", test_y_rotate.shape)
print("val_x_rotate shape: ", val_x_rotate.shape)
print("val_y_rotate shape: ", val_y_rotate.shape)


# cv2.imshow(train_x[0])
a = 'y'
while a == 'y':
    index = int(input('picture index: '))
    print('label: ', train_y_rotate[index])
    cv2.imshow('picture', train_x_rotate[index])
    cv2.waitKey(0)
    a = input('continue type y: ')