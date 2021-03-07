import cv2 
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import skimage
from skimage import transform
import PIL

train_path = 'Dataset/train/'
test_path = 'Dataset/test/'
train_label_path = 'Dataset/train_label/TsignRecgTrain4170Annotation.txt'
test_label_path = 'Dataset/test_label/TsignRecgTest1994Annotation.txt'

file1 = open(train_label_path, 'r')
Lines1 = file1.readlines()
file1.close()

file1 = open(test_label_path, 'r')
Lines2 = file1.readlines()
file1.close()

min_dim1 = int(Lines1[0].split(';')[2])
min_dim2 = int(Lines1[0].split(';')[1])
sum_dim1 = 0
sum_dim2 = 0
train_count = 0
test_count = 0

for i in range(len(Lines1)):
    sum_dim1 += int(Lines1[i].split(';')[2])
    sum_dim2 += int(Lines1[i].split(';')[1])
    if int(Lines1[i].split(';')[2]) < min_dim1:
        min_dim1 = int(Lines1[i].split(';')[2])
    if int(Lines1[i].split(';')[1]) < min_dim2:
        min_dim2 = int(Lines1[i].split(';')[1])
    # if int(Lines1[i].split(';')[2]) == 28:
        # print(Lines1[i].split(';')[0])
        # print(1, i)
    # if int(Lines1[i].split(';')[1]) == 26:
        # print(Lines1[i].split(';')[0])
        # print(1, i)
    train_count += 1
        
for i in range(len(Lines2)):
    sum_dim1 += int(Lines2[i].split(';')[2])
    sum_dim2 += int(Lines2[i].split(';')[1])
    if int(Lines2[i].split(';')[2]) < min_dim1:
        min_dim1 = int(Lines2[i].split(';')[2])
    if int(Lines2[i].split(';')[1]) < min_dim2:
        min_dim2 = int(Lines2[i].split(';')[1])
    # if int(Lines2[i].split(';')[2]) == 28:
        # print(Lines2[i].split(';')[0])
        # print(2, i)
    # if int(Lines2[i].split(';')[1]) == 26:
        # print(Lines2[i].split(';')[0])
        # print(2, i)
    test_count += 1

# print(len(Lines1))
# print(len(Lines2))
print('train set total count: ', train_count)
print('test set total count: ', test_count)
print('average dimension: ')
print(sum_dim1/(train_count + test_count), sum_dim2/(train_count + test_count))
print()

train_x = np.zeros((train_count * 2, 128, 128, 3))
train_y = np.zeros((train_count * 2,))

'''
for i in range(len(Lines1)):
    image = cv2.imread(train_path + Lines1[i].split(';')[0])
    image = cv2.resize(image, (128,128), 1, 1)
    cv2.imshow('hello', image)
    cv2.waitKey(0)
'''

transformer = torchvision.transforms.RandomCrop((128,128), padding=False, pad_if_needed=True, padding_mode='symmetric')
for i in range(len(Lines1)):
    image = PIL.Image.open(train_path + Lines1[i].split(';')[0])
    factor = max(128 / image.size[0], 128 / image.size[1])
    if factor < 1:
        factor = 1
    image = image.resize((int(factor * image.size[0]), int(factor * image.size[1])))
    image_1 = transformer.forward(image)
    image_1 = np.array(image_1)
    image_2 = transformer.forward(image)
    image_2 = np.array(image_2)
    cv2.imshow('hello', cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.imshow('hello', cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    train_x[2*i] = image_1
    train_y[2*i] = Lines1[i].split(';')[7]
    train_x[2*i + 1] = image_2
    train_y[2*i + 1] = Lines1[i].split(';')[7]
    
test_x = np.zeros((test_count, 128, 128, 3))
test_y = np.zeros((test_count,))

'''
for i in range(len(Lines2)):
    image = cv2.imread(test_path + Lines2[i].split(';')[0])
    image = cv2.resize(image, (128,128), 1, 1)
    cv2.imshow('hello', image)
    cv2.waitKey(0)
'''

for i in range(len(Lines2)):
    image = PIL.Image.open(test_path + Lines2[i].split(';')[0])
    factor = max(128 / image.size[0], 128 / image.size[1])
    if factor < 1:
        factor = 1
    image = image.resize((int(factor * image.size[0]), int(factor * image.size[1])))
    image = transformer.forward(image)
    test_x[i] = np.array(image)
    test_y[i] = Lines2[i].split(';')[7]
    
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

print("train_x shape: ", train_x.size())
print("train_y shape: ", train_y.size())
print("test_x shape: ", test_x.size())
print("test_y shape: ", test_y.size())

torch.save(train_x, 'train_x.pt')
torch.save(train_y, 'train_y.pt')
torch.save(test_x, 'test_x.pt')
torch.save(test_y, 'test_y.pt')