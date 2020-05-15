import os
import numpy as np
import json
import cv2
import torch

basicPath = os.sys.path[0]
trainPath = os.path.join(basicPath, "Classification\\Data\\Train")
testPath = os.path.join(basicPath, "Classification\\Data\\Test")

classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
             'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']

dirs = os.listdir(trainPath)
data = []
label = []
class_idx = 0   # 类别编号
for d in dirs:
    classPath = os.path.join(trainPath, d)
    images = os.listdir(classPath)
    data_temp = torch.zeros([len(images), 64, 64])
    label_temp = torch.zeros(len(images))
    i = 0
    for image in images:
        imagePath = os.path.join(classPath, image)      # 处理训练集图片
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = torch.tensor(img)
        data_temp[i] = img
        # label_temp[i][class_idx] = 1    # 样本对应类别置为1
        label_temp[i] = class_idx
        i = i + 1
    data.append(data_temp)
    label.append(label_temp)
    class_idx = class_idx + 1


train_data = data[0]
train_label = label[0]
for i in range(1, len(data)):
    train_data = torch.cat((train_data, data[i]), 0)
    train_label = torch.cat((train_label, label[i]), 0)

torch.save(train_data, 'train_data.pth')
torch.save(train_label, 'train_label.pth')


images = os.listdir(testPath)
test_data = torch.zeros([len(images), 64, 64])
i = 0
for image in images:
    imagePath = os.path.join(testPath, image)  # 处理训练集图片
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = torch.tensor(img)
    test_data[i] = img
    i = i + 1
torch.save(test_data, 'test_data.pth')