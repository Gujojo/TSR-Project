import os
import cv2
import torch
from train import train, Model
import json

basicPath = os.sys.path[0]
trainPath = os.path.join(basicPath, "Classification\\Data\\Train")
testPath = os.path.join(basicPath, "Classification\\Data\\Test")

classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
             'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']


def cross_process():
    dirs = os.listdir(trainPath)
    train_data = []
    train_label = []
    validate_data = []
    validate_label = []
    class_idx = 0   # 类别编号
    for d in dirs:
        classPath = os.path.join(trainPath, d)
        images = os.listdir(classPath)
        tag = len(images) // 10
        data_temp = torch.zeros([len(images)-tag, 64, 64])
        label_temp = torch.zeros(len(images)-tag)
        data_val_temp = torch.zeros([tag, 64, 64])
        label_val_temp = torch.zeros(tag)
        i = 0
        for image in images:
            imagePath = os.path.join(classPath, image)      # 处理训练集图片
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = torch.tensor(img)
            if i < len(images)-tag:
                data_temp[i] = img
                # label_temp[i][class_idx] = 1    # 样本对应类别置为1
                label_temp[i] = class_idx
            else:
                data_val_temp[i-len(images)+tag] = img
                label_val_temp[i-len(images)+tag] = class_idx
            i = i + 1
        train_data.append(data_temp)
        train_label.append(label_temp)
        validate_data.append(data_val_temp)
        validate_label.append(label_val_temp)
        class_idx = class_idx + 1

    tr_data = train_data[0]
    tr_label = train_label[0]
    val_data = validate_data[0]
    val_label = validate_label[0]
    for i in range(1, len(train_data)):
        tr_data = torch.cat((tr_data, train_data[i]), 0)
        tr_label = torch.cat((tr_label, train_label[i]), 0)
        val_data = torch.cat((val_data, validate_data[i]), 0)
        val_label = torch.cat((val_label, validate_label[i]), 0)

    torch.save(tr_data, 'tr_data.pth')
    torch.save(tr_label, 'tr_label.pth')
    torch.save(val_data, 'val_data.pth')
    torch.save(val_label, 'val_label.pth')

def val():
    net = Model()
    net.eval()
    pre = torch.load('val_model.pth', map_location=torch.device('cpu'))
    net.load_state_dict(pre)
    data = torch.load('val_data.pth').unsqueeze(1)
    label = torch.load('val_label.pth')
    if torch.cuda.is_available():
        net = net.cuda()
        img = data.cuda()
    else:
        img = data
    torch.no_grad()
    class_pred = net(img).argmax(1)
    num = torch.nonzero(label.squeeze() - class_pred).size(0)
    acc = 1 - num/label.size(0)
    return acc

def test():
    net = Model()
    net.eval()
    pre = torch.load('model.pth', map_location=torch.device('cpu'))
    net.load_state_dict(pre)
    data = torch.load('test_data.pth').unsqueeze(1)
    if torch.cuda.is_available():
        net = net.cuda()
        img = data.cuda()
    else:
        img = data
    torch.no_grad()
    class_pred = net(img).argmax(1)
    torch.save(class_pred, 'class_pred.pth')
    return class_pred


def post_processing(class_pred, predPath):
    pred_list = {}
    images = os.listdir(testPath)
    i = 0
    for image in images:
        pred_list[image] = classList[int(class_pred[i])]
        i = i + 1
    with open(predPath, 'w') as outFile:
        json.dump(pred_list, outFile)



if __name__ =='__main__':
    # cross_process()   # 处理验证集数据
    acc = val()     # 交叉验证
    print(acc)
    predPath = os.path.join(basicPath, "Classification\\Data\\pred.json")
    class_pred = test()     # 在测试集上运行
    post_processing(class_pred, predPath)      # 输出json文件
