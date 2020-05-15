from torch import nn
import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


batch_size = 32
lr = 0.001
dr = 0.0001
num_epoch = 15
mode = True    # True表示使用全部数据集训练， False表示使用90%训练，10%验证

class Model(nn.Module):
    def __init__(self, bn=False):
        super(Model, self).__init__()
        if not bn:
            self. conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # 32*32
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 16*16
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8*8
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 4*4
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 32*32
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 16*16
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8*8
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 4*4
                nn.ReLU(inplace=True)
            )

        self.fc_layers = nn.Sequential(
            nn.Linear(128*4*4, 128),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 19),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # x = x.argmax(1)
        return x


class TrainData(Data.Dataset):
    def __init__(self, flag=True):
        if flag:
            self.data = torch.load('train_data.pth').unsqueeze(1).numpy()
            self.label = torch.load('train_label.pth').unsqueeze(1).numpy()
            # self.data = self.data.transpose(0, 3, 1, 2).astype('float32')  # conv2d 的输入维度为(N,C,H,W)
            self.len = self.data.shape[0]
        else:
            self.data = torch.load('tr_data.pth').unsqueeze(1).numpy()
            self.label = torch.load('tr_label.pth').unsqueeze(1).numpy()
            # self.data = self.data.transpose(0, 3, 1, 2).astype('float32')  # conv2d 的输入维度为(N,C,H,W)
            self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

    def get_data(self):
        return torch.from_numpy(self.data)

    def get_label(self):
        return torch.from_numpy(self.label)


def train():
    model = Model()
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            # nn.init.kaiming_normal(m.weight.data)  # 卷积层参数初始化
            m.bias.data.fill_(0)
    dataset = TrainData(mode)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=dr)

    loss_all = []
    acc_all = []
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for i, (data, label) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = Variable(data).cuda()
                label = Variable(label).cuda()
            else:
                data = Variable(data)
                label = Variable(label)

            pred = model(data).float()
            label = label
            loss = criterion(pred, label.type(torch.LongTensor).squeeze())
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d], Average Loss: %.4f'
                    % (epoch, num_epoch, i, len(train_loader), total_loss /float(batch_size * (i + 1))))

        class_truth = dataset.get_label().squeeze()
        class_pred = model(dataset.get_data()).argmax(1)
        num = torch.nonzero(class_truth - class_pred).size(0)
        print('acc for this epoch is:%.4f' % (1 - num/class_truth.size(0)))

        loss_all.append(total_loss)
        acc_all.append(1 - num/class_truth.size(0))

    if mode:
        torch.save(model.state_dict(), 'model.pth')
    else:
        torch.save(model.state_dict(), 'val_model.pth')

    return model, loss_all, acc_all

if __name__ =='__main__':
    model, loss_all, acc_all = train()
    x = np.arange(0, num_epoch)
    plt.title("loss for each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss_all)
    plt.show()

    plt.title("acc for each epoch")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.plot(x, acc_all)
    plt.show()

