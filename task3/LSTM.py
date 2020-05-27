import numpy as np
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
num_classes = 11
batch_size = 11
learning_rate = 5e-5
num_epochs = 600
training_dir = ".\\Train"
testing_dir = ".\\Test"
label_dir = ".\\Test\\test\\test.json"

train_dataset = dset.ImageFolder(root=training_dir,
                                 transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=0,
                               drop_last=True)
test_dataset = dset.ImageFolder(root=testing_dir,
                                transform=transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DCNN = nn.Sequential(
            # input: 64*64*3, output: 30*30*8
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 30*30*8, output: 13*13*16
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 13*13*16, output: 4*4*32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 4*4*32, output: 2*2*32
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
        )

        # input: 2*2*32, output: batch*1*30
        self.FC1 = nn.Sequential(
            nn.Linear(in_features=32 * 2 * 2, out_features=30, bias=True),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        # input: batch*1*30, output: batch*1*(bidirectional+1)*hidden_size
        self.input_size = 30
        self.hidden_size = 20
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.hidden = self.init_hidden()

        # input: batch*1*(bidirectional+1)*hidden_size, output: batch*1*11
        self.FC2 = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size if self.lstm.bidirectional else self.hidden_size,
                      out_features=num_classes,
                      bias=True),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Softmax(dim=2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_hidden(self):
        if self.lstm.bidirectional:
            return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device))
        else:
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        x = self.DCNN(x)
        x = x.view(x.size(0), -1)
        x = self.FC1(x)
        x = x.unsqueeze(1)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.FC2(x)
        x = x.squeeze(1)
        return x


model = Model().to(device)
model.train()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             betas=(0.6, 0.999),
                             weight_decay=0.01)

Loss = []
for epoch in range(num_epochs):
    running_loss = 0
    cnt = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad()
        model.hidden = model.init_hidden()
        # forward
        outputs = model(images)
        target = torch.zeros(batch_size, num_classes)
        for n in range(batch_size):
            target[n][labels[n]] = 1
        loss = criterion(outputs, target)
        # backward
        tmp = loss.cpu()
        running_loss += tmp.data.numpy()
        loss.backward()
        optimizer.step()
        cnt += 1
    running_loss = running_loss / cnt
    Loss.append(running_loss)
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, running_loss))

acc = 0
cnt = 0
reference = []
for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    model.zero_grad()
    model.hidden = model.init_hidden()
    outputs = model(images)
    reference = outputs.detach().numpy()
    tmp = np.argmax(outputs.detach().numpy(), axis=1).tolist()
    for n in range(len(tmp)):
        cnt += 1
        if tmp[n] == labels[n]:
            acc += 1
print(acc / cnt)

# test
model.eval()
class_dict = train_dataset.class_to_idx
pred = []
pred_dict = {}
for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    model.zero_grad()
    model.hidden = model.init_hidden()
    outputs = model(images)
    file_name = test_loader.dataset.samples[i*batch_size: (i+1)*batch_size]
    tmp = []
    for pic in range(num_classes):
        mse = []
        a = outputs[pic].detach().numpy()
        for c in range(num_classes):
            b = reference[c]
            test = np.linalg.norm(a-b)
            mse.append(test)
        tmp.append(np.argmax(mse).tolist())
    pred = pred + tmp
    for n in range(batch_size):
        pred_dict[file_name[n]] = tmp[n]

with open(label_dir, 'r') as inFile:
    labelDict = json.load(inFile)
test = []
for key in labelDict.keys():
    item = labelDict[key]
    tmp = class_dict[item]
    test.append(tmp)

acc = 0
for i in range(len(test)):
    if test[i] == pred[i]:
        acc += 1
acc /= len(test)
print(acc)

x = np.arange(0, num_epochs)
plt.title("loss for each epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x, Loss)
plt.show()
