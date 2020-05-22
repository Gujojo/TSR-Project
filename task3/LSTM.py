import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.datasets as dset
import torchvision.transforms as transforms


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
num_classes = 11
batch_size = 2
learning_rate = 2e-4
num_epochs = 1000
training_dir = ".\\Train"
testing_dir = ".\\Test"
label_dir = ".\\Test\\test\\test.json"

train_dataset = dset.ImageFolder(root=training_dir,
                                 transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
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
            # input: 64*64*3, output: 30*30*4
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 30*30*4, output: 13*13*8
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 13*13*8, output: 4*4*16
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),

            # input: 4*4*16, output: 2*2*32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
        )

        # input: 2*2*32, output: 11
        self.FC1 = nn.Sequential(
            nn.Linear(in_features=32 * 2 * 2, out_features=num_classes, bias=True),
            nn.ReLU(inplace=True)
        )

        # input: 11, output: (bidirectional+1)*hidden_size
        self.input_size = num_classes
        self.hidden_size = 20
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)
        self.hidden = self.init_hidden()

        # input: (bidirectional+1)*hidden_size, output: 11
        self.FC2 = nn.Sequential(
            nn.Linear(in_features=2*self.hidden_size if self.lstm.bidirectional else self.hidden_size,
                      out_features=num_classes,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=2)
        )

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             betas=(0.5, 0.999))

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
        loss = criterion(outputs, labels)
        # backward
        tmp = loss.cpu()
        running_loss += tmp.data.numpy()
        loss.backward()
        optimizer.step()
        cnt += 1
    running_loss = running_loss / cnt
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, running_loss))

class_dict = train_dataset.class_to_idx
pred = []
for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    model.zero_grad()
    model.hidden = model.init_hidden()
    outputs = model(images)
    file_name = test_loader.dataset.samples[i*batch_size: (i+1)*batch_size]
    tmp = np.argmax(outputs.detach().numpy(), axis=1)
    pred = pred + tmp.tolist()

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
