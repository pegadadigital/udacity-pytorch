from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# setting working directory
import os
os.chdir("D:\\Projetos\\udacity-pytorch\\udacity-pytorch")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training settings
batch_size = 64

# MNIST Dataset
# images 28x28, 784 pixels
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520).to(device)
        self.l2 = nn.Linear(520, 320).to(device)
        self.l3 = nn.Linear(320, 240).to(device)
        self.l4 = nn.Linear(240, 120).to(device)
        self.l5 = nn.Linear(120, 10).to(device)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
model_loss = []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        model_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()

# loss function
plt.plot(range(1,len(model_loss)+1), model_loss)
plt.title('loss function')
plt.xlabel('iterations')
plt.ylabel('loss function')
plt.show()

# predicting
targets = []
preds = []
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    targets.append(target.cpu().numpy())
    preds.append(pred.cpu().detach().numpy())

targets = np.concatenate(targets).ravel()
preds= np.concatenate(preds).ravel()
# confusion matrix
confusion_matrix(targets,preds)
