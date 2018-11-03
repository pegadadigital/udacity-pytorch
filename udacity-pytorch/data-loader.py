import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

# setting working directory
import os
os.chdir("D:\\Projetos\\udacity-pytorch\\udacity-pytorch")


class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        # super(Dataset, self).__init__()
        xy = np.loadtxt('data/diabetes.csv',delimiter = ',', dtype = np.float32)
        self.len = xy.shape[0]
        self.x_data = Variable(torch.from_numpy(xy[:,0:-1])).float().to(device)
        self.y_data = Variable(torch.from_numpy(xy[:,[-1]])).float().to(device)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 3).to(device)
        self.l2 = torch.nn.Linear(3, 1).to(device)
        self.sigmoid = torch.nn.Sigmoid().to(device)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        y_pred = self.sigmoid(self.l2(out1))
        return y_pred


dataset = DiabetesDataset()

train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True)
                          #num_workers=2)
# our model
model = Model().to(device)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.BCELoss(reduction = 'sum').to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model_loss = []

# Training loop
for epoch in range(3):
    for i, data in enumerate(train_loader,0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        model_loss.append(loss.item())
        #print(epoch, i, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# loss function
plt.plot(range(1,len(model_loss)+1), model_loss)
plt.xlabel('iterations')
plt.ylabel('loss function')
plt.show()

# predict
y_pred = model.forward(dataset.x_data)

# confusion matrix
confusion_matrix(dataset.y_data.cpu().numpy(),y_pred.cpu().detach().numpy()>0.5)

# accuracy
accuracy_score(dataset.y_data.cpu().numpy(),y_pred.cpu().detach().numpy()>0.5)
