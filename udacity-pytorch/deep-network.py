import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

# setting working directory
import os
os.chdir("D:\\Projetos\\udacity-pytorch\\udacity-pytorch")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
xy = np.loadtxt('data/diabetes.csv',delimiter = ',', dtype = np.float32)
x_data = Variable(torch.from_numpy(xy[:,0:-1])).float().to(device)
y_data = Variable(torch.from_numpy(xy[:,[-1]])).float().to(device)


# Number of features and outputs
print(x_data.shape)
print(y_data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        """
        In the construtor we instantiate nn.Linear modules and
        nn.sigmoid
        """
        super(Model, self).__init__()
        # Linear Layers
        # torch.nn.Linear
        # Sigmoid Layers
        # torch.nn.Sigmoid
        self.l1 = torch.nn.Linear(8, 6).to(device) # Eight in and six out
        self.l2 = torch.nn.Linear(6, 4).to(device) # Six in and four out
        self.l3 = torch.nn.Linear(4, 1).to(device) # Four in and one out

        self.sigmoid = torch.nn.Sigmoid().to(device)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # The sequence of layers are Linear > Sigmoid > Linear > Sigmoid >Linear > Sigmoid
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

# our model
model = Model().to(device)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model

# Using Binary Cross Entropy
#criterion = torch.nn.BCELoss(size_average=False) deprecated
criterion = torch.nn.BCELoss(reduction = 'sum').to(device)

# Using Stochastic Gradient Descent
# with learning ratio 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model_loss = []

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    model_loss.append(loss.data[0])
    print(epoch, loss.data[0])

    # Zero gradients, perfom a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# loss function
plt.plot(range(1,501), model_loss)
plt.xlabel('epoch')
plt.ylabel('loss function')
plt.show()

# predict
y_pred = model.forward(x_data)

# confusion matrix
confusion_matrix(y_data.cpu().numpy(),y_pred.cpu().detach().numpy()>0.5)

# accuracy
accuracy_score(y_data.cpu().numpy(),y_pred.cpu().detach().numpy()>0.5)