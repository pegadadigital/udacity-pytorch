import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class Model(torch.nn.Module):
    def __init__(self):
        """
        In the construtor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        # Linear Layers
        # torch.nn.Linear
        # torch.nn.Bilinear
        self.linear = torch.nn.Linear(1, 1) # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # Here we are using Sigmoid as activation function but there are others:
        # F.relu
        # F.relu6
        # F.elu
        # F.selu
        # F.prelu
        # F.leaky_relu
        # F.threshold
        # F.hardtanh
        # F.tanh
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model

# List of loss functions
# torch.nn.L1Loss
# torch.nn.MSELoss
# torch.nn.CrossEntropyLoss
# torch.nn.NLLLoss
# torch.nn.PoissonNLLLoss
# torch.nn.KLDivLoss
# torch.nn.BCELoss
# torch.nn.BCEWithLogitsLoss
# torch.nn.MarginRankingLoss
# torch.nn.HingeEmbeddingLoss
# torch.nn.MultiLabelMarginLoss
# torch.nn.SmoothL1Loss
# torch.nn.SoftMarginLoss
# torch.nn.MultiLabelSoftMarginLoss
# torch.nn.CosineEmbeddingLoss
# torch.nn.MultiMarginLoss
# torch.nn.TripletMarginLoss
# Using Binary Cross Entropy
#criterion = torch.nn.BCELoss(size_average=False) deprecated
criterion = torch.nn.BCELoss(reduction = 'sum')

# List of Optimizers
# torch.optim.Adagrad
# torch.optim.Adam
# torch.optim.Adamax
# torch.optim.ASGD
# torch.optim.LBFGS
# torch.optim.RMSprop
# torch.optim.Rprop
# torch.optim.SGD
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
plt.plot(x_data.numpy(), y_data.numpy(), label = 'from data', alpha = .5)
plt.plot(x_data.numpy(), y_pred.detach().numpy()>0.5, label = 'prediction', alpha = 0.5)
plt.legend()
plt.show()
