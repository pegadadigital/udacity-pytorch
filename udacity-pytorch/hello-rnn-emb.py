import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(777)  # reproducibility

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
y_data = [1, 0, 2, 3, 3, 4]    # ihello

# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.LongTensor(x_data)).to(device)
labels = Variable(torch.LongTensor(y_data)).to(device)

num_classes = 5
input_size = 5
embedding_size = 10  # embedding size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


class Model(nn.Module):

    def __init__(self,num_layers,hidden_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=5, batch_first=True).to(device)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        emb = self.embedding(x).to(device)
        emb = emb.view(batch_size, sequence_length, -1)

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes)).to(device)


# Instantiate RNN model
model = Model(num_layers,hidden_size).to(device)
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
model_loss = []

# Train the model
for epoch in range(100):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    model_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.cpu().numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")

# loss function
plt.plot(range(1,len(model_loss)+1), model_loss)
plt.title('loss function')
plt.xlabel('iterations')
plt.ylabel('loss function')
plt.show()