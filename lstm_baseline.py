import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.lstm = nn.LSTM(num_inputs, num_hidden)
        self.mlp = nn.Linear(num_hidden, num_inputs)

    def init_hidden(self, batch_size):
        num_hidden = self.num_hidden
        self.hidden = (autograd.Variable(torch.randn((1, batch_size, self.num_hidden))),
                       autograd.Variable(torch.randn((1, batch_size, self.num_hidden))))

    def forward(self, x):
        x, self.hidden = self.lstm(x.unsqueeze(0), self.hidden)
        x = self.mlp(x)
        return F.sigmoid(x)

    def num_params(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params