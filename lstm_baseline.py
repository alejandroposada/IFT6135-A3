from torch.autograd import Variable
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, data_width, num_hidden):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(data_width + 1, num_hidden)
        self.mlp = nn.Linear(num_hidden, data_width)
        # self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        num_hidden = self.config['num_hidden']
        self.hidden = (autograd.Variable(torch.randn(1, batch_size, num_hidden)),
                       autograd.Variable(torch.randn((1, batch_size, num_hidden))))

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.mlp(x)
        return x

    def num_params(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params