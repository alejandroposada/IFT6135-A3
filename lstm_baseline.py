import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.lstm = nn.LSTM(num_inputs, num_hidden)
        self.mlp = nn.Linear(num_hidden, num_inputs)

        self.init_weights(self.lstm)
        self.init_weights(self.mlp)

    def init_weights(self, layer):
        # Initialize weights
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

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


class LSTM_v2(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers=1):
        super(LSTM_v2, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_inputs, num_hidden, num_layers)
        self.mlp = nn.Linear(num_hidden, num_inputs - 1)  # TODO: make num_inputs 8 instead of 9. More clear!

        self.init_weights(self.lstm)
        self.init_weights(self.mlp)

    def init_weights(self, layer):
        # Initialize weights
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def init_hidden(self, batch_size, cuda):
        self.hidden = (autograd.Variable(torch.randn((self.num_layers, batch_size, self.num_hidden))),
                       autograd.Variable(torch.randn((self.num_layers, batch_size, self.num_hidden))))
        if cuda:
            self.hidden = (autograd.Variable(torch.randn((self.num_layers, batch_size, self.num_hidden)).cuda()),
                           autograd.Variable(torch.randn((self.num_layers, batch_size, self.num_hidden))).cuda())

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.mlp(x)
        return F.sigmoid(x)

    def num_params(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params