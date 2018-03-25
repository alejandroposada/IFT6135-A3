import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform


class Controller(nn.Module):
    """
    Controller for NTM.
    """
    def __init__(self, network, input_dim, output_dim, num_layers):
        """network: object which takes as input r_t and x_t and returns h_t
        """
        super(Controller, self).__init__()
        self.network = network            # A LSTM or MLP network
        self.input_dim = input_dim        # (8 + 1) + M*num_heads
        self.output_dim = output_dim      # 100
        self.num_layers = num_layers      # 1

    def reset_parameters(self):
        for param in self.network.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                xavier_uniform(param)

    def size(self):
        """Returns the size of the controller 
        """
        return self.num_inputs, self.num_outputs


class LSTMController(Controller):
    """
    LSTM controller for the NTM.
    """
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__(nn.LSTM(input_size=input_dim,
                            hidden_size=output_dim,
                            num_layers=num_layers), input_dim, output_dim, num_layers)

        # From https://github.com/fanxiao001/ift6135-assignment/blob/master/assignment3/NTM/controller.py
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def forward(self, x, r, state):
        x = x.unsqueeze(0)
        x = torch.cat([x] + r, dim=1)
        output, state = self.lstm(x, state)
        return output.squeeze(0), state

    def create_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        # From https://github.com/fanxiao001/ift6135-assignment/blob/master/assignment3/NTM/controller.py
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c


class MLPController(Controller):
    """
    MLP controller for the NTM.
    """
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__(nn.Linear(input_dim, output_dim), input_dim, output_dim, num_layers)

    def forward(self, x, r, state):
        x = x.unsqueeze(0)
        x = torch.cat([x] + r, dim=1)
        output = self.mlp(x)
        return output.squeeze(0), state

    def create_state(self, batch_size):
        return torch.zeros(1, batch_size, 1)


class NTMReadHead(nn.Module):
    def __init__(self):
        super(NTMReadHead, self).__init__()

    def forward(self, w, memory):
        """(2)
        """
        pass

class NTMWriteHead(nn.Module):
    def __init__(self):
        super(NTMWriteHead, self).__init__()

    def forward(self, w, memory, e, a):
        """(3) and (4)
        """
        pass


class NTMAttention(nn.Module):
    def __init__(self):
        super(NTMAttention, self).__init__()

    def forward(self, beta, kappa, gamma, g, s):
        """(5), (6), (7), (8), (9)
        """
        pass


class NTM(nn.Module):
    """
    Neural Turing Machine
    """
    def __init__(self, num_inputs, num_outputs, controller, attention,
                 read_head, write_head, memory_size, memory_feature_size):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`Controller`
        :param attention: :class:`NTMAttention`
        :param read_head: list of :class:`NTMReadHead`
        :param write_head: list of :class:`NTMWriteHead`
        """
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.attention = attention
        self.read_head = read_head
        self.write_head = write_head
        self.memory = np.zeros(shape=(memory_size, memory_feature_size))

        # Initialize a fully connected layer to produce the actual output:
        self.fc = nn.Linear(self.controller.size(), num_outputs)

    def convert_to_params(self, output):
        """Transform output from controller into parameters for attention and write heads
        :param output: output from controller.
        """
        beta, kappa, gamma, g, s, e, a = 0, 0, 0, 0, 0, 0, 0
        return beta, kappa, gamma, g, s, e, a

    def forward(self, x, r):
        """Perform forward pass from the NTM.
        :param x: current input.
        :param r: previous read head output.
        """

        o = self.controller.forward(x, r)
        beta, kappa, gamma, g, s, e, a = self.convert_to_params(o)
        w = self.attention.forward(beta, kappa, gamma, g, s)
        next_r = self.read_head.forward(w, self.memory)
        self.memory = self.write_head.forward(w, self.memory, e, a)

        # Generate Output
        output = F.sigmoid(self.fc(o))

        return output, next_r





