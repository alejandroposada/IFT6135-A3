import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F

class Controller(nn.Module):
    """
    controller for NTM
    """
    def __init__(self, network):
        """network: object which takes as input r_t and x_t and returns h_t
        """
        super(Controller, self).__init__()
        self.network = network

    def forward(self, x, r):
        pass

    def size(self):
        """Returns the size of the controller output (100 for us)
        """
        pass


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





