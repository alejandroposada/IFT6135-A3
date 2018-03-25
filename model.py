import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

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
                 read_head, write_head, memory_size, memory_feature_size, integer_shift):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`Controller`
        :param attention: :class:`NTMAttention`
        :param read_head: list of :class:`NTMReadHead`
        :param write_head: list of :class:`NTMWriteHead`
        :param memory_size: N in the paper
        :param memory_feature_size: M in the paper
        :param integer_shift: allowed integer shift (see pg 8 of paper)
        """
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.attention = attention
        self.read_head = read_head
        self.write_head = write_head
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size
        self.integer_shift = integer_shift
        self.memory = Variable(torch.zeros(self.memory_size, self.memory_feature_size))

        # Initialize a fully connected layer to produce the actual output:
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        # Corresponding to beta, kappa, gamma, g, s, e, a sizes from the paper
        self.params = ['beta', 'kappa', 'gamma', 'g', 's', 'e', 'a']
        self.params_lengths = [1, self.memory_feature_size, 1, 1, self.integer_shift,
                               self.memory_feature_size, self.memory_feature_size]

        self.fc_params = nn.Linear(self.num_outputs, sum(self.params_lengths))

        # Corresponding to beta, kappa, gamma, g, s, e, a
        # (choice of activations selected to obey corresponding domain restrictions from the paper)
        self.activations = {'beta': F.softplus,
                            'kappa': lambda x: x,
                            'gamma': lambda x: 1 + F.softplus(x),
                            'g': F.sigmoid,
                            's': lambda x: F.softmax(F.softplus(x)),
                            'e': F.sigmoid,
                            'a': F.sigmoid}


    def convert_to_params(self, output):
        """Transform output from controller into parameters for attention and write heads
        :param output: output from controller.
        """
        to_return = {'beta': 0,
                     'kappa': 0,
                     'gamma': 0,
                     'g': 0,
                     's': 0,
                     'e': 0,
                     'a': 0}
        o = self.fc_params(output)
        l = np.cumsum([0] + self.params_lengths)
        for idx in range(len(l)-1):
            to_return[self.params[idx]] = self.activations[self.params[idx]](o[l[idx]:l[idx+1]])

        return to_return

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


#  for testing purposes only!
controller = Controller(network=0)
attn =NTMAttention()
read_head = NTMReadHead()
write_head = NTMWriteHead()

ntm = NTM(num_inputs=10, num_outputs=100, controller=controller,
          attention=attn, read_head=read_head, write_head=write_head,
          memory_size=20, memory_feature_size=15, integer_shift=3)

output = Variable(torch.FloatTensor(np.random.rand(100)))
x = ntm.convert_to_params(output)

print('done')


