import torch
from torch import nn as nn


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


class NTMWriteHead(nn.Module):
    def __init__(self):
        super(NTMWriteHead, self).__init__()


class NTMMemory(nn.Module):
    def __init__(self):
        super(NTMMemory, self).__init__()

    def size(self):
        """Returns the size of the controller output (100 for us)
        """
        pass


class NTM(nn.Module):
    """
    Neural Turing Machine
    """
    def __init__(self, num_inputs, num_outputs, controller, memory, read_head, write_head):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`Controller`
        :param memory: :class:`NTMMemory`
        :param read_head: list of :class:`NTMReadHead`
        :param write_head: list of :class:`NTMWriteHead`
        """
        super(NTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.read_head = read_head
        self.write_head = write_head

        self.N, self.M = self.memory.size()
        _, self.controller_size = controller.size()


