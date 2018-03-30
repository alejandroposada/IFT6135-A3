import numpy as np
import torch
from torch.autograd import Variable


def sequence_loader(num_batches, batch_size, num_bits=8, min_length=1, max_length=20):
    """
    Adapted from https://github.com/loudinthecloud/pytorch-ntm/blob/2d4a0bbc1 \
    beec4b2f83b92dd87b2d2fd28b8245e/tasks/copytask.py.
    """
    for i in range(num_batches):
        length = np.random.randint(min_length, max_length)
        sequence = Variable(torch.from_numpy(np.random.binomial(1, 0.5, (length, batch_size, num_bits))))

        x = Variable(torch.zeros(length + 1, batch_size, num_bits + 1))
        x[:length, :, :num_bits] = sequence
        x[length, :, num_bits] = 1.0
        y = sequence.clone()

        dummy_ = Variable(torch.zeros(length, batch_size, num_bits) + 0.5)
        dummy = Variable(torch.zeros(length, batch_size, num_bits + 1))
        dummy[:length, :, :num_bits] = dummy_

        yield x.float(), y.float(), dummy.float()

