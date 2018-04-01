from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import torch

class random_binary(Dataset):

    def __init__(self, max_seq_length, num_sequences, vector_dim=8, batch_Size=32, min_seq_length=1):
        '''
        :param max_seq_length: maximum sequence length allowed
        :param num_sequences: number of sequences
        :param vector_dim: dimension of input binary vector
        :param batch_Size: batch size
        '''
        self.max_seq_length = max_seq_length
        self.num_sequences = num_sequences
        self.vector_dim = vector_dim
        self.batch_Size = batch_Size
        self.min_seq_length = min_seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        vector_end = np.append(np.zeros(self.vector_dim), 1).reshape(self.vector_dim + 1, 1)  # x_{T+1}
        seq_length = np.random.randint(self.min_seq_length, self.max_seq_length)                 # T
        batch = np.ndarray(shape=(self.batch_Size, self.vector_dim + 1, seq_length+1))
        for i in range(self.batch_Size):
            sequence = np.random.binomial(1, 0.5, size=(self.vector_dim, seq_length))  # 8 x T
            sequence = np.append(sequence, np.zeros((1, seq_length)), axis=0)          # Append 0 at the end of each vector
            sequence = np.append(sequence, vector_end, axis=1)                         # Append x_{T+1} at the end of the seq.
            batch[i, :, :] = sequence
        return torch.FloatTensor(batch)


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

