from torch.utils.data import Dataset
import numpy as np


class random_binary(Dataset):

    def __init__(self, max_seq_length, num_sequences, vector_dim=8, batch_Size=32):
        self.max_seq_length = max_seq_length
        self.num_sequences = num_sequences
        self.vector_dim = vector_dim
        self.batch_Size = batch_Size

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        vector_end = np.append(np.zeros(self.vector_dim), 1).reshape(self.vector_dim + 1, 1)  # x_{T+1}
        seq_length = np.random.randint(1, self.max_seq_length)                 # T
        batch = np.ndarray(shape=(self.batch_Size, self.vector_dim + 1, seq_length+1))
        for i in range(self.batch_Size):
            sequence = np.random.binomial(1, 0.5, size=(self.vector_dim, seq_length))  # 8 x T
            sequence = np.append(sequence, np.zeros((1, seq_length)), axis=0)          # Append 0 at the end of each vector
            sequence = np.append(sequence, vector_end, axis=1)                         # Append x_{T+1} at the end of the seq.
            batch[i, :, :] = sequence
        return batch
