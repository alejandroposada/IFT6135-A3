from model import NTM
import pickle
import numpy as np
from torch.autograd import Variable
import torch
from training_dataset import random_binary

#  load dataset
#with open('data/train.pkl', 'rb') as f:
#    dataset = pickle.load(f)


#  for testing purposes only!
ntm = NTM(num_inputs=9, num_outputs=9, controller_size=100, controller_type='LSTM', controller_layers=15,
          memory_size=20, memory_feature_size=15, integer_shift=3)

training_dataset = random_binary(max_seq_length=20, num_sequences=10, vector_dim=8, batch_Size=32)

for batch in training_dataset:
    batch = Variable(torch.FloatTensor(batch))
    next_r = Variable(torch.FloatTensor(np.random.rand(15)))
    for i in range(batch.size()[2]):
        x = batch[:, :, i]
        output, next_r = ntm.forward(x=batch, r=next_r)
        print(output)
        print(next_r)

#print('done')