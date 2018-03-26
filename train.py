from model import NTM
import pickle
import numpy as np
from torch.autograd import Variable
import torch
from training_dataset import random_binary

#  load dataset
#with open('data/train.pkl', 'rb') as f:
#    dataset = pickle.load(f)

learning_rate = 0.01
batch_size = 32
cuda = True

#  for testing purposes only!
ntm = NTM(num_inputs=9, num_outputs=9, controller_size=100, controller_type='LSTM', controller_layers=15,
          memory_size=20, memory_feature_size=15, integer_shift=3)

training_dataset = random_binary(max_seq_length=20, num_sequences=10, vector_dim=8, batch_Size=batch_size)

optimizer = torch.optim.Adam(ntm.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()

'''
def train(decoder, optimizer, criterion, inp, target, batch_size, chunk_len, cuda):
    hidden = decoder.init_hidden(batch_size)
    if cuda:
        if decoder.model == "gru":
            hidden = hidden.cuda()
        else: # lstm
            hidden = (hidden[0].cuda(), hidden[1].cuda())
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    optimizer.step()

    return loss.data[0] / chunk_len
'''
def train(ntm, optimizer, criterion, batch):
    for i in range(batch.size()[2]):
        x = batch[:, :, i]
        output, next_r = ntm.forward(x=batch, r=next_r)
        print(output)
        print(next_r)

train_loss = 0
for batch in training_dataset:
    batch = Variable(torch.FloatTensor(batch))
    next_r = Variable(torch.FloatTensor(np.random.rand(15)))
    train_loss += train(ntm, optimizer, criterion, batch)





#print('done')