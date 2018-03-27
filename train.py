from model import NTM
import numpy as np
from torch.autograd import Variable
import torch
from training_dataset import random_binary
from train_utils import save_checkpoint, evaluate

# Hyperparameters
learning_rate = 0.01
batch_size = 32
cuda = True
memory_feature_size = 20
num_inputs = 9
num_outputs = 9
checkpoint_interval = 512
num_sequences = 20
total_batches = 40

# Seeding
SEED = 1000
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model Loading
ntm = NTM(num_inputs=9, num_outputs=9, controller_size=100, controller_type='LSTM', controller_layers=1,
          memory_size=100, memory_feature_size=memory_feature_size, integer_shift=3, batch_size=batch_size)

# Dataset creation
training_dataset = random_binary(max_seq_length=20, num_sequences=20, vector_dim=8, batch_Size=batch_size)
testing_dataset = random_binary(max_seq_length=10, num_sequences=5, vector_dim=8, batch_Size=batch_size)

# Optimizer type and loss function
optimizer = torch.optim.Adam(ntm.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()

# Constants for keeping track
total_examples = 0
losses = []
costs = []
seq_lens = []
for batch in training_dataset:
    batch = Variable(torch.FloatTensor(batch))
    next_r = Variable(torch.FloatTensor(np.random.rand(batch_size, memory_feature_size) * 0.05))
    lstm_h, lstm_c = ntm.controller.create_state(batch_size)

    optimizer.zero_grad()
    output = Variable(torch.zeros(batch.size()))
    for i in range(batch.size()[2]):
        x = batch[:, :, i]
        output[:, :, i], next_r, lstm_h, lstm_c = ntm.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)

    loss = criterion(output, batch)
    loss.backward(retain_graph=True)
    optimizer.step()

    print("Current Batch Loss:", loss.data)
    total_examples += batch_size

    # The cost is the number of error bits per sequence
    binary_output = output.clone().data
    binary_output.apply_(lambda y: 0 if y < 0.5 else 1)
    cost = torch.sum(torch.abs(binary_output - batch.data))

    losses += [loss.data[0]]
    costs += [cost/batch_size]
    seq_lens += [batch.size(2)]

    # Checkpoint model
    if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):
        print("Saving Checkpoint!")
        save_checkpoint(ntm, total_examples/batch_size, losses, costs, seq_lens)

        # Evaluate model on this saved checkpoint
        test_cost, prediction, input = evaluate(ntm, testing_dataset, batch_size, memory_feature_size)
        print("Total Test Cost (in bits per sequence):", test_cost)
        print("Example of Input/Output")
        print("prediction:", prediction[0])
        print("Input:", input[0])

    if total_examples / checkpoint_interval >= total_batches:
        break

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
'''


#print('done')