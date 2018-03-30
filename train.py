from model import NTM
from lstm_baseline import LSTM
import numpy as np
from torch.autograd import Variable
import torch
from training_dataset import sequence_loader
from train_utils import save_checkpoint, evaluate, evaluate_lstm_baseline
import argparse
from training_dataset import sequence_loader


def run(learning_rate, batch_size, cuda, memory_feature_size, num_inputs, num_outputs,
        controller_size, controller_type, controller_layers, memory_size, integer_shift,
        checkpoint_interval, total_batches, model_file):
    # Seeding
    SEED = 1000
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Model Loading
    if model_file == 'None':
        ntm = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                  controller_type=controller_type, controller_layers=controller_layers,
                  memory_size=memory_size, memory_feature_size=memory_feature_size, integer_shift=integer_shift,
                  batch_size=batch_size, use_cuda=cuda)
        # Constants for keeping track
        total_examples = 0
        losses = []
        costs = []
        seq_lens = []
    else:
        from_before = torch.load(model_file)
        state_dict = from_before['state_dict']
        controller_type = from_before['controller_type']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        controller_size = from_before['controller_size']
        controller_layers = from_before['controller_layers']
        memory_size = from_before['memory_size']
        memory_feature_size = from_before['memory_feature_size']
        integer_shift = from_before['integer_shift']
        batch_size = from_before['batch_size']
        cuda = from_before['cuda']
        ntm = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                  controller_type=controller_type, controller_layers=controller_layers,
                  memory_size=memory_size, memory_feature_size=memory_feature_size, integer_shift=integer_shift,
                  batch_size=batch_size, use_cuda=cuda)
        ntm.load_state_dict(state_dict)
        losses = from_before['loss']
        costs = from_before['cost']
        seq_lens = from_before['seq_lengths']
        total_examples = from_before['total_examples']

    # Dataset creation
    training_dataset = random_binary(max_seq_length=20, num_sequences=200, vector_dim=8,
                                     batch_Size=batch_size)
    testing_dataset = random_binary(max_seq_length=10, num_sequences=50, vector_dim=8,
                                    batch_Size=batch_size)

    # Optimizer type and loss function
    optimizer = torch.optim.Adam(ntm.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    np.random.seed(SEED)  # reset training seed to ensure that batches remain the same between runs!
    for batch in training_dataset:
        batch = Variable(batch)
        if cuda:
            batch = batch.cuda()
        next_r = ntm.read_head.create_state(batch_size, memory_feature_size)
        if controller_type == 'LSTM':
            lstm_h, lstm_c = ntm.controller.create_state(batch_size)

        optimizer.zero_grad()
        output = Variable(torch.zeros(batch.size()))
        if cuda:
            output = output.cuda()
        # Read batch in
        for i in range(batch.size()[2]):
            x = batch[:, :, i]
            if controller_type == 'LSTM':
                _, next_r, lstm_h, lstm_c = ntm.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                _, next_r = ntm.forward(x=x, r=next_r)
        # Output response
        x = Variable(torch.zeros(batch.size()[0:2]))
        if cuda:
            x = x.cuda()

        for i in range(batch.size()[2]):
            if controller_type == 'LSTM':
                output[:, :, i], next_r, lstm_h, lstm_c = ntm.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                output[:, :, i], next_r = ntm.forward(x=x, r=next_r)

        loss = criterion(output, batch)
        loss.backward(retain_graph=True)
        optimizer.step()

        print("Current Batch Loss:", round(loss.data[0], 3))
        total_examples += batch_size

        # The cost is the number of error bits per sequence
        binary_output = output.clone().data
        binary_output = binary_output > 0.5
        cost = torch.sum(torch.abs(binary_output.float() - batch.data))

        losses += [loss.data[0]]
        costs += [cost / batch_size]
        seq_lens += [batch.size(2)]

        # Checkpoint model
        if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):
            print("Saving Checkpoint!")
            save_checkpoint(ntm, total_examples / batch_size, losses, costs, seq_lens, total_examples, controller_type,
                            num_inputs, num_outputs, controller_size, controller_layers, memory_size,
                            memory_feature_size, integer_shift, batch_size, cuda)

            # Evaluate model on this saved checkpoint
            test_cost, prediction, input = evaluate(model=ntm, testset=testing_dataset, batch_size=batch_size,
                                                    memory_feature_size=memory_feature_size,
                                                    controller_type=controller_type, cuda=cuda)
            print("Total Test Cost (in bits per sequence):", test_cost)
            print("Example of Input/Output")
            print("prediction:", prediction[0])
            print("Input:", input[0])
        if total_examples / checkpoint_interval >= total_batches:
            break


def train_baseline(learning_rate, batch_size, cuda, num_inputs, num_outputs,
                   hidden_dim, num_layers, checkpoint_interval, print_interval, total_batches, model_file):
    """
    Train LSTM baseline.
    """
    # Seeding
    SEED = 1000
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Model Loading
    if model_file == 'None':
        model = LSTM(num_inputs, hidden_dim, num_layers)
        if cuda:
            model.cuda()
        # Constants for keeping track
        total_examples = 0
        losses = []
        costs = []
        seq_lens = []
        prev_print_batch = 0
    else:
        from_before = torch.load(model_file)
        state_dict = from_before['state_dict']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        batch_size = from_before['batch_size']
        cuda = from_before['cuda']
        model = LSTM(num_inputs, hidden_dim)
        total_examples = from_before['total_examples']
        losses = from_before['loss']
        costs = from_before['cost']
        seq_lens = from_before['seq_lengths']
        model.load_state_dict(state_dict)
    if cuda:
        model.cuda()

    # Dataset creation
    dataloader = sequence_loader(num_batches=total_batches, batch_size=batch_size, max_length=20)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.BCELoss()

    for batch_num, (x, y, dummy) in enumerate(dataloader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
            dummy = dummy.cuda()

        # Forward pass
        model.init_hidden(batch_size, cuda)
        model.forward(x)
        output = model.forward(dummy)

        # Backward pass
        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_examples += batch_size

        # Compute cost (error bits per sequence)
        binary_output = output.clone().data
        binary_output = binary_output > 0.5
        cost = torch.sum(torch.abs(binary_output.float() - y.data))

        losses += [loss.data[0]]
        costs += [cost / batch_size]
        seq_lens += [y.size(0)]

        # Show progress
        if batch_num % print_interval == 0:
            print("Batch %d, loss %f, cost %f" % (batch_num,
                                                  sum(losses[prev_print_batch:-1]) / print_interval,
                                                  sum(costs[prev_print_batch:-1]) / print_interval))
            prev_print_batch = batch_num

        # Checkpoint model
        if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):
            print("Saving checkpoint!")
            save_checkpoint(model, total_examples / batch_size,
                            losses, costs, seq_lens,
                            total_examples, None, num_inputs,
                            num_outputs, None, None,
                            None, None, None,
                            batch_size, cuda, hidden_dim,
                            num_layers, 'LSTM')
        if total_examples / checkpoint_interval >= total_batches:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NTM', help='"NTM" or "LSTM" (baseline)')
    parser.add_argument('--learn_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--M', type=int, default=20, help='memory feature size')
    parser.add_argument('--N', type=int, default=128, help='memory size')
    parser.add_argument('--num_inputs', type=int, default=9, help='number of inputs in NTM')
    parser.add_argument('--num_outputs', type=int, default=9, help='number of outputs in NTM')
    parser.add_argument('--controller_size', type=int, default=100, help='size of controller output of NTM')
    parser.add_argument('--controller_type', type=str, default="LSTM", help='type of controller of NTM')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--controller_layers', type=int, default=1, help='number of layers of controller of NTM')
    parser.add_argument('--integer_shift', type=int, default=3, help='integer shift in location attention of NTM')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='intervals to checkpoint')
    parser.add_argument('--print_interval', type=int, default=100, help='intervals to checkpoint')
    parser.add_argument('--total_batches', type=int, default=40, help='total number of batches to iterate through')
    parser.add_argument('--model_file', type=str, default='None', help='model file to load')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units in the baseline LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (LSTM) baseline)')
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    # Train NTM
    if args.model == 'NTM':
        run(learning_rate=args.learn_rate, batch_size=args.batch_size, cuda=args.cuda, memory_feature_size=args.M,
            num_inputs=args.num_inputs, num_outputs=args.num_outputs, controller_size=args.controller_size,
            controller_type=args.controller_type, controller_layers=args.controller_layers, memory_size=args.N,
            integer_shift=args.integer_shift, checkpoint_interval=args.checkpoint_interval,
            total_batches=args.total_batches, model_file=args.model_file)

    # Train LSTM (baseline)
    elif args.model == 'LSTM':
        train_baseline(learning_rate=args.learn_rate,
                       batch_size=args.batch_size,
                       cuda=args.cuda,
                       num_inputs=args.num_inputs,
                       num_outputs=args.num_outputs,
                       hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers,
                       checkpoint_interval=args.checkpoint_interval,
                       print_interval=args.print_interval,
                       total_batches=args.total_batches,
                       model_file=args.model_file)
