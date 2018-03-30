from model import NTM
from lstm_baseline import LSTM
import numpy as np
from torch.autograd import Variable
import torch
from training_dataset import sequence_loader
from train_utils import save_checkpoint, evaluate, evaluate_lstm_baseline
import argparse
from training_dataset import sequence_loader


def train_ntm(learning_rate, batch_size, cuda, memory_feature_size, num_inputs, num_outputs,
              controller_size, controller_type, controller_layers, memory_size, integer_shift,
              checkpoint_interval, print_interval, total_batches, model_file):
    # Seeding
    SEED = 1000
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Model Loading
    if model_file == 'None':
        model = NTM(num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    controller_size=controller_size,
                    controller_type=controller_type,
                    controller_layers=controller_layers,
                    memory_size=memory_size,
                    memory_feature_size=memory_feature_size,
                    integer_shift=integer_shift,
                    batch_size=batch_size,
                    use_cuda=cuda)
        # Constants for keeping track
        total_examples = 0
        losses = []
        costs = []
        seq_lens = []
        prev_print_batch = 0  # For printing purposes
        final_checkpoint_batch = 0
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
        model = NTM(num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    controller_size=controller_size,
                    controller_type=controller_type,
                    controller_layers=controller_layers,
                    memory_size=memory_size,
                    memory_feature_size=memory_feature_size,
                    integer_shift=integer_shift,
                    batch_size=batch_size,
                    use_cuda=cuda)
        model.load_state_dict(state_dict)
        losses = from_before['loss']
        costs = from_before['cost']
        seq_lens = from_before['seq_lengths']
        total_examples = from_before['total_examples']
        final_checkpoint_batch = len(losses)
        prev_print_batch = 0
    if cuda:
        model.cuda()

    # Dataset creation
    dataloader = sequence_loader(num_batches=total_batches, batch_size=batch_size, max_length=20)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, alpha=0.95)
    criterion = torch.nn.BCELoss()

    np.random.seed(SEED)  # reset training seed to ensure that batches remain the same between runs!
    for batch_num, (x, y, dummy) in enumerate(dataloader):
        input_seq_length = x.shape[0]
        output_seq_length = y.shape[0]
        output = Variable(torch.zeros(y.size()))

        if cuda:
            x = x.cuda()
            y = y.cuda()
            dummy = dummy.cuda()
            output = output.cuda()

        next_r = model.read_head.create_state(batch_size, memory_feature_size)
        if controller_type == 'LSTM':
            lstm_h, lstm_c = model.controller.create_state(batch_size)

        # Forward pass
        for i in range(input_seq_length):
            if controller_type == 'LSTM':
                _, next_r, lstm_h, lstm_c = model.forward(x=x[i], r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                _, next_r = model.forward(x=x[i], r=next_r)
        # Get output
        for i in range(output_seq_length):
            if controller_type == 'LSTM':
                output[i], next_r, lstm_h, lstm_c = model.forward(x=dummy[i], r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                output[i], next_r = model(x=dummy[i], r=next_r)

        # Backward pass
        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward(retain_graph=True)

        # Clip Gradient between [10, 10].
        parameters = list(filter(lambda p: p.grad is not None, ntm.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)

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
            prev_print_batch = batch_num + final_checkpoint_batch

        # Checkpoint model
        if (checkpoint_interval != 0) and (total_examples % checkpoint_interval == 0):
            print("Saving checkpoint!")
            save_checkpoint(model, total_examples / batch_size,
                            losses, costs, seq_lens,
                            total_examples, controller_type, num_inputs,
                            num_outputs, controller_size, controller_layers,
                            memory_size, memory_feature_size, integer_shift,
                            batch_size, cuda, None,
                            None, 'NTM')
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
        prev_print_batch = 0  # For printing purposes
        final_checkpoint_batch = 0
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
        final_checkpoint_batch = len(losses)
        prev_print_batch = 0
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
            prev_print_batch = batch_num + final_checkpoint_batch

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
    parser.add_argument('--learn_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--M', type=int, default=128, help='memory feature size')
    parser.add_argument('--N', type=int, default=20, help='memory size')
    parser.add_argument('--num_inputs', type=int, default=9, help='number of inputs in NTM')
    parser.add_argument('--num_outputs', type=int, default=8, help='number of outputs in NTM')
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
        train_ntm(learning_rate=args.learn_rate,
                  batch_size=args.batch_size,
                  cuda=args.cuda,
                  memory_feature_size=args.M,
                  num_inputs=args.num_inputs,
                  num_outputs=args.num_outputs,
                  controller_size=args.controller_size,
                  controller_type=args.controller_type,
                  controller_layers=args.controller_layers,
                  memory_size=args.N,
                  integer_shift=args.integer_shift,
                  checkpoint_interval=args.checkpoint_interval,
                  print_interval=args.print_interval,
                  total_batches=args.total_batches,
                  model_file=args.model_file)

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
