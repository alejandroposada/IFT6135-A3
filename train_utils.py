import torch
from torch.autograd import Variable


def save_checkpoint(model, batch_num, losses, costs, seq_lengths, total_examples, controller_type,
                    num_inputs, num_outputs, controller_size, controller_layers, memory_size,
                    memory_feature_size, integer_shift, batch_size, cuda, num_hidden=None, model_type='NTM'):
    if model_type == 'NTM':
        basename = "checkpoints/ntm/copy-batch-{}--{}".format(batch_num, controller_type)
        model_fname = basename + ".model"
        state = {
            'state_dict': model.state_dict(),
            'loss': losses,
            'cost': costs,
            'seq_lengths': seq_lengths,
            'total_examples': total_examples,
            'controller_type': controller_type,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'controller_size': controller_size,
            'controller_layers': controller_layers,
            'memory_size': memory_size,
            'memory_feature_size': memory_feature_size,
            'integer_shift': integer_shift,
            'batch_size': batch_size,
            'cuda': cuda
        }
    elif model_type == 'LSTM':
        basename = "checkpoints/lstm/copy-batch-{}".format(batch_num)
        model_fname = basename + ".model"
        state = {
            'state_dict': model.state_dict(),
            'loss': losses,
            'cost': costs,
            'seq_lengths': seq_lengths,
            'total_examples': total_examples,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'batch_size': batch_size,
            'cuda': cuda
        }
    torch.save(state, model_fname)


def evaluate(model, testset, batch_size, controller_type, cuda, memory_feature_size):

    count = 0  # Ugly, I know...
    total_cost = 0
    for batch in testset:
        batch = Variable(batch)

        if cuda:
            batch = batch.cuda()
        next_r = model.read_head.create_state(batch_size, memory_feature_size)
        if controller_type == 'LSTM':
            lstm_h, lstm_c = model.controller.create_state(batch_size)
        output = Variable(torch.zeros(batch.size()))
        if cuda:
            output = output.cuda()

        for i in range(batch.size()[2]):
            x = batch[:, :, i]
            if controller_type == 'LSTM':
                output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                output[:, :, i], next_r = model.forward(x=x, r=next_r)

        # The cost is the number of error bits per sequence
        binary_output = output.clone().data
        binary_output = binary_output > 0.5
        # binary_output.apply_(lambda y: 0 if y < 0.5 else 1)
        cost = torch.sum(torch.abs(binary_output.float() - batch.data))
        total_cost += cost / batch_size

        count += 1
        if count >= 4:
            break

    return cost/(count-1), binary_output, batch


def evaluate_lstm_baseline(model, testset, batch_size, cuda):
    count = 0
    total_cost = 0
    for batch in testset:
        batch = Variable(batch)

        if cuda:
            batch = batch.cuda()
        output = Variable(torch.zeros(batch.size()))
        if cuda:
            output = output.cuda()

        for i in range(batch.size()[2]):
            x = batch[:, :, i]
            output[:, :, i] = model.forward(x)

        # The cost is the number of error bits per sequence
        binary_output = output.clone().data
        binary_output = binary_output > 0.5
        # binary_output.apply_(lambda y: 0 if y < 0.5 else 1)
        cost = torch.sum(torch.abs(binary_output.float() - batch.data))
        total_cost += cost / batch_size

        count += 1
        if count >= 4:
            break
    return cost / (count - 1), binary_output, batch