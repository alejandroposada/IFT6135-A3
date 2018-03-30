import torch
from torch.autograd import Variable
import torch.nn as nn


def save_checkpoint(model, batch_num, losses, costs, seq_lengths, total_examples, controller_type,
                    num_inputs, num_outputs, controller_size, controller_layers, memory_size,
                    memory_feature_size, integer_shift, batch_size, cuda, hidden_dim=None,
                    num_layers=None, model_type='NTM'):
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
            'hidden_dim': hidden_dim,
            'num_outputs': num_outputs,
            'num_layers': num_layers,
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
            model.forward(x)

        # Output response
        x = Variable(torch.zeros(batch.size()[0:2]) + 0.5)
        if cuda:
            x = x.cuda()
        for i in range(batch.size()[2]):
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


# Taken from https://github.com/kuc2477/pytorch-ntm/blob/7f5fa872f08c0214ccf69a35aeba286ff76890b6/utils.py
def xavier_init(model, uniform=False):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        nn.init.xavier_normal(p) if uniform else nn.init.xavier_normal(p)


def clip_gradients(net, range=10):
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-range, range)