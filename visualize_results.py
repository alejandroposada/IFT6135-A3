import torch
from torch.autograd import Variable
from model import NTM
from lstm_baseline import LSTM, LSTM_v2
from training_dataset import random_binary, sequence_loader
from train_utils import evaluate, evaluate_lstm_baseline_v2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm_notebook


def visualize_sequence(checkpoint, model_type='NTM', cuda=False, seq_len=100):
    if model_type == 'NTM':
        if not cuda:  # load to CPU
            from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            state_dict = from_before['state_dict']
            controller_type = from_before['controller_type']
            num_inputs = from_before['num_inputs']
            num_outputs = from_before['num_outputs']
            controller_size = from_before['controller_size']
            controller_layers = from_before['controller_layers']
            memory_size = from_before['memory_size']
            batch_size = from_before['batch_size']
            batch_size = 2
            memory_feature_size = from_before['memory_feature_size']
            integer_shift = from_before['integer_shift']
            saved_biases = True

            model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                        controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                        memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                        use_cuda=cuda, saved_biases=saved_biases)
            model.load_state_dict(state_dict)

            dataset = random_binary(max_seq_length=seq_len, num_sequences=1, vector_dim=8,
                                    batch_Size=batch_size, min_seq_length=seq_len - 1)

            for batch in dataset:
                batch = Variable(batch)
                model.init_headweights()
                model.init_memory()

                if cuda:
                    batch = batch.cuda()
                next_r = model.read_head.create_state(batch_size)
                if controller_type == 'LSTM':
                    lstm_h, lstm_c = model.controller.create_state(batch_size)

                for i in range(batch.size()[2]):
                    x = batch[:, :, i]
                    if controller_type == 'LSTM':
                        _, next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
                    elif controller_type == 'MLP':
                        _, next_r = model.forward(x=x, r=next_r)

                # Read output without input
                x = Variable(torch.zeros(batch.size()[0:2]))
                output = Variable(torch.zeros(batch[:, :, :-1].size()))
                if cuda:
                    output = output.cuda()
                for i in range(output.size()[2]):
                    if controller_type == 'LSTM':
                        output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h,
                                                                                lstm_c=lstm_c)
                    elif controller_type == 'MLP':
                        output[:, :, i], next_r = model.forward(x=x, r=next_r)
                output = output[0, :-1, :]  # Only one batch
                batch = batch[0, :-1, :-1]
                break

    elif model_type == "LSTM":
        if not cuda:  # load to CPU
            from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            state_dict = from_before['state_dict']
            num_inputs = from_before['num_inputs']
            # num_hidden = from_before['num_hidden']
            num_outputs = from_before['num_outputs']
            batch_size = 1
        model = LSTM(num_inputs, 100)
        model.load_state_dict(state_dict)

        dataset = random_binary(max_seq_length=seq_len, num_sequences=1, vector_dim=8,
                                batch_Size=batch_size, min_seq_length=seq_len - 1)

        for batch in dataset:
            model.init_hidden(batch_size)
            batch = Variable(torch.FloatTensor(batch))
            if cuda:
                batch = batch.cuda()
            output = Variable(torch.zeros(batch.size()))
            if cuda:
                output = output.cuda()
            for i in range(batch.size()[2]):
                x = batch[:, :, i]
                output[:, :, i] = model.forward(x)
            break

    x = batch.squeeze(0).data.numpy()
    y = output.squeeze(0).data.numpy()

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(seq_len / 6, 4))
    fig.suptitle('$T = {}$'.format(seq_len))

    ax[0].imshow(x, cmap='binary', interpolation='nearest', aspect='auto')
    ax[0].set_ylabel('Target', rotation=0, labelpad=30, fontsize=13)
    im = ax[1].imshow(y, cmap='binary', interpolation='nearest', aspect='auto')
    ax[1].set_ylabel('Output', rotation=0, labelpad=30, fontsize=13)

    fig.colorbar(im, ax=ax.ravel().tolist())
    for ax_ in ax:
        ax_.set_xticks([])
        ax_.set_yticks([])
    plt.show()


def plot_cost(checkpoint, label, spacing=1000, batch_size=1, fig=None, ax=None, figsize=(12, 7), marker='o'):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    costs = np.array(from_before['cost'])
    x_axis = np.arange(0, len(costs), spacing)
    costs = costs.reshape(-1, spacing).mean(axis=1)
    ax.plot(x_axis * batch_size / 1000, costs, linestyle='-', marker=marker, label=label)

    ax.set_xlabel('Sequence number (thousands)', fontsize=13)
    ax.set_ylabel('Cost per sequence (bits)', fontsize=13)
    ax.legend()


def plot_loss(checkpoint, label, spacing=1000, batch_size=1, fig=None, ax=None, figsize=(12, 7), marker='o'):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    costs = np.array(from_before['loss'])
    x_axis = np.arange(0, len(costs), spacing)
    costs = costs.reshape(-1, spacing).mean(axis=1)
    ax.plot(x_axis * batch_size / 1000, costs, linestyle='-', marker=marker, label=label)

    ax.set_xlabel('Sequence number (thousands)', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.legend()


def load_model(checkpoint, model_type='NTM'):
    if model_type == 'NTM':
        from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = from_before['state_dict']
        controller_type = from_before['controller_type']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        controller_size = from_before['controller_size']
        controller_layers = from_before['controller_layers']
        memory_size = from_before['memory_size']
        batch_size = from_before['batch_size']
        memory_feature_size = from_before['memory_feature_size']
        integer_shift = from_before['integer_shift']
        batch_size = 1
        saved_biases = True
        model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                    controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                    memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                    use_cuda=False, saved_biases=saved_biases)
        model.load_state_dict(state_dict)

    elif model_type == "LSTM":
        from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = from_before['state_dict']
        num_inputs = from_before['num_inputs']
        num_hidden = from_before['num_hidden']
        num_outputs = from_before['num_outputs']
        batch_size = 1
        model = LSTM(num_inputs, num_hidden)
        model.load_state_dict(state_dict)
        model.init_hidden(batch_size)
    return model


def visualize_sequence_v2(checkpoint, model_type='NTM', cuda=False, seq_len=20):
    """
	For models trained with the second version of the dataloader.
	"""
    if model_type == 'NTM':
        from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = from_before['state_dict']
        controller_type = from_before['controller_type']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        controller_size = from_before['controller_size']
        controller_layers = from_before['controller_layers']
        memory_size = from_before['memory_size']
        batch_size = from_before['batch_size']
        memory_feature_size = from_before['memory_feature_size']
        integer_shift = from_before['integer_shift']
        batch_size = 1
        model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                    controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                    memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                    use_cuda=False)
        model.load_state_dict(state_dict)
        model.init_memory()
        model.init_headweights()

        dataloader = sequence_loader(num_batches=1, batch_size=batch_size, min_length=seq_len - 1, max_length=seq_len)
        x, y, dummy = next(dataloader)

        input_seq_length = x.shape[0]
        output_seq_length = y.shape[0]
        output = Variable(torch.zeros(y.size()))

        next_r = model.read_head.create_state(batch_size)
        if controller_type == 'LSTM':
            lstm_h, lstm_c = model.controller.create_state(batch_size)

        # Forward pass
        for i in range(input_seq_length):
            if controller_type == 'LSTM':
                _, next_r, lstm_h, lstm_c = model(x=x[i], r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
            elif controller_type == 'MLP':
                _, next_r = model.forward(x=x[i], r=next_r)
        # Get output
        for i in range(output_seq_length):
            if controller_type == 'LSTM':
                a, next_r, lstm_h, lstm_c = model(x=dummy[i], r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)
                output[i] = a[:, :]  # TODO
            elif controller_type == 'MLP':
                output[i], next_r = model(x=dummy[i], r=next_r)
        print(output.shape)

    elif model_type == "LSTM":
        if not cuda:  # load to CPU
            from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            state_dict = from_before['state_dict']
            num_inputs = from_before['num_inputs']
            hidden_dim = from_before['hidden_dim']
            num_layers = from_before['num_layers']
            num_outputs = from_before['num_outputs']
            batch_size = 1
        model = LSTM_v2(num_inputs, hidden_dim, num_layers)
        model.load_state_dict(state_dict)
        model.init_hidden(batch_size, cuda)

        dataloader = sequence_loader(num_batches=1, batch_size=batch_size, min_length=seq_len - 1, max_length=seq_len)
        x, y, dummy = next(dataloader)

        model.forward(x)

        output = model.forward(dummy)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(seq_len / 6, 4))
    fig.suptitle('$T = {}$'.format(seq_len))
    target_np = x.data.numpy().T.squeeze()
    output_np = output.data.numpy().T.squeeze()

    ax[0].imshow(target_np[:-1, :], cmap='binary', aspect='auto')
    ax[0].set_ylabel('Target', rotation=0, labelpad=30, fontsize=13)
    im = ax[1].imshow(output_np, cmap='binary', interpolation='nearest', aspect='auto')
    ax[1].set_ylabel('Output', rotation=0, labelpad=30, fontsize=13)

    fig.colorbar(im, ax=ax.ravel().tolist())
    for ax_ in ax:
        ax_.set_xticks([])
        ax_.set_yticks([])
    plt.show()


def load_model_v2(checkpoint, model_type='NTM'):
    if model_type == 'NTM':
        from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = from_before['state_dict']
        controller_type = from_before['controller_type']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        controller_size = from_before['controller_size']
        controller_layers = from_before['controller_layers']
        memory_size = from_before['memory_size']
        batch_size = from_before['batch_size']
        memory_feature_size = from_before['memory_feature_size']
        integer_shift = from_before['integer_shift']
        batch_size = 1
        saved_biases = True
        model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                    controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                    memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                    use_cuda=False, saved_biases=saved_biases)
        model.load_state_dict(state_dict)

    elif model_type == "LSTM":
        from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        state_dict = from_before['state_dict']
        num_inputs = from_before['num_inputs']
        num_outputs = from_before['num_outputs']
        batch_size = 1
        model = LSTM_v2(num_inputs, 100)
        model.load_state_dict(state_dict)
        model.init_hidden(batch_size, cuda=False)
    return model, controller_type


def generalization_test():
    lengths = np.arange(10, 101, 10)
    costs = {'lstm': [], 'ntm_lstm': [], 'ntm_mlp': []}

    # Load trained models
    ntm_lstm = load_model('checkpoints/ntm/copy-batch-1125.0--LSTM.model', 'NTM')
    ntm_mlp = load_model('checkpoints/ntm/copy-batch-7500.0--MLP.model', 'NTM')
    lstm, _ = load_model_v2('checkpoints/lstm/copy-batch-1000000.0.model', model_type='LSTM')

    # Average over 20 runs
    for T in tqdm_notebook(lengths):
        dataloader = random_binary(max_seq_length=T, num_sequences=None, batch_Size=1, min_seq_length=T - 1)
        cost, _, _ = evaluate(ntm_lstm, dataloader, 1, 'LSTM', False, how_many=20)
        costs['ntm_lstm'].append(cost)

        cost, _, _ = evaluate(ntm_mlp, dataloader, 1, 'MLP', False, how_many=20)
        costs['ntm_mlp'].append(cost)

        dataloader = sequence_loader(100, batch_size=1, min_length=T - 1, max_length=T)
        cost, _, _ = evaluate_lstm_baseline_v2(lstm, dataloader, 1, False)
        costs['lstm'].append(cost)
    return costs, lengths


def visualize_heads(checkpoint, seq_len=128, model_type='NTM'):
    model, controller_type = load_model_v2(checkpoint=checkpoint, model_type=model_type)
    batch_size = 2
    dataset = random_binary(max_seq_length=seq_len+1, num_sequences=1, vector_dim=8,
                            batch_Size=batch_size, min_seq_length=seq_len)

    for batch in dataset:
        batch = Variable(batch)
        model.init_headweights()
        model.init_memory()

        next_r = model.read_head.create_state(batch_size)
        if controller_type == 'LSTM':
            lstm_h, lstm_c = model.controller.create_state(batch_size)

        for i in range(batch.size()[2]):
            x = batch[:, :, i]
            if controller_type == 'LSTM':
                _, next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c,
                                                          vis_heads=True)
            elif controller_type == 'MLP':
                _, next_r = model.forward(x=x, r=next_r, vis_heads=True)

        # Read output without input
        x = Variable(torch.zeros(batch.size()[0:2]))
        output = Variable(torch.zeros(batch[:, :, :-1].size()))
        for i in range(output.size()[2]):
            if controller_type == 'LSTM':
                output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h,
                                                                        lstm_c=lstm_c, vis_heads=True)
            elif controller_type == 'MLP':
                output[:, :, i], next_r = model.forward(x=x, r=next_r, vis_heads=True)
        output = output[0, :, :]  # Only one batch
        batch = batch[0, :, :]
        break

    # Creating input/output matrices
    x = batch.squeeze(0).data.numpy()
    y = output.squeeze(0).data.numpy()
    input = np.zeros((len(x), len(x[0])+len(y[0])))
    input[:, :len(x[0])] = x
    output = np.zeros((len(x), len(x[0]) + len(y[0])))
    output[:, len(x[0]):] = y

    # Preparing Read/Write head weightings
    read_heads = torch.zeros(len(model.read_heads[0][0]), seq_len*2 + 1)
    write_heads = torch.zeros(len(model.write_heads[0][0]), seq_len*2 + 1)
    for idx in range(len(model.read_heads)):
        read_heads[:, idx] = model.read_heads[idx].data[0]
        write_heads[:, idx] = model.write_heads[idx].data[0]
    read_state = read_heads.numpy()
    write_state = write_heads.numpy()

    # Plotting the grid of results
    plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3])

    ax = plt.subplot(gs[0, 0])
    ax.imshow(input, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Inputs')

    ax = plt.subplot(gs[0, 1])
    ax.imshow(output, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Outputs')

    ax = plt.subplot(gs[1, 0])
    ax.imshow(write_state, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('<------------- Location In Memory ------------->')
    ax.set_xlabel('Time Step -------------------------------->')
    ax.set_title('Write Head Weights')

    ax = plt.subplot(gs[1, 1])
    ax.imshow(read_state, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Time Step -------------------------------->')
    ax.set_title('Read Head Weights')

    plt.tight_layout()
    plt.show()

checkpoint = 'checkpoints/ntm/copy-batch-1125.0--LSTM.model'
visualize_sequence(checkpoint, model_type='NTM', seq_len=128)
