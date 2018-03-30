import torch
from model import NTM
from lstm_baseline import LSTM
from training_dataset import sequence_loader
import matplotlib.pyplot as plt


def visualize_sequence(checkpoint, model_type='NTM', cuda=False, seq_len=20):
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
        batch_size = 2

        model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                    controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                    memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                    use_cuda=False)
        model.load_state_dict(state_dict)

        dataloader = sequence_loader(num_batches=1, batch_size=batch_size, min_length=seq_len - 1, max_length=seq_len)
        batch_num, x, y, dummy = next(dataloader)

        # TODO
        # for batch in dataset:
        #    batch = Variable(torch.FloatTensor(batch))
        #    if cuda:
        #        batch = batch.cuda()
        #    next_r = model.read_head.create_state(batch_size, memory_feature_size)
        #    if controller_type == 'LSTM':
        #        lstm_h, lstm_c = model.controller.create_state(1)
    #
    #    output = Variable(torch.zeros(batch.size()))
    #    if cuda:
    #        output = output.cuda()
    #    for i in range(batch.size()[2]):
    #        x = batch[:, :, i]
    #        if controller_type == 'LSTM':
    #            output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r,
    #                                                                    lstm_h=lstm_h, lstm_c=lstm_c)
    #        elif controller_type == 'MLP':
    #            output[:, :, i], next_r = model.forward(x=x, r=next_r)
    #    break

    if model_type == "LSTM":
        if not cuda:  # load to CPU
            from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            state_dict = from_before['state_dict']
            num_inputs = from_before['num_inputs']
            hidden_dim = from_before['hidden_dim']
            num_layers = from_before['num_layers']
            num_outputs = from_before['num_outputs']
            batch_size = 1
        model = LSTM(num_inputs, hidden_dim, num_layers)
        model.load_state_dict(state_dict)

        dataloader = sequence_loader(num_batches=1, batch_size=batch_size, min_length=seq_len - 1, max_length=seq_len)
        x, y, dummy = next(dataloader)

        model.init_hidden(batch_size, cuda=False)
        model.forward(x)

        output = model.forward(dummy)

        # Putting the matrices together for nice display, with empty_rows between the two plots
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(seq_len / 6, 4))
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


def plot_cost(checkpoint, label, spacing=1000, fig=None, ax=None, figsize=(12, 7)):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    costs = np.array(from_before['cost'])
    x_axis = np.arange(0, len(costs), spacing)
    costs = costs.reshape(-1, spacing).mean(axis=1)
    ax.plot(x_axis / 1000, costs, linestyle='-', marker='o', c='midnightblue', label=label)

    ax.set_xlabel('Sequence number (thousands)', fontsize=13)
    ax.set_ylabel('Cost per sequence (bits)', fontsize=13)
    ax.legend()


def plot_loss(checkpoint, label, spacing=1000, fig=None, ax=None, figsize=(12, 7)):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    losses = np.array(from_before['loss'])
    x_axis = np.arange(0, len(losses), spacing)
    losses = losses.reshape(-1, spacing).mean(axis=1)
    ax.plot(x_axis / 1000, losses, linestyle='-', marker='o', c='midnightblue', label=label)

    ax.set_xlabel('Sequence number (thousands)', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.legend()

