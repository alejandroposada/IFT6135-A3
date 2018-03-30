import torch
from torch.autograd import Variable
from model import NTM
from lstm_baseline import LSTM
from training_dataset import random_binary
import numpy as np
import matplotlib.pyplot as plt


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
            memory_feature_size = from_before['memory_feature_size']
            integer_shift = from_before['integer_shift']
            batch_size = 2

            model = NTM(num_inputs=num_inputs, num_outputs=num_outputs, controller_size=controller_size,
                        controller_type=controller_type, controller_layers=controller_layers, memory_size=memory_size,
                        memory_feature_size=memory_feature_size, integer_shift=integer_shift, batch_size=batch_size,
                        use_cuda=cuda)
            # model.load_state_dict(state_dict)

            dataset = random_binary(max_seq_length=seq_len, num_sequences=1, vector_dim=8,
                                    batch_Size=batch_size, min_seq_length=seq_len - 1)


            for batch in dataset:
                batch = Variable(torch.FloatTensor(batch))
                if cuda:
                    batch = batch.cuda()
                next_r = model.read_head.create_state(batch_size, memory_feature_size)
                if controller_type == 'LSTM':
                    lstm_h, lstm_c = model.controller.create_state(1)

                output = Variable(torch.zeros(batch.size()))
                if cuda:
                    output = output.cuda()
                for i in range(batch.size()[2]):
                    x = batch[:, :, i]
                    if controller_type == 'LSTM':
                        output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r,
                                                                                lstm_h=lstm_h, lstm_c=lstm_c)
                    elif controller_type == 'MLP':
                        output[:, :, i], next_r = model.forward(x=x, r=next_r)
                break

        X = from_before['cost']
        plt.plot(X)
        plt.ylabel('Cost')
        plt.xlabel('Number of batches (16)')
        plt.show()

    elif model_type == "LSTM":
        if not cuda:  # load to CPU
            from_before = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            state_dict = from_before['state_dict']
            num_inputs = from_before['num_inputs']
            num_hidden = from_before['num_hidden']
            num_outputs = from_before['num_outputs']
            batch_size = 2
        model = LSTM(num_inputs, num_hidden)
        model.load_state_dict(state_dict)

        dataset = random_binary(max_seq_length=seq_len, num_sequences=1, vector_dim=8,
                                batch_Size=batch_size, min_seq_length=seq_len - 1)
        criterion = torch.nn.BCELoss()

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

        X = batch.squeeze(0).data.numpy()
        Y = output.squeeze(0).data.numpy()

        # Putting the matrices together for nice display, with empty_rows between the two plots
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(seq_len/6, 4))

        empty_rows = 2
        data = np.ones((X.shape[1] + Y.shape[1] + empty_rows, X.shape[2])) * 0.5
        x = X[0]
        y = Y[0]

        ax[0].imshow(x, cmap='binary', interpolation='nearest', aspect='auto')
        ax[0].set_ylabel('Target', rotation=0, labelpad=30, fontsize=13)
        im = ax[1].imshow(y, cmap='binary', interpolation='nearest', aspect='auto')
        ax[1].set_ylabel('Output', rotation=0, labelpad=30, fontsize=13)

        fig.colorbar(im, ax=ax.ravel().tolist())
        for ax_ in ax:
            ax_.set_xticks([])
            ax_.set_yticks([])
        plt.show()

checkpoint = "checkpoints/ntm/copy-batch-2112.0--LSTM.model"
visualize_sequence(checkpoint, model_type='NTM', cuda=False, seq_len=40)
