import torch
import json
from torch.autograd import Variable
import numpy as np


# Code snippet taken from:
# https://github.com/loudinthecloud/pytorch-ntm/blob/db56fb86f9f68abb799ff9120f9beda64837bece/train.py#L67
def save_checkpoint(model, batch_num, losses, costs, seq_lengths):
    basename = "checkpoints/copy-batch-{}".format(batch_num)

    model_fname = basename + ".model"
    torch.save(model.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def evaluate(model, testset, batch_size, memory_M):
    count = 0  # Ugly, I know...
    total_cost = 0
    for batch in testset:
        batch = Variable(torch.FloatTensor(batch))
        next_r = Variable(torch.FloatTensor(np.random.rand(batch_size, memory_M) * 0.05))
        lstm_h, lstm_c = model.controller.create_state(batch_size)

        output = Variable(torch.zeros(batch.size()))
        for i in range(batch.size()[2]):
            x = batch[:, :, i]
            output[:, :, i], next_r, lstm_h, lstm_c = model.forward(x=x, r=next_r, lstm_h=lstm_h, lstm_c=lstm_c)

        # The cost is the number of error bits per sequence
        binary_output = output.clone().data
        binary_output.apply_(lambda y: 0 if y < 0.5 else 1)
        cost = torch.sum(torch.abs(binary_output - batch.data))
        total_cost += cost / batch_size

        count += 1
        if count >= 4:
            break

    return cost/(count-1), binary_output, batch


