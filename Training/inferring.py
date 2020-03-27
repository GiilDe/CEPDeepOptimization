import torch
import constants
from torch.autograd import Variable
import pandas as pd


def getX(row):
    events = []
    i = 0
    while i < len(row):
        type = constants.type_to_vec[row[i]]
        events = events + type
        events = events + list(row[i + 1:i + len(constants.event_format[0]) - 1])
        i += len(constants.event_format[0])
    x = Variable(torch.tensor(events), requires_grad=True)
    return x


def get_batch(sequences):
    try:
        batch = next(sequences).to_numpy()
        batch_x = []
        for row in batch:
            batch_x.append(getX(row))
        batch_x = torch.stack(batch_x)
        return batch_x
    except StopIteration:
        return None


def infer(batch_size=32):
    net = torch.load(constants.model_path)
    net.to(device=constants.device)
    net.eval()
    sequences = pd.read_csv(constants.test_file_path_sequences, chunksize=batch_size, header=None)
    scores = pd.DataFrame()
    for _ in range(constants.window_limit[0] - 1):
        scores = scores.append(pd.Series(-1), ignore_index=True)
    i = 0
    x = get_batch(sequences)
    while x is not None:
        y_hat = net.forward(x).data.numpy().reshape(-1)
        for row in y_hat:
            scores = scores.append(pd.Series(row.item()), ignore_index=True)
        if i % 1000 == 0:
            print(i)
        i += 32
        x = get_batch(sequences)

    for _ in range(constants.window_limit[0] - 1):
        scores = scores.append(pd.Series(-1), ignore_index=True)
    sequences.close()
    scores.to_csv(constants.scores_path, index=False, header=False)


if __name__ == "__main__":
    infer()

