import torch
import constants
from torch.autograd import Variable
import processing_utilities
import pandas as pd


def getX(x_line):
    x_temp = x_line.split(";")[:-1]
    events = [processing_utilities.convert_event(processing_utilities.get_event_from_str(event, *constants.event_format))
              for event in x_temp]
    x = Variable(torch.stack(events), requires_grad=True).reshape(-1)
    return x


def get_batch(sequences, batch_size=32):
    batch_x = []
    finished = False
    i = 0
    while i < batch_size:
        x_line = sequences.readline()
        if x_line in constants.stop_lines:
            finished = True
            break
        x = getX(x_line)
        batch_x.append(x)
        i += 1

    batch_x = torch.stack(batch_x) if batch_x != [] else None
    return batch_x, finished


def get_sequence(sequences):
    x_line = sequences.readline()
    if x_line in constants.stop_lines:
        return None, True
    return getX(x_line), False


def infer():
    net = torch.load(constants.model_path)
    net.to(device=constants.device)
    net.eval()
    sequences = open("training data/FOR_TEST_SEQS - Copy.txt", 'r')
    df = pd.DataFrame()
    for _ in range(constants.window_limit[0] - 1):
        df = df.append(pd.Series(-1), ignore_index=True)
    x, almost_finished = get_batch(sequences)
    i = 0
    finished = False
    while not finished and x is not None:
        y_hat = net.forward(x)
        for row in y_hat:
            df = df.append(pd.Series(row.item()), ignore_index=True)
        finished = almost_finished
        x, almost_finished = get_batch(sequences)
        if i % 1000 == 0:
            print(i)
        i += 32
    for _ in range(constants.window_limit[0] - 1):
        df = df.append(pd.Series(-1), ignore_index=True)
    sequences.close()
    df.to_csv(constants.scores_path, index=False, header=False)


if __name__ == "__main__":
    infer()

