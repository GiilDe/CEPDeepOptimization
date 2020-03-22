import torch
import constants
from torch.autograd import Variable
import processing_utilities


def getXY(x_line):
    x_temp = x_line.split(";")[:-1]
    events = [processing_utilities.convert_event(processing_utilities.get_event_from_str(event, *constants.event_format))
              for event in x_temp]
    x = Variable(torch.stack(events), requires_grad=True).reshape(-1)
    return x


def get_sequence(sequences):
    x_line = sequences.readline()
    if x_line in {"$", "$\n", " ", "\n", "", "-1", "-1\n"}:
        return None, True
    return getXY(x_line), False


def infer():
    net = torch.load(constants.model_path)
    net.eval()
    scores_file = open(constants.scores_path, 'w')
    sequences = open(constants.test_file_path_sequences, 'r')
    for _ in range(constants.window_limit[0]):
        sequences.readline()
        scores_file.write(str(-1) + "\n")
    x, finished = get_sequence(sequences)
    i = 0
    while not finished and type(x) != str:
        x = x.reshape((1, -1))
        y_hat = net.forward(x)
        scores_file.write(str(max(y_hat.item(), 0)) + "\n")
        x, finished = get_sequence(sequences)
        if i % 1000 == 0:
            print(i)
        i += 1
    for _ in range(constants.window_limit[0]):
        scores_file.write(str(-1) + "\n")
    scores_file.close()
    sequences.close()


if __name__ == "__main__":
    infer()

