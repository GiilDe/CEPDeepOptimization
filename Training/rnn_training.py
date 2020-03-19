from Training import processing_utilities


def get_batch_RNN(sequences, labels, batch_size, normalize):
    batch_x = []
    batch_y = []
    i = 0
    finished = False
    while i < batch_size:
        x_line, y_line = sequences.readline(), labels.readline()
        if len(x_line) == 0:
            finished = True
            break
        if bad_x_line(x_line):
            continue
        x, y = getXYRNN(x_line, y_line, normalize)
        batch_x.append(x)
        batch_y.append(y)
        i += 1

    batch_x = torch.stack(batch_x, dim=1) if batch_x != [] else []
    batch_y = torch.stack(batch_y) if batch_y != [] else []
    return batch_x, batch_y, finished


def getXYRNN(x_line, y_line, normalize=False):
    x_temp = x_line.split(";")[:-1]
    before_events = [convert_values_event(
        processing_utilities.get_event_from_str(event, *Constants.synthetic_with_values_format)) for event in x_temp[:Constants.max_seq_RNN + 1]]
    after_events = list(reversed([convert_values_event(
        processing_utilities.get_event_from_str(event, *Constants.synthetic_with_values_format)) for event in x_temp[Constants.max_seq_RNN + 1:]]))
    x = Variable(torch.stack(before_events + after_events), requires_grad=True)
    y = Variable(torch.tensor([float(y_line.split("\n")[0])]))
    if normalize:
        y = torch.tanh(y)
    return x, y