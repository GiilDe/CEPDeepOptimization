from constants import constants
import torch
import pandas as pd

batch_size = 8

allow_gpu = True
dev = "cuda" if allow_gpu and torch.cuda.is_available() else "cpu"

device = torch.device(dev)
convert_type = dict(zip(constants['event_types'], range(1, len(constants['event_types']) + 1)))


def get_batch_matches(M):
    class Match:
        def __init__(self, idxs):
            for i, obj_i in enumerate(idxs):
                idxs[i] = int(obj_i) + 1
            idxs.sort()
            self.idxs = idxs

        def __lt__(self, other):
            i = 0
            while self.idxs[i] == other.idxs[i]:
                i += 1
            return self.idxs[i] < other.idxs[i]

    batches = []
    max_size = 0
    for _ in range(batch_size):
        batch = M.readline()
        if batch == "":
            return None
        batch = batch.replace('\n', '').split(",")
        batch_len = len(batch)
        if batch_len > max_size:
            max_size = batch_len
        t = []
        if len(batch) > 1:
            for i in range(0, len(batch), constants['match_size']):
                t.append(Match(batch[i:i + constants['match_size']]))
        t.sort()
        t_ = []
        for match in t:
            t_ += match.idxs
        t_ += [0]
        batches.append(t_)
    max_size += 1
    for i in range(len(batches)):
        enlarge_size = max_size - len(batches[i])
        batches[i] = torch.tensor(batches[i] + ([0] * enlarge_size))
    batches_ = torch.stack(batches)
    return batches_


def get_batch_events(X):
    def get_dummies(x):
        padding = pd.DataFrame([[type_, -1] for type_ in constants['event_types']])
        x = padding.append(x, ignore_index=True)
        x = pd.get_dummies(x)
        x = x.drop(axis=0, labels=list(range(len(constants['event_types']))))
        return x
    try:
        batch_ = None
        batch_ = next(X)
        batch = get_dummies(batch_)
        batch = torch.tensor(batch.to_numpy(), dtype=torch.float64, requires_grad=True, device=device) \
            .reshape((batch_size, constants['window_size'], constants['event_size']))
        return batch
    except (StopIteration, RuntimeError):
        print(batch_)
        return None


def get_batch_events_as_events(X):
    try:
        batch = next(X)
        return batch
    except StopIteration:
        return None


def initialize_data_x(is_train):
    X = pd.read_csv(constants['train_stream_path'] if is_train else constants['test_stream_path'],
                    chunksize=batch_size * constants['window_size'], header=None, usecols=[0, 1])
    return X


def initialize_data_matches(is_train):
    M = open(constants['train_matches'] if is_train else constants['test_matches'], "r")
    return M
