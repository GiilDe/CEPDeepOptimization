import torch
from torch.utils.data import IterableDataset
import pandas as pd
from constants import constants
import numpy as np


class Iterable:
    def __init__(self, X, M):
        self.X = X
        self.M = M

    def __next__(self):
        return self.get_events(self.X), self.get_batch_matches(self.M)

    @staticmethod
    def get_batch_matches(M):
        matches = next(M)
        return torch.tensor(matches.to_numpy())

    @staticmethod
    def get_events(X):
        def get_dummies(x):
            padding = pd.DataFrame([['A', -1], ['B', -1], ['C', -1], ['D', -1]])
            x = padding.append(x, ignore_index=True)
            x = pd.get_dummies(x, columns=[0])
            x = x.drop(axis=0, labels=[0, 1, 2, 3])
            return x
        events = next(X)
        events = get_dummies(events)
        events = torch.tensor(np.transpose(events.to_numpy()), requires_grad=True).double()
        return events


class Dataset(IterableDataset):
    def __init__(self, is_train: str, repeat_batch):
        super(Dataset, self).__init__()
        self.is_train = is_train

        x_path = constants['train_stream_path'] if is_train == "train" else constants['test_stream_path']
        self.X = pd.read_csv(x_path, chunksize=constants['window_size'], header=None, usecols=[0, 1])

        m_path = constants['train_matches'] if is_train == "train" else constants['test_matches']
        self.M = pd.read_csv(m_path, chunksize=1, header=None)

        self.repeat_batch = repeat_batch

    def __len__(self):
        size_ = constants['train_size'] if self.is_train == "train" else constants['test_size']
        return int(size_*self.repeat_batch/constants['window_size'])

    def __iter__(self):
        return Iterable(self.X, self.M)


FULL_WINDOW_COMPLEXITY = \
    2**(constants['pattern_window_size'])*(constants['window_size'] - constants['pattern_window_size'] + 1)
UNFOUND_MATCHES_PENALTY = 6
REQUIRED_MATCHES_PORTION = 0.8


def get_rewards(matches, chosen_events):
    batch_matches_sum = 0
    batch_found_matches_sum = 0
    batch_chosen_events_num = 0

    def get_window_complexity_ratio(i):
        batch_chosen_events = chosen_events[i]
        selected = torch.zeros(constants['window_size'] + 1).bool()
        selected[batch_chosen_events] = 1
        pattern_window_size = constants['pattern_window_size']
        window_complexity = 0
        pattern_window_selected = 0

        for i in range(pattern_window_size):
            pattern_window_selected += int(selected[i].item())

        window_complexity += 2**pattern_window_selected

        for i in range(1, constants['window_size'] + 1 - pattern_window_size):
            pattern_window_selected -= int(selected[i].item())
            pattern_window_selected += int(selected[i + pattern_window_size].item())
            window_complexity += 2**pattern_window_selected

        return window_complexity/FULL_WINDOW_COMPLEXITY

    def get_window_matches(i):
        batch_matches = matches[i].tolist()
        batch_chosen_events = chosen_events[i] - 1

        non_chosen = set(range(constants['window_size'])) - set(batch_chosen_events.tolist())

        matches_ = []
        for i in range(0, len(batch_matches), constants['match_size']):
            if batch_matches[i] == -1:
                break
            match = set(batch_matches[i:i + constants['match_size']])
            matches_.append(match)

        matches_num = len(matches_)
        filtered = 0
        for match in matches_:
            for count in match:
                if count in non_chosen:
                    filtered += 1
                    break

        found_matches_num = matches_num - filtered
        chosen_events_num = constants['window_size'] - len(non_chosen)

        return matches_num, found_matches_num, chosen_events_num

    def unfound_match_penalty(matches_ratio):
        # gap = max(REQUIRED_MATCHES_PORTION * matches_num - found_matches_num, 0)
        # return UNFOUND_MATCHES_PENALTY * gap
        return UNFOUND_MATCHES_PENALTY * max(REQUIRED_MATCHES_PORTION - matches_ratio, 0)

    rewards = []
    for i in range(len(chosen_events)):
        # window_complexity_ratio = max(get_window_complexity_ratio(i), 0.005)
        matches_num, found_matches_num, chosen_events_num = get_window_matches(i)

        batch_matches_sum += matches_num
        batch_found_matches_sum += found_matches_num
        batch_chosen_events_num += chosen_events_num

        window_complexity_ratio = max(chosen_events_num, 1)/constants['window_size']

        matches_ratio = found_matches_num/matches_num if matches_num != 0 else 1

        ratio = matches_ratio/window_complexity_ratio
        penalty = unfound_match_penalty(matches_ratio)

        reward = ratio - penalty
        reward = reward * (-1)  # reward is actually loss
        rewards.append(reward)

    rewards = torch.tensor(rewards)
    chosen_events_num = batch_chosen_events_num/len(chosen_events)

    return rewards, batch_found_matches_sum, batch_matches_sum, chosen_events_num

