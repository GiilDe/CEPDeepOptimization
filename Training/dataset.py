from constants import constants
import torch
import pandas as pd
import numpy as np
import typing

allow_gpu = True
dev = "cuda" if allow_gpu and torch.cuda.is_available() else "cpu"

device = torch.device(dev)

batch_size = 128

UNFOUND_MATCHES_PENALTY = 3
REQUIRED_MATCHES_PORTION = 0.6

FULL_WINDOW_COMPLEXITY = \
    2**(constants['pattern_window_size'])*(constants['window_size'] - constants['pattern_window_size'] + 1)


def get_batch_matches(M):
    try:
        batches = []
        for _ in range(batch_size):
            batch = M.readline()
            if batch == "":
                raise StopIteration
            batch = batch.split(",")
            for i, obj_i in enumerate(batch):
                batch[i] = int(obj_i)
            batches.append(batch)
        return batches
    except (StopIteration, RuntimeError):
        return None


def get_batch_events(X):
    def get_dummies(x):
        padding = pd.DataFrame([['A', -1], ['B', -1], ['C', -1], ['D', -1]])
        x = padding.append(x, ignore_index=True)
        x = pd.get_dummies(x)
        x = x.drop(axis=0, labels=[0, 1, 2, 3])
        return x
    try:
        batch = next(X)
        batch = get_dummies(batch)
        batch = torch.tensor(batch.to_numpy(), dtype=torch.float64, requires_grad=True, device=device) \
            .reshape((batch_size, constants['window_size'], constants['event_size']))
        return batch
    except (StopIteration, RuntimeError):
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


def get_rewards(matches: typing.List, chosen_events: np.ndarray):

    def get_window_complexity_ratio(i):
        batch_chosen_events = chosen_events[i]
        pattern_window_size = constants['pattern_window_size']

        pattern_window_selected = np.sum(batch_chosen_events[:pattern_window_size])
        window_complexity = 2**pattern_window_selected

        for i in range(pattern_window_size, constants['window_size']):
            pattern_window_selected -= batch_chosen_events[i - pattern_window_size]
            pattern_window_selected += batch_chosen_events[i]
            window_complexity += 2**pattern_window_selected

        return window_complexity/FULL_WINDOW_COMPLEXITY
        # batch_chosen_events_num = np.sum(batch_chosen_events)
        # return max(batch_chosen_events_num, 1)/constants['window_size']

    def get_window_matches(i):
        batch_matches = matches[i]
        batch_chosen_events = chosen_events[i]
        non_chosen = set()
        for i in range(constants['window_size']):
            if batch_chosen_events[i] == 0:
                non_chosen.add(i)
        matches_ = []
        for i in range(0, len(batch_matches), constants['match_size']):
            if batch_matches[i] == -1:
                break
            match = set(batch_matches[i:i + constants['match_size']])
            matches_.append(match)

        matches_num = len(matches_)
        filtered_matches_num = 0
        for match in matches_:
            for count in match:
                if count in non_chosen:
                    filtered_matches_num += 1
                    break

        found_matches_num = matches_num - filtered_matches_num
        return matches_num, found_matches_num

    def unfound_match_penalty(matches_ratio):
        return UNFOUND_MATCHES_PENALTY * max(REQUIRED_MATCHES_PORTION - matches_ratio, 0)

    found_matches_portions = []
    rewards = []
    matches_sum = 0
    found_matches_sum = 0
    for i in range(batch_size):
        # window_complexity_ratio = max(get_window_complexity_ratio(i), 0.005)
        window_complexity_ratio = get_window_complexity_ratio(i)
        matches_num, found_matches_num = get_window_matches(i)
        matches_sum += matches_num
        found_matches_sum += found_matches_num
        found_matches_portion = found_matches_num/matches_num if matches_num != 0 else 1
        found_matches_portions.append(found_matches_portion)

        ratio = found_matches_portion/window_complexity_ratio
        penalty = unfound_match_penalty(found_matches_portion)

        reward = ratio - penalty
        rewards.append(reward)

    rewards = torch.tensor(rewards, device=device)
    chosen_events_num = np.sum(chosen_events, axis=1)
    actual_found_matches_portion = found_matches_sum/matches_sum
    return rewards, chosen_events_num, found_matches_portions, actual_found_matches_portion
