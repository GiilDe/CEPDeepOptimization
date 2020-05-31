from constants import constants
import torch
import pandas as pd
import numpy as np
import typing
import OpenCEP

time_calc_types = ["steps_calculation", "time_measurement", "complexity_calculation", "event_num"]
time_calc_index = 3

allow_gpu = True
dev = "cuda" if allow_gpu and torch.cuda.is_available() else "cpu"

device = torch.device(dev)

batch_size = 128

UNFOUND_MATCHES_PENALTY = 15
REQUIRED_MATCHES_PORTION = 0.6

FULL_WINDOW_COMPLEXITY = \
    2 ** (constants['pattern_window_size']) * (constants['window_size'] - constants['pattern_window_size'] + 1)

convert_type = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], range(1, 9)))


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
        padding = pd.DataFrame([['A', -1], ['B', -1], ['C', -1], ['D', -1], ['E', -1], ['F', -1], ['G', -1], ['H', -1]])
        x = padding.append(x, ignore_index=True)
        x = pd.get_dummies(x)
        x = x.drop(axis=0, labels=[0, 1, 2, 3, 4, 5, 6, 7])
        return x
    try:
        batch = next(X)
        batch = get_dummies(batch)
        batch = torch.tensor(batch.to_numpy(), dtype=torch.float64, requires_grad=True, device=device) \
            .reshape((batch_size, constants['window_size'], constants['event_size']))
        return batch
    except (StopIteration, RuntimeError):
        return None


def get_batch_events_non_onehot(X):
    try:
        batch = next(X)
        batch.iloc[:, 0] = batch.iloc[:, 0].apply(lambda event_type: convert_type[event_type])
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


def condition1(A: OpenCEP.processing_utilities.Event, B: OpenCEP.processing_utilities.Event) -> bool:
    return A.value < B.value


condition = OpenCEP.processing_utilities.Condition(condition1, [0, 1])

event_types = ['A', 'B']
event_types_with_identifiers = \
    [OpenCEP.processing_utilities.EventTypeOrPatternAndIdentifier(type, i) for i, type in enumerate(event_types)]
seq_event_pattern = OpenCEP.processing_utilities. \
    EventPattern(event_types_with_identifiers, OpenCEP.processing_utilities.Seq(range(len(event_types))))
seq_pattern_query = OpenCEP.processing_utilities. \
    CleanPatternQuery(seq_event_pattern, [condition], time_limit=constants['pattern_window_size'])

cep_processor = OpenCEP.processor.TimeCalcProcessor(['count', 'type', 'value'], 0, 1, [seq_pattern_query])


def get_rewards(matches: typing.List, chosen_events: np.ndarray,
                window_events: typing.Union[None, pd.DataFrame] = None):
    def get_window_complexity_ratio(i):
        batch_chosen_events = chosen_events[i]
        pattern_window_size = constants['pattern_window_size']

        pattern_window_selected = np.sum(batch_chosen_events[:pattern_window_size])
        window_complexity = 2 ** pattern_window_selected

        for i in range(pattern_window_size, constants['window_size']):
            pattern_window_selected -= batch_chosen_events[i - pattern_window_size]
            pattern_window_selected += batch_chosen_events[i]
            window_complexity += 2 ** pattern_window_selected

        return window_complexity / FULL_WINDOW_COMPLEXITY

    def get_window_event_num_ratio(i):
        batch_chosen_events = chosen_events[i]
        batch_chosen_events_num = np.sum(batch_chosen_events)
        return max(batch_chosen_events_num, 1) / constants['window_size']

    def get_window_time_ratio(i):
        filtered_window, window = get_window(i)

        whole_time, _ = cep_processor.query(window)
        filtered_time, _ = cep_processor.query(filtered_window)

        return filtered_time / whole_time

    def get_window_steps_ratio(i):
        filtered_window, window = get_window(i)

        _, whole_steps = cep_processor.query(window)
        _, filtered_steps = cep_processor.query(filtered_window)

        return filtered_steps / whole_steps

    def get_window(i):
        batch_chosen_events = chosen_events[i].astype(bool)
        batch_events = window_events.iloc[i * constants['window_size']:(i + 1) * constants['window_size']]
        filtered_batch_events = batch_events.iloc[batch_chosen_events] if any(batch_chosen_events) \
            else batch_events.iloc[0]
        window = batch_events.to_string(header=False).replace("  ", ",").replace(" ", "")
        filtered_window = filtered_batch_events.to_string(header=False).replace("  ", ",").replace(" ", "")
        return filtered_window, window

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

    get_denominator_fun = [get_window_steps_ratio, get_window_time_ratio, get_window_complexity_ratio,
                           get_window_event_num_ratio]
    found_matches_portions = []
    rewards = []
    matches_sum = 0
    found_matches_sum = 0
    denominator = 0
    for i in range(batch_size):
        # window_complexity_ratio = max(get_window_complexity_ratio(i), 0.005)
        time_func = get_denominator_fun[time_calc_index]
        window_complexity_ratio = time_func(i)
        denominator += window_complexity_ratio
        matches_num, found_matches_num = get_window_matches(i)
        matches_sum += matches_num
        found_matches_sum += found_matches_num
        found_matches_portion = found_matches_num / matches_num if matches_num != 0 else 1
        found_matches_portions.append(found_matches_portion)

        ratio = found_matches_portion / window_complexity_ratio
        penalty = unfound_match_penalty(found_matches_portion)

        reward = ratio - penalty
        rewards.append(reward)

    rewards = torch.tensor(rewards, device=device)
    chosen_events_num = np.sum(chosen_events, axis=1)
    actual_found_matches_portion = found_matches_sum / matches_sum
    denominator = denominator / batch_size
    return rewards, chosen_events_num, found_matches_portions, actual_found_matches_portion, denominator


