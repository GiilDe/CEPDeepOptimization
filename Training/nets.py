
import torch
import torch.nn as nn
from constants import constants
from itertools import product
import numpy as np
import time


def get_log_probs(events_probs, chosen_events, use_unchosen_probs=False):
    if use_unchosen_probs:
        flipped_chosen_events = 1 - chosen_events
        flipped_probs = torch.abs(flipped_chosen_events - events_probs)  # flip unchosen probabilities
        flipped_log_probs = torch.log(flipped_probs)
        log_probs = torch.sum(flipped_log_probs, dim=1)  # dims: (batch_size, 1)
    else:
        log_probs = torch.log(events_probs)
        masked_log_probs = log_probs * chosen_events
        log_probs = torch.sum(masked_log_probs, dim=1)
    return log_probs


def sample_events(events_probs):
    batch_size = events_probs.size(0)
    chosen_events = torch.empty_like(events_probs).int()

    for batch_i, event_i in product(range(batch_size), range(constants['window_size'])):
        choice = np.random.choice([0, 1], size=None, p=[1 - events_probs[batch_i, event_i].item(),
                                                        events_probs[batch_i, event_i].item()])

        chosen_events[batch_i, event_i] = torch.tensor(choice).item()

    return chosen_events


def get_fc_layer(in_dim, out_dim, use_dropout):
    fc = nn.Linear(in_dim, out_dim)
    b_norm = nn.BatchNorm1d(out_dim)
    relu = nn.ReLU()
    modules = [fc, b_norm, relu]
    if use_dropout:
        drop_out = nn.Dropout()
        modules.append(drop_out)
    return modules


class ConvWindowToFilters(nn.Module):
    def __init__(self, batch_size, use_dropout):
        super(ConvWindowToFilters, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(constants['event_size'], 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(6, 5, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(5, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 3, kernel_size=3),
            nn.ReLU(inplace=True),
        ).double()
        fc_mods = []
        fc_mods += get_fc_layer(276, 230, use_dropout)
        fc_mods += get_fc_layer(230, 180, use_dropout)
        fc_mods += get_fc_layer(180, 150, use_dropout)
        fc_mods += get_fc_layer(150, 130, use_dropout)
        fc = nn.Linear(130, constants['window_size'])
        fc_mods.append(fc)
        sigmoid = nn.Sigmoid()
        fc_mods.append(sigmoid)
        self.fc = nn.Sequential(*fc_mods).double()
        self.batch_size = batch_size

    def forward(self, events, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        events = events.transpose(1, 2)
        t1 = time.perf_counter()
        features = self.conv(events)
        features = features.reshape((batch_size, 276)).double()
        probs = self.fc(features)
        t2 = time.perf_counter()
        return probs, t2 - t1

    def __str__(self):
        return "conv"

    @staticmethod
    def get_log_probs(events_probs, chosen_events, use_unchosen_probs=False):
        return get_log_probs(events_probs, chosen_events, use_unchosen_probs)

    @staticmethod
    def sample_events(events_probs):
        return sample_events(events_probs)


class LinearWindowToFilters(nn.Module):
    def __init__(self, batch_size, use_dropout=False):
        super(LinearWindowToFilters, self).__init__()
        modules = []
        # constants['event_size'] * constants['window_size'] = 150
        modules += get_fc_layer(constants['event_size'] * constants['window_size'], 750, use_dropout)
        modules += get_fc_layer(750, 600, use_dropout)
        modules += get_fc_layer(600, 450, use_dropout)
        modules += get_fc_layer(450, 300, use_dropout)
        modules += get_fc_layer(300, 200, use_dropout)
        modules += get_fc_layer(200, 150, use_dropout)
        modules += [nn.Linear(150, constants['window_size'])]
        modules += [nn.Sigmoid()]
        self.probs_net = nn.Sequential(*modules)
        self.probs_net.double()
        self.batch_size = batch_size

    def forward(self, events, batch_size=None):
        # events/events_probs dims: (batch_size, window_size)
        if batch_size is None:
            batch_size = self.batch_size
        events = events.reshape((batch_size, constants['window_size'] * constants['event_size']))
        t1 = time.perf_counter()
        events_probs = self.probs_net(events)
        t2 = time.perf_counter()
        return events_probs, t2 - t1

    def __str__(self):
        return "linear"

    @staticmethod
    def get_log_probs(events_probs, chosen_events, use_unchosen_probs=False):
        return get_log_probs(events_probs, chosen_events, use_unchosen_probs)

    @staticmethod
    def sample_events(events_probs):
        return sample_events(events_probs)
