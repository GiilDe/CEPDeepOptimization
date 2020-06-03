
import torch
import torch.nn as nn
from constants import constants
from itertools import product
import numpy as np
import time


def get_fc_layer(in_dim, out_dim, use_dropout):
    fc = nn.Linear(in_dim, out_dim)
    b_norm = nn.BatchNorm1d(out_dim)
    relu = nn.ReLU()
    modules = [fc, b_norm, relu]
    if use_dropout:
        drop_out = nn.Dropout()
        modules.append(drop_out)
    return modules


def sample_events(events_probs, batch_size, use_unchosen_probs=True):
    chosen_events = torch.zeros_like(events_probs)
    chosen_events_np = np.zeros((batch_size, constants['window_size']))

    for batch_i, event_i in product(range(batch_size), range(constants['window_size'])):
        choice = np.random.choice([0, 1], size=None, p=[events_probs[batch_i, event_i].item(),
                                                        1 - events_probs[batch_i, event_i].item()])
        # choice = 0 event is chosen, choice = 1 event is not chosen
        chosen_events[batch_i, event_i] = torch.tensor(choice).item()
        chosen_events_np[batch_i, event_i] = 1 - choice  # for output reasons: flip choice

    if use_unchosen_probs:
        flipped_probs = torch.abs(chosen_events - events_probs)  # flip unchosen probabilities
        windows_probs = torch.prod(flipped_probs, dim=1)
        log_probs = torch.log(windows_probs)  # dims: (batch_size, 1)
    else:
        log_probs = torch.log(events_probs)
        masked_log_probs = log_probs * torch.tensor(chosen_events_np).int()
        log_probs = torch.sum(masked_log_probs, dim=1)

    return chosen_events_np, log_probs


class DummyModel(nn.Module):
    def __init__(self, batch_size):
        super(DummyModel, self).__init__()
        self.batch_size = batch_size

    def forward(self, events):
        # chosen_events = np.ones((self.batch_size, constants['window_size']))
        # for i in range(self.batch_size):
        #     chosen_events[i, :4] = 0
        #     chosen_events[i, -4:] = 0

        chosen_events = np.zeros((self.batch_size, constants['window_size']))
        for i in range(self.batch_size):
            batch_chosen_events = np.random.choice(a=range(constants['window_size']), size=20, replace=False)
            chosen_events[i, batch_chosen_events] = 1

        return chosen_events, torch.full(size=(self.batch_size,), fill_value=-2)


class ConvWindowToFilters(nn.Module):
    def __init__(self, batch_size, use_dropout):
        super(ConvWindowToFilters, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(constants['event_size'], 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(6, 5, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(5, 4, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 2, kernel_size=5),
            nn.ReLU(inplace=True),
        ).double()
        fc_mods = []
        fc_mods += get_fc_layer(468, 400, use_dropout)
        fc_mods += get_fc_layer(400, 350, use_dropout)
        fc_mods += get_fc_layer(350, 300, use_dropout)
        fc = nn.Linear(300, constants['window_size'])
        fc_mods.append(fc)
        sigmoid = nn.Sigmoid()
        fc_mods.append(sigmoid)
        self.fc = nn.Sequential(*fc_mods).double()
        self.batch_size = batch_size

    def forward(self, events):
        events = events.transpose(1, 2)
        t1 = time.perf_counter()
        features = self.conv(events)
        features = features.reshape((self.batch_size, 468)).double()
        probs = self.fc(features)
        t2 = time.perf_counter()
        chosen_events, log_probs = sample_events(probs, self.batch_size)
        return chosen_events, log_probs, t2 - t1


class LinearWindowToFilters(nn.Module):
    def __init__(self, batch_size):
        super(LinearWindowToFilters, self).__init__()
        self.probs_net = WindowToFiltersFC(batch_size)
        self.batch_size = batch_size

    def forward(self, events):
        # events/events_probs dims: (batch_size, window_size)
        events_probs = self.probs_net(events)
        return sample_events(events_probs, self.batch_size)


class WindowToFiltersFC(nn.Module):
    def __init__(self, batch_size, use_dropout=True):
        super(WindowToFiltersFC, self).__init__()
        modules = []
        # constants['event_size'] * constants['window_size'] = 150
        modules += get_fc_layer(constants['event_size'] * constants['window_size'], 130, use_dropout)
        modules += get_fc_layer(130, 110, use_dropout)
        modules += get_fc_layer(110, 90, use_dropout)
        modules += get_fc_layer(90, 70, use_dropout)
        modules += get_fc_layer(70, 50, use_dropout)
        fc = nn.Linear(50, constants['window_size'])
        modules.append(fc)
        sigmoid = nn.Sigmoid()
        modules.append(sigmoid)
        self.network = nn.Sequential(*modules)
        self.network.double()
        self.batch_size = batch_size

    def forward(self, events):
        # events dimensions: (batch_size, window_size, event_size)
        # output dimensions: (batch_size, window_size)
        x = events.reshape((self.batch_size, constants['window_size'] * constants['event_size']))
        return self.network(x)
