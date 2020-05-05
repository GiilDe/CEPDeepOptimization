import torch
import torch.nn as nn
from constants import constants
from training import event_size
from itertools import product
import numpy as np

batch_size = 32


class WindowToFiltersReinforce(nn.Module):
    def __init__(self, batch_size):
        super(WindowToFiltersReinforce, self).__init__()
        self.probs_net = WindowToFiltersFC(batch_size)
        self.batch_size = batch_size

    def forward(self, events):
        # events/events_probs dims: (batch_size, window_size)
        events_probs = self.probs_net(events)
        chosen_events = torch.zeros_like(events_probs)
        chosen_events_np = np.zeros((self.batch_size, constants['window_size']))

        for batch_i, event_i in product(range(self.batch_size), range(constants['window_size'])):
            choice = np.random.choice([0, 1], size=None, p=[events_probs[batch_i, event_i].item(),
                                                            1 - events_probs[batch_i, event_i].item()])
            # choice = 0 event is chosen, choice = 1 event is not chosen
            chosen_events[batch_i, event_i] = torch.tensor(choice).item()
            chosen_events_np[batch_i, event_i] = 1 - choice  # for output reasons: flip choice

        flipped_probs = torch.abs(chosen_events - events_probs)  # flip unchosen probabilities
        windows_probs = torch.prod(flipped_probs, dim=1)
        log_prob = torch.log(windows_probs)  # dims: (batch_size, 1)
        return chosen_events_np, log_prob


class WindowToFiltersFC(nn.Module):
    def __init__(self, batch_size):
        super(WindowToFiltersFC, self).__init__()
        fc1 = nn.Linear(event_size*constants['window_size'], 50)
        b_norm1 = nn.BatchNorm1d(50)
        relu1 = nn.ReLU()
        drop1 = nn.Dropout()
        fc2 = nn.Linear(50, 30)
        b_norm2 = nn.BatchNorm1d(30)
        relu2 = nn.ReLU()
        drop2 = nn.Dropout()
        fc3 = nn.Linear(30, 20)
        b_norm3 = nn.BatchNorm1d(20)
        relu3 = nn.ReLU()
        drop3 = nn.Dropout()
        fc4 = nn.Linear(20, 15)
        relu4 = nn.ReLU()
        drop4 = nn.Dropout()
        fc5 = nn.Linear(15, constants['window_size'])
        sigmoid = nn.Sigmoid()
        modules = [fc1, b_norm1, relu1, fc2, b_norm2, relu2, fc3, b_norm3, relu3, fc4, relu4, fc5, sigmoid]
        self.network = nn.Sequential(*modules)
        self.network.double()
        self.batch_size = batch_size

    def forward(self, events):
        # events dimensions: (batch_size, window_size, event_size)
        # output dimensions: (batch_size, window_size)
        x = events.reshape((self.batch_size, constants['window_size'] * event_size))
        return self.network(x)


def filter_events(events, filtering):
    new_events = torch.zeros_like(events, requires_grad=False)
    for i, j in product(range(batch_size), range(constants['window_size'])):
        if filtering[i, j, 0] > 0.5:
            new_events[i, j, :-1] = events[i, j, :-1]
    new_events.requires_grad = True
    return new_events


class FilteredWindowToScoreFC(nn.Module):
    def __init__(self, keep_filterings=False, automatic_filtering=False):
        super(FilteredWindowToScoreFC, self).__init__()
        size_ = (event_size + 1) if keep_filterings else event_size
        fc1 = nn.Linear(constants['window_size'] * size_, 40)
        b_norm1 = nn.BatchNorm1d(40)
        relu1 = nn.ReLU()
        drop1 = nn.Dropout()
        fc2 = nn.Linear(40, 30)
        b_norm2 = nn.BatchNorm1d(30)
        relu2 = nn.ReLU()
        drop2 = nn.Dropout()
        fc3 = nn.Linear(30, 15)
        b_norm3 = nn.BatchNorm1d(15)
        relu3 = nn.ReLU()
        drop3 = nn.Dropout()
        fc4 = nn.Linear(15, 7)
        b_norm4 = nn.BatchNorm1d(7)
        relu4 = nn.ReLU()
        drop4 = nn.Dropout()
        fc5 = nn.Linear(7, 1)
        modules = [fc1, b_norm1, relu1, drop1, fc2, b_norm2, relu2, drop2, fc3, b_norm3, relu3, drop3, fc4, b_norm4,
                   relu4, drop4, fc5]
        self.network = nn.Sequential(*modules)
        self.automatic_filtering = automatic_filtering
        self.keep_filterings = keep_filterings

    def forward(self, events_filtering):
        # events dimensions: (batch_size, window_size, event_size)
        # filtering dimensions: (batch_size, window_size, 1)
        # x dimensions: (batch_size, window_size, event_size + 1)
        # output dimensions: (batch_size, 1)
        events = events_filtering[0]
        filtering = events_filtering[1]
        if self.automatic_filtering:
            events = filter_events(events, filtering)
        x = torch.cat((events, filtering), dim=2) if self.keep_filterings else events
        size_ = (event_size + 1) if self.keep_filterings else event_size
        x = x.reshape((batch_size, constants['window_size'] * size_))
        return self.network(x)


class WindowToScoreFC(nn.Module):
    def __init__(self, filtered_window_to_score: FilteredWindowToScoreFC):
        super(WindowToScoreFC, self).__init__()
        self.window_to_filters = WindowToFiltersFC()
        self.filtered_window_to_score = filtered_window_to_score

    def forward(self, events):
        # events dimensions: (batch_size, window_size, event_size)
        # output dimensions: (batch_size, 1)
        filters = self.window_to_filters(events)
        score = self.filtered_window_to_score((events, filters))
        return score
