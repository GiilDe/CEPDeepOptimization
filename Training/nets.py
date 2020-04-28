import torch
import torch.nn as nn
from constants import constants
from training import batch_size, event_size
from itertools import product


class WindowToFiltersFC(nn.Module):
    def __init__(self):
        super(WindowToFiltersFC, self).__init__()
        fc1 = nn.Linear(event_size, 40)
        b_norm1 = nn.BatchNorm1d(constants['window_size'])
        relu1 = nn.ReLU()
        drop1 = nn.Dropout()
        fc2 = nn.Linear(40, 20)
        b_norm2 = nn.BatchNorm1d(constants['window_size'])
        relu2 = nn.ReLU()
        drop2 = nn.Dropout()
        fc3 = nn.Linear(20, 7)
        b_norm3 = nn.BatchNorm1d(constants['window_size'])
        relu3 = nn.ReLU()
        drop3 = nn.Dropout()
        fc4 = nn.Linear(7, 1)
        sigmoid = nn.Sigmoid()
        modules = [fc1, b_norm1, relu1, drop1, fc2, b_norm2, relu2, drop2, fc3, b_norm3, relu3, drop3, fc4, sigmoid]
        self.network = nn.Sequential(*modules)

    def forward(self, events):
        # events dimensions: (batch_size, window_size, event_size)
        # output dimensions: (batch_size, window_size, 1)
        return self.network(events)


class FilteredWindowToScoreFC(nn.Module):
    def __init__(self, automatic_filtering=False):
        super(FilteredWindowToScoreFC, self).__init__()
        fc1 = nn.Linear(constants['window_size']*(event_size + 1), 40)
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

    def forward(self, events_filtering):
        # events dimensions: (batch_size, window_size, event_size)
        # filtering dimensions: (batch_size, window_size, 1)
        # x dimensions: (batch_size, window_size, event_size + 1)
        # output dimensions: (batch_size, 1)
        events = events_filtering[0]
        filtering = events_filtering[1]
        x = torch.cat((events, filtering), dim=2)
        if self.automatic_filtering:
            for i, j in product(range(batch_size), range(constants['window_size'])):
                if x[i, j, event_size] < 0.5:
                    x[i, j, :-1] = 0
        x = x.reshape((batch_size, constants['window_size']*(event_size + 1)))
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
