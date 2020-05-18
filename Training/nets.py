import torch
import torch.nn as nn
from constants import constants
from training import event_size
from itertools import product
import numpy as np


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
        log_probs = torch.log(windows_probs)  # dims: (batch_size, 1)

        # log_probs = torch.log(events_probs)
        # masked_log_probs = log_probs * torch.tensor(chosen_events_np).int()
        # log_prob = torch.sum(masked_log_probs, dim=1)
        
        return chosen_events_np, log_probs


class WindowToFiltersFC(nn.Module):
    def __init__(self, batch_size):
        super(WindowToFiltersFC, self).__init__()
        fc1 = nn.Linear(constants['event_size']*constants['window_size'], 50)
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


